"""
NativeStrategyEntry — runs the same strategy.generate_signals() logic in live
trading that the backtest path uses, so a named strategy means the same thing
both places.

Drops in alongside EntryDecider. Returns the SAME EntryDecision shape so
downstream plumbing (PositionManager, ExitManager, RiskGovernor, broker block,
sizing) keeps working without modification.

Activation: SingleStockTrader is constructed with entry_mode="native_strategy"
and a list of native_strategies. On each evaluation cycle, this helper:

  1. Maps the brain MarketStateResult.state (TREND_UP/...) to the legacy regime
     vocabulary the strategies expect (BULL_OPEN/BEAR_OPEN/CHOPPY).
  2. Calls strategy.generate_signals(df_5m, df_15m, symbol, config, regime) for
     each enabled native strategy.
  3. Filters by direction_mode (long_only / short_only / both).
  4. Picks the highest-confidence accepted signal.
  5. Converts that DayTradeSignal into an EntryDecision so the trader's existing
     sizing/execution path stays untouched.

If no strategy accepts, returns a NO_TRADE EntryDecision whose `checks` dict
includes per-strategy rejection diagnostics for the UI/decision log.
"""
from __future__ import annotations

import logging
from datetime import time
from typing import Any

import pandas as pd

from app.services.strategy.daytrading.autotrader.entry_decider import EntryDecision
from app.services.strategy.daytrading.brain.market_state import (
    CHOPPY, HIGH_VOL, NEWS_RISK, TREND_DOWN, TREND_UP, MarketStateResult,
)
from app.services.strategy.daytrading.market_open import now_et
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.strategies import STRATEGY_MAP

logger = logging.getLogger(__name__)

_NO_ENTRY_AFTER = time(15, 15)  # universal hard cutoff — matches EntryDecider

# Strategies that have been audited and verified to produce live-compatible
# signals via their generate_signals() output. Order matters only for the
# `native_candidates_considered` diagnostic list.
SUPPORTED_NATIVE_STRATEGIES: tuple[str, ...] = (
    "BollingerMomentum",
    "SupertrendTrend",
    "EMAMomentum",
    "ORBBreakout",
)


def map_brain_state_to_regime(state: str | None) -> str:
    """Bridge the brain's MarketStateResult vocab to the legacy regime string
    the strategies' generate_signals() expects.

    Brain states (TREND_UP/TREND_DOWN/CHOPPY/HIGH_VOL/NEWS_RISK/UNKNOWN) →
    legacy strings (BULL_OPEN/BEAR_OPEN/CHOPPY).
    """
    if state == TREND_UP:
        return "BULL_OPEN"
    if state == TREND_DOWN:
        return "BEAR_OPEN"
    # CHOPPY, HIGH_VOL, NEWS_RISK, UNKNOWN → CHOPPY (most conservative regime
    # for the strategies' internal filters).
    return "CHOPPY"


class NativeStrategyEntry:
    """Live-trading entry helper that delegates to strategy.generate_signals().

    Parameters
    ----------
    direction_mode : "long_only" | "short_only" | "both"
    risk_per_trade_pct : retained for parity with EntryDecider (sizing uses it)
    min_rr : minimum risk/reward; rejects strategy signals below this floor
    min_confidence : minimum confidence; rejects strategy signals below this
    native_strategies : list of strategy names to run; defaults to the audited
                        4-strategy set. Names not present in STRATEGY_MAP are
                        silently skipped.
    """

    def __init__(
        self,
        direction_mode: str = "long_only",
        risk_per_trade_pct: float = 0.01,
        min_rr: float = 1.5,
        min_confidence: float = 0.45,
        native_strategies: list[str] | None = None,
    ):
        self.direction_mode = direction_mode
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_rr = min_rr
        self.min_confidence = min_confidence
        names = native_strategies or list(SUPPORTED_NATIVE_STRATEGIES)
        self.native_strategies: list[str] = [n for n in names if n in STRATEGY_MAP]
        self._strategy_objs = [STRATEGY_MAP[n] for n in self.native_strategies]

    def decide(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        market_state: MarketStateResult | None,
        account_equity: float = 10_000.0,
    ) -> EntryDecision:
        """Mirror of EntryDecider.decide() signature so SingleStockTrader can
        call either one without branching on data plumbing.
        """
        # ── Time gate (matches EntryDecider) ──────────────────────────────
        now_t = now_et().time()
        if now_t >= _NO_ENTRY_AFTER:
            return _no_trade(
                "Too late in day — no new entries after 15:15 ET",
                entry_mode="native_strategy",
                extra_checks={"gate": "late_entry_window"},
            )

        if df_5m is None or len(df_5m) < 15:
            return _no_trade(
                "Insufficient 5m bar history",
                entry_mode="native_strategy",
                extra_checks={"gate": "insufficient_bars"},
            )

        brain_state = market_state.state if market_state else "UNKNOWN"
        if brain_state == NEWS_RISK:
            return _no_trade(
                "Market is NEWS_RISK — no entries during catalyst events",
                entry_mode="native_strategy",
                extra_checks={"gate": "news_risk", "state": brain_state},
            )

        regime = map_brain_state_to_regime(brain_state)
        can_long = self.direction_mode in ("long_only", "both")
        can_short = self.direction_mode in ("short_only", "both")

        # ── Run each native strategy ──────────────────────────────────────
        accepted: list[DayTradeSignal] = []
        rejections: dict[str, str] = {}      # human-readable per-strategy reason
        categories: dict[str, str] = {}      # stable category code for aggregation

        df_15m_safe = df_15m if df_15m is not None else pd.DataFrame()

        for strat in self._strategy_objs:
            sname = strat.name
            try:
                sigs = strat.generate_signals(
                    df_5m=df_5m,
                    df_15m=df_15m_safe,
                    symbol=symbol,
                    config=None,
                    regime=regime,
                )
            except Exception as e:
                rejections[sname] = f"exception: {type(e).__name__}: {e}"
                categories[sname] = "exception"
                logger.warning(
                    "[native_entry] %s.generate_signals raised: %s", sname, e,
                )
                continue

            if not sigs:
                rejections[sname] = "no signal at current bar"
                categories[sname] = "no_signal"
                continue

            # Strategies may return historical signals; we only act on the
            # most-recent bar to avoid replaying old setups.
            sig = sigs[-1]
            last_bar_ts = df_5m.index[-1] if len(df_5m) else None
            sig_ts = _parse_sig_time(sig.signal_time)
            if last_bar_ts is not None and sig_ts is not None:
                # Tolerate a 5m bar of slack — strategies emit at bar close.
                try:
                    bar_pd = pd.Timestamp(last_bar_ts)
                    sig_pd = pd.Timestamp(sig_ts)
                    if bar_pd.tzinfo is not None and sig_pd.tzinfo is None:
                        sig_pd = sig_pd.tz_localize(bar_pd.tzinfo)
                    elif bar_pd.tzinfo is None and sig_pd.tzinfo is not None:
                        sig_pd = sig_pd.tz_convert(None)
                    delta = abs((bar_pd - sig_pd).total_seconds())
                    if delta > 600:  # >10min stale
                        rejections[sname] = (
                            f"stale signal ({delta:.0f}s old)"
                        )
                        categories[sname] = "stale"
                        continue
                except Exception:
                    pass  # if comparison fails, accept the signal

            direction = (sig.direction or "").upper()
            if direction == "BUY" and not can_long:
                rejections[sname] = "BUY signal but direction_mode disallows longs"
                categories[sname] = "direction_disallowed"
                continue
            if direction == "SELL" and not can_short:
                rejections[sname] = "SELL signal but direction_mode disallows shorts"
                categories[sname] = "direction_disallowed"
                continue
            if direction not in ("BUY", "SELL"):
                rejections[sname] = f"unexpected direction '{direction}'"
                categories[sname] = "bad_direction"
                continue

            if sig.confidence < self.min_confidence:
                rejections[sname] = (
                    f"confidence {sig.confidence:.2f} < min {self.min_confidence}"
                )
                categories[sname] = "low_confidence"
                continue

            if sig.r_multiple and sig.r_multiple < self.min_rr:
                rejections[sname] = (
                    f"R:R {sig.r_multiple:.2f} < min {self.min_rr}"
                )
                categories[sname] = "low_rr"
                continue

            accepted.append(sig)

        # ── Resolve winner ────────────────────────────────────────────────
        if not accepted:
            return _no_trade(
                "No native strategy accepted at current bar",
                entry_mode="native_strategy",
                extra_checks={
                    "gate": "no_accepted_signal",
                    "state": brain_state,
                    "regime_mapped": regime,
                    "native_candidates_considered": list(self.native_strategies),
                    "native_rejections": rejections,
                    "native_rejection_categories": categories,
                },
            )

        winner = max(accepted, key=lambda s: s.confidence)
        action = "BUY" if winner.direction.upper() == "BUY" else "SELL_SHORT"

        # ── Sanity-check stop/target ──────────────────────────────────────
        if winner.stop_price <= 0 or winner.target_price <= 0:
            return _no_trade(
                f"{winner.strategy}: invalid stop/target prices",
                entry_mode="native_strategy",
                extra_checks={
                    "gate": "invalid_stop_target",
                    "native_rejections": rejections,
                    "native_rejection_categories": categories,
                },
            )

        if action == "BUY":
            risk = winner.entry_price - winner.stop_price
            reward = winner.target_price - winner.entry_price
        else:
            risk = winner.stop_price - winner.entry_price
            reward = winner.entry_price - winner.target_price

        if risk <= 0:
            return _no_trade(
                f"{winner.strategy}: stop {winner.stop_price:.2f} on wrong side of entry "
                f"{winner.entry_price:.2f} for {action}",
                entry_mode="native_strategy",
                extra_checks={
                    "gate": "invalid_risk",
                    "native_rejections": rejections,
                    "native_rejection_categories": categories,
                },
            )

        rr = reward / risk if risk > 0 else 0.0

        # Size multiplier: simple proportional-to-confidence rule, floored
        # at 0.25 and capped at 1.0. EntryDecider applies a more complex
        # formula but it ends in the same [0.25, 1.0] band, so this stays
        # comparable.
        size_mult = max(0.25, min(1.0, winner.confidence))

        checks: dict[str, Any] = {
            "entry_mode": "native_strategy",
            "strategy_source": "native",
            "state": brain_state,
            "regime_mapped": regime,
            "winning_strategy": winner.strategy,
            "winning_confidence": round(winner.confidence, 3),
            "winning_r_multiple": round(winner.r_multiple, 2),
            "native_candidates_considered": list(self.native_strategies),
            "native_accepted": [
                {"strategy": s.strategy, "confidence": round(s.confidence, 3)}
                for s in accepted
            ],
            "native_rejections": rejections,
            "native_rejection_categories": categories,
            "rr": round(rr, 2),
        }

        # Roll up the strategy's own indicator snapshot for the UI panel.
        if isinstance(winner.indicators, dict):
            checks["indicators"] = winner.indicators

        logger.info(
            "%s NATIVE ENTRY: %s %s @ %.2f stop=%.2f target=%.2f conf=%.2f reason=%s",
            symbol, action, winner.strategy, winner.entry_price,
            winner.stop_price, winner.target_price, winner.confidence,
            winner.reason,
        )

        return EntryDecision(
            action=action,  # type: ignore[arg-type]
            confidence=round(winner.confidence, 3),
            chosen_strategy=winner.strategy,
            entry_reason=f"[native] {winner.reason}",
            stop_price=round(winner.stop_price, 4),
            target_price=round(winner.target_price, 4),
            size_multiplier=round(size_mult, 2),
            checks=checks,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _no_trade(
    reason: str,
    *,
    entry_mode: str = "native_strategy",
    extra_checks: dict[str, Any] | None = None,
) -> EntryDecision:
    checks: dict[str, Any] = {
        "entry_mode": entry_mode,
        "strategy_source": "native",
    }
    if extra_checks:
        checks.update(extra_checks)
    return EntryDecision(
        action="NO_TRADE",
        confidence=0.0,
        chosen_strategy="",
        entry_reason=reason,
        stop_price=0.0,
        target_price=0.0,
        size_multiplier=0.0,
        checks=checks,
    )


def _parse_sig_time(sig_time: str | None) -> Any:
    if not sig_time:
        return None
    try:
        return pd.Timestamp(sig_time)
    except Exception:
        return None
