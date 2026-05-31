"""
PositionManager — enforces trade management rules once a position is open.

Rules (in order of priority):
  1. Never average down or add to a loser.
  2. Move stop to breakeven the moment the trade reaches +1R.
  3. Scale out at first target — fraction is strategy + regime dependent.
  4. Activate trailing stop at +1.25R (momentum) or +1.5R (reversion) by strategy.
  5. The trailing stop never moves backwards.
  6. At +2R on trend days, widen trail; on choppy days, tighten trail.

Regime-aware trail modes:
  TREND_UP/DOWN : ATR trail (1.5×)
  CHOPPY        : candle trail (tighter, locks in gains)
  HIGH_VOL      : EMA trail (smooth, avoids whipsaws)

Strategy-aware partial exit:
  ORBBreakout          : 25% out at first target — let the runner run
  VWAPMeanReversion    : 50% out — reversion moves often exhaust quickly
  EMAMomentum          : 33% out (default)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from app.services.strategy.daytrading.autotrader.trade_state import (
    State, TradeStateMachine,
)
from app.services.strategy.daytrading.market_open import ET, now_et

logger = logging.getLogger(__name__)

TrailMode = Literal["ema", "atr", "candle"]

# Partial-exit fraction: strategy takes precedence; regime can further modify
_STRATEGY_PARTIAL_PCT: dict[str, float] = {
    "ORBBreakout":       0.25,   # let runners run
    "VWAPMeanReversion": 0.50,   # mean-reversion exhausts quickly
    "EMAMomentum":       0.33,
}

# Regime modifier on top of strategy partial pct (additive)
_REGIME_PARTIAL_MOD: dict[str, float] = {
    "TREND_UP":   -0.10,   # take less off on trend day — let it run
    "TREND_DOWN": -0.10,
    "CHOPPY":     +0.15,   # take more off on choppy day
    "HIGH_VOL":   +0.10,
    "NEWS_RISK":  +0.25,   # take most off during news events
    "UNKNOWN":     0.00,
}

# Trail activation threshold by strategy (in R multiples)
_STRATEGY_TRAIL_THRESHOLD: dict[str, float] = {
    "ORBBreakout":       1.5,   # wait for full extension
    "VWAPMeanReversion": 1.0,   # tighter — price often stalls at VWAP
    "EMAMomentum":       1.25,
}

# Preferred trail mode by regime
_REGIME_TRAIL_MODE: dict[str, TrailMode] = {
    "TREND_UP":   "atr",
    "TREND_DOWN": "atr",
    "CHOPPY":     "candle",
    "HIGH_VOL":   "ema",
    "NEWS_RISK":  "candle",
    "UNKNOWN":    "atr",
}


@dataclass
class PositionUpdate:
    """What the PositionManager wants to do this bar."""
    action: Literal["HOLD", "MOVE_STOP", "PARTIAL_EXIT", "ACTIVATE_TRAIL"]
    new_stop: float | None = None
    exit_qty: float = 0.0
    reason: str = ""
    trail_mode_used: str = ""   # for UI explainability

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "new_stop": round(self.new_stop, 4) if self.new_stop else None,
            "exit_qty": self.exit_qty,
            "reason": self.reason,
            "trail_mode_used": self.trail_mode_used,
        }


class PositionManager:
    """
    Manages stop / target / trailing logic for one open position.

    Parameters
    ----------
    partial_tp : take partial profits at first target.
    trail_mode : override trail mode (if None, auto-selects by regime).
    partial_tp_pct : override partial exit fraction (if None, auto-selects by strategy+regime).
    """

    def __init__(
        self,
        partial_tp: bool = True,
        trail_mode: TrailMode | None = None,   # None = auto by regime
        partial_tp_pct: float | None = None,   # None = auto by strategy+regime
    ):
        self.partial_tp = partial_tp
        self._trail_mode_override = trail_mode
        self._partial_tp_pct_override = partial_tp_pct
        self._partial_taken = False

    def reset(self) -> None:
        self._partial_taken = False

    def evaluate(
        self,
        tsm: TradeStateMachine,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame | None,
        market_state_str: str = "UNKNOWN",
    ) -> PositionUpdate:
        """
        Evaluate the current bar and return what to do with the position.
        Does NOT execute — only returns instructions; the trader acts.
        """
        if not tsm.has_position:
            return PositionUpdate(action="HOLD", reason="No active position")

        close = _last_close(df_5m)
        if close <= 0:
            return PositionUpdate(action="HOLD", reason="No price data")

        tsm.update_excursion(close)

        side         = tsm.side
        entry        = tsm.entry_price
        current_stop = tsm.current_stop
        initial_stop = tsm.initial_stop
        strategy     = tsm.strategy or "EMAMomentum"

        risk_unit = (entry - initial_stop) if side == "LONG" else (initial_stop - entry)
        current_gain = (close - entry) if side == "LONG" else (entry - close)

        if risk_unit <= 0:
            return PositionUpdate(action="HOLD", reason="Risk unit is zero")

        r_multiple = current_gain / risk_unit

        # Prefer ExitPlan values when present; fall back to per-strategy tables
        exit_plan = tsm.exit_plan
        trail_mode = self._effective_trail_mode(market_state_str, exit_plan)
        partial_pct = self._effective_partial_pct(strategy, market_state_str)
        trail_threshold = (
            exit_plan.trail_trigger_r
            if exit_plan and exit_plan.trail_trigger_r > 0
            else _STRATEGY_TRAIL_THRESHOLD.get(strategy, 1.25)
        )

        # ── Breakeven stop: trigger at +1R ────────────────────────────────────
        be_stop = entry
        if r_multiple >= 1.0:
            if side == "LONG" and current_stop < be_stop:
                logger.info(
                    "%s MOVE STOP to breakeven %.4f (was %.4f, +%.2fR) [%s]",
                    tsm.symbol, be_stop, current_stop, r_multiple, market_state_str,
                )
                return PositionUpdate(
                    action="MOVE_STOP",
                    new_stop=round(be_stop, 4),
                    reason=f"+{r_multiple:.1f}R → breakeven {be_stop:.2f} [{market_state_str}]",
                )
            if side == "SHORT" and current_stop > be_stop:
                return PositionUpdate(
                    action="MOVE_STOP",
                    new_stop=round(be_stop, 4),
                    reason=f"+{r_multiple:.1f}R → breakeven {be_stop:.2f} [{market_state_str}]",
                )

        # ── Partial take-profit ───────────────────────────────────────────────
        if self.partial_tp and not self._partial_taken and r_multiple >= 1.0:
            qty_to_exit = round(tsm.qty * partial_pct, 0)
            if qty_to_exit >= 1:
                self._partial_taken = True
                logger.info(
                    "%s PARTIAL EXIT %.0f shares at +%.2fR (%.0f%% [%s/%s])",
                    tsm.symbol, qty_to_exit, r_multiple,
                    partial_pct * 100, strategy, market_state_str,
                )
                return PositionUpdate(
                    action="PARTIAL_EXIT",
                    exit_qty=qty_to_exit,
                    reason=f"Partial {partial_pct:.0%} at +{r_multiple:.1f}R [{strategy}/{market_state_str}]",
                )

        # ── Activate trailing stop ────────────────────────────────────────────
        if r_multiple >= trail_threshold and tsm.state not in (State.TRAILING, State.PARTIAL_EXIT_TAKEN):
            trail_stop = self._compute_trail(close, side, df_5m, df_1m, trail_mode)
            if trail_stop and self._is_better_stop(trail_stop, current_stop, side):
                return PositionUpdate(
                    action="ACTIVATE_TRAIL",
                    new_stop=trail_stop,
                    reason=f"Trail activated at +{r_multiple:.1f}R [{trail_mode}/{market_state_str}] = {trail_stop:.2f}",
                    trail_mode_used=trail_mode,
                )

        # ── Update trailing stop (once in TRAILING state) ─────────────────────
        if tsm.state == State.TRAILING:
            trail_stop = self._compute_trail(close, side, df_5m, df_1m, trail_mode)
            if trail_stop and self._is_better_stop(trail_stop, current_stop, side):
                logger.debug(
                    "%s TRAIL update: %.4f -> %.4f [%s]",
                    tsm.symbol, current_stop, trail_stop, trail_mode,
                )
                reason = f"Trail = {trail_stop:.2f} (+{r_multiple:.1f}R) [{trail_mode}]"
                # At +2R, apply extra-tight trail on choppy/high-vol days
                if r_multiple >= 2.0 and market_state_str in ("CHOPPY", "HIGH_VOL"):
                    tight = self._compute_trail(close, side, df_5m, df_1m, "candle")
                    if tight and self._is_better_stop(tight, trail_stop, side):
                        trail_stop = tight
                    reason += " [tight on choppy]"
                return PositionUpdate(
                    action="MOVE_STOP",
                    new_stop=trail_stop,
                    reason=reason,
                    trail_mode_used=trail_mode,
                )

        return PositionUpdate(
            action="HOLD",
            reason=f"Holding +{r_multiple:.2f}R [{market_state_str}] trail_thresh={trail_threshold:.1f}R",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _effective_trail_mode(self, market_state_str: str, exit_plan=None) -> TrailMode:
        if self._trail_mode_override:
            return self._trail_mode_override
        # ExitPlan trail type takes precedence over regime default
        if exit_plan and exit_plan.trail_type and exit_plan.trail_type != "none":
            # Map ExitPlan trail_type → PositionManager TrailMode
            _PLAN_TO_MODE: dict[str, TrailMode] = {
                "ema9_5m":          "ema",
                "ema9_15m":         "ema",
                "supertrend_5m":    "atr",   # ST trail handled by ExitManager; use ATR here
                "prior_bar_low_5m": "candle",
                "atr_fixed":        "atr",
            }
            mapped = _PLAN_TO_MODE.get(exit_plan.trail_type)
            if mapped:
                return mapped
        return _REGIME_TRAIL_MODE.get(market_state_str, "atr")

    def _effective_partial_pct(self, strategy: str, market_state_str: str) -> float:
        if self._partial_tp_pct_override is not None:
            return self._partial_tp_pct_override
        base = _STRATEGY_PARTIAL_PCT.get(strategy, 0.33)
        mod  = _REGIME_PARTIAL_MOD.get(market_state_str, 0.0)
        return round(max(0.10, min(0.75, base + mod)), 2)

    def _compute_trail(
        self,
        close: float,
        side: str,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame | None,
        trail_mode: TrailMode,
    ) -> float | None:
        try:
            if trail_mode == "ema":
                return _ema_trail(close, side, df_1m or df_5m)
            elif trail_mode == "atr":
                return _atr_trail(close, side, df_5m)
            elif trail_mode == "candle":
                return _candle_trail(close, side, df_5m)
        except Exception as e:
            logger.debug("Trail computation error: %s", e)
        return None

    @staticmethod
    def _is_better_stop(new_stop: float, current_stop: float, side: str) -> bool:
        return new_stop > current_stop if side == "LONG" else new_stop < current_stop


# ── Trail implementations ─────────────────────────────────────────────────────

def _ema_trail(close: float, side: str, df: pd.DataFrame) -> float | None:
    import ta.trend as tat
    ema9_series = tat.EMAIndicator(df["Close"], window=9).ema_indicator()
    if ema9_series.empty or pd.isna(ema9_series.iloc[-1]):
        return None
    ema9 = float(ema9_series.iloc[-1])
    atr = _quick_atr(df)
    buf = atr * 0.10
    return round(ema9 - buf, 4) if side == "LONG" else round(ema9 + buf, 4)


def _atr_trail(close: float, side: str, df: pd.DataFrame) -> float | None:
    atr = _quick_atr(df)
    if atr <= 0:
        return None
    return round(close - atr * 1.5, 4) if side == "LONG" else round(close + atr * 1.5, 4)


def _candle_trail(close: float, side: str, df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    prev_bar = df.iloc[-2]
    if side == "LONG":
        return round(float(prev_bar["Low"]) * 0.9995, 4)
    else:
        return round(float(prev_bar["High"]) * 1.0005, 4)


def _quick_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        import ta.volatility as tav
        atr = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period)
        val = atr.average_true_range().iloc[-1]
        return float(val) if not pd.isna(val) else 0.0
    except Exception:
        return 0.0


def _last_close(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])
