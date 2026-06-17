"""
EntryDecider — human-style pre-trade filter for one symbol.

Decision logic mirrors what a discretionary trader checks before pulling
the trigger:
  1. Market regime — don't fight SPY.
  2. Stock vs VWAP + EMA stack — are we aligned with the trend?
  3. Signal quality — risk/reward, extension from levels, volume.
  4. Time of day — no chasing extended moves after 3 PM.
  5. Confidence score — only enter when multiple factors agree.

Returns an EntryDecision with a clear one-line reason string so every
no-trade decision is explainable in the UI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.services.strategy.daytrading.risk_templates import ExitPlan

import pandas as pd

from app.services.strategy.daytrading.market_open import (
    ET, compute_vwap, is_past_last_entry, market_session, now_et,
)
from app.services.strategy.daytrading.brain.market_state import (
    MarketStateResult,
    TREND_UP, TREND_DOWN, CHOPPY, HIGH_VOL, NEWS_RISK,
)

logger = logging.getLogger(__name__)

_MIN_RR         = 1.5              # minimum risk/reward to enter
_MIN_CONFIDENCE = 0.45             # below this → NO_TRADE


@dataclass
class EntryDecision:
    """Full record of what the EntryDecider decided and why."""
    action: Literal["BUY", "SELL_SHORT", "NO_TRADE"]
    confidence: float               # 0.0 – 1.0
    chosen_strategy: str
    entry_reason: str               # human-readable explanation
    stop_price: float
    target_price: float
    size_multiplier: float          # 0.0 – 1.0

    # Individual sub-scores (for debugging / UI)
    checks: dict[str, Any] = field(default_factory=dict)

    # Structured exit plan from risk_templates (None for legacy/scoring-only paths)
    exit_plan: "ExitPlan | None" = field(default=None, repr=False)

    @property
    def is_tradeable(self) -> bool:
        return self.action != "NO_TRADE" and self.confidence >= _MIN_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "chosen_strategy": self.chosen_strategy,
            "entry_reason": self.entry_reason,
            "stop_price": round(self.stop_price, 4),
            "target_price": round(self.target_price, 4),
            "size_multiplier": round(self.size_multiplier, 2),
            "checks": self.checks,
        }


class EntryDecider:
    """
    Evaluates whether to enter a trade for one symbol right now.

    Parameters
    ----------
    direction_mode : "long_only" | "short_only" | "both"
    risk_per_trade_pct : fraction of capital to risk per trade (e.g. 0.01 = 1%)
    min_rr : minimum risk/reward ratio to accept (default 1.5)
    min_confidence : minimum confidence to pull the trigger (default 0.45)
    """

    def __init__(
        self,
        direction_mode: str = "long_only",
        risk_per_trade_pct: float = 0.01,
        min_rr: float = _MIN_RR,
        min_confidence: float = _MIN_CONFIDENCE,
    ):
        self.direction_mode = direction_mode
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_rr = min_rr
        self.min_confidence = min_confidence

    def decide(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        market_state: MarketStateResult | None,
        account_equity: float = 10_000.0,
        now_override: "datetime | None" = None,
    ) -> EntryDecision:
        """
        Main entry point. Evaluates all conditions and returns an EntryDecision.
        Call this on every new 1m or 5m bar close.

        `now_override` lets the paper-replay simulator evaluate time gates at the
        replayed bar's timestamp instead of wall-clock.
        """
        no_trade = _no_trade  # shorthand

        # ── Time gate (market-aware) ───────────────────────────────────────────
        # Use the symbol's own session cutoff (15:15 ET for US, 15:15 IST for
        # NSE) instead of a US-clock literal, so an India autotrader stops taking
        # entries at the correct wall-clock time. now_override = bar time in replay.
        if is_past_last_entry(symbol, now=now_override):
            _cut = market_session(symbol).last_entry_time
            return no_trade(f"Too late in day — no new entries after {_cut.strftime('%H:%M')} (session local)")

        # ── Need enough bars ──────────────────────────────────────────────────
        if df_5m is None or len(df_5m) < 15:
            return no_trade("Insufficient 5m bar history")

        # ── Indicators ────────────────────────────────────────────────────────
        indicators = _compute_indicators(df_5m)
        if indicators is None:
            return no_trade("Could not compute indicators")

        close       = indicators["close"]
        vwap        = indicators["vwap"]
        ema9        = indicators["ema9"]
        ema21       = indicators["ema21"]
        atr         = indicators["atr"]
        rsi         = indicators["rsi"]
        vol_ratio   = indicators["vol_ratio"]
        above_vwap  = close > vwap
        ema_bullish = ema9 > ema21

        # ── Market regime check ───────────────────────────────────────────────
        state = market_state.state if market_state else "UNKNOWN"
        state_conf = market_state.confidence if market_state else 0.5

        if state == NEWS_RISK:
            return no_trade(f"Market is NEWS_RISK — no entries during catalyst events")

        # Determine which direction the market environment favours
        if state == TREND_UP:
            preferred_long = True
            preferred_short = False
        elif state == TREND_DOWN:
            preferred_long = False
            preferred_short = True
        elif state == CHOPPY:
            # Only VWAP-reversion allowed; no breakout entries
            preferred_long = above_vwap is False  # buying a dip toward VWAP
            preferred_short = above_vwap is True   # shorting a rip toward VWAP
        else:
            preferred_long = preferred_short = True

        # ── Direction mode filtering ──────────────────────────────────────────
        can_long  = self.direction_mode in ("long_only", "both")
        can_short = self.direction_mode in ("short_only", "both")

        # ── Build BUY candidate ───────────────────────────────────────────────
        long_score, long_reason, long_stop, long_target, long_strategy = \
            self._score_long(
                symbol, df_5m, df_1m, indicators,
                state, state_conf, preferred_long,
            )

        # ── Build SELL_SHORT candidate ────────────────────────────────────────
        short_score, short_reason, short_stop, short_target, short_strategy = \
            self._score_short(
                symbol, df_5m, df_1m, indicators,
                state, state_conf, preferred_short,
            )

        # ── Pick best action ──────────────────────────────────────────────────
        action: Literal["BUY", "SELL_SHORT", "NO_TRADE"] = "NO_TRADE"
        score = 0.0
        reason = "No clear edge found"
        stop = 0.0
        target = 0.0
        strategy = ""

        if can_long and long_score >= self.min_confidence:
            action = "BUY"
            score = long_score
            reason = long_reason
            stop = long_stop
            target = long_target
            strategy = long_strategy

        if can_short and short_score >= self.min_confidence and short_score > long_score:
            action = "SELL_SHORT"
            score = short_score
            reason = short_reason
            stop = short_stop
            target = short_target
            strategy = short_strategy

        if action == "NO_TRADE":
            # Build a helpful explanation
            parts = []
            if long_score < self.min_confidence and can_long:
                parts.append(f"Long score {long_score:.2f} < min {self.min_confidence}")
            if short_score < self.min_confidence and can_short:
                parts.append(f"Short score {short_score:.2f} < min {self.min_confidence}")
            if not parts:
                parts.append(reason)
            return no_trade("; ".join(parts))

        # ── Final R/R check ───────────────────────────────────────────────────
        if stop <= 0 or target <= 0:
            return no_trade("Invalid stop/target prices")

        if action == "BUY":
            risk  = close - stop
            reward = target - close
        else:
            risk  = stop - close
            reward = close - target

        if risk <= 0:
            return no_trade(f"Stop {stop:.2f} is not below entry {close:.2f} for {action}")

        rr = reward / risk
        if rr < self.min_rr:
            return no_trade(
                f"R:R {rr:.2f} < minimum {self.min_rr} "
                f"(risk=${risk:.2f}, reward=${reward:.2f})"
            )

        # ── Extension check — don't chase extended candles ────────────────────
        if atr > 0:
            dist_from_vwap = abs(close - vwap) / atr
            if dist_from_vwap > 2.5:
                return no_trade(
                    f"Price {dist_from_vwap:.1f}x ATR from VWAP — too extended to chase"
                )

        # ── Size multiplier ───────────────────────────────────────────────────
        size_mult = _compute_size_mult(score, vol_ratio, state, state_conf)

        checks = {
            "state": state,
            "state_confidence": round(state_conf, 2),
            "close": round(close, 4),
            "vwap": round(vwap, 4),
            "ema9": round(ema9, 4),
            "ema21": round(ema21, 4),
            "atr": round(atr, 4),
            "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
            "rr": round(rr, 2),
            "long_score": round(long_score, 3),
            "short_score": round(short_score, 3),
        }

        logger.info(
            "%s ENTRY: %s %s @ %.2f  stop=%.2f  target=%.2f  conf=%.2f  reason=%s",
            symbol, action, strategy, close, stop, target, score, reason,
        )

        return EntryDecision(
            action=action,
            confidence=round(score, 3),
            chosen_strategy=strategy,
            entry_reason=reason,
            stop_price=round(stop, 4),
            target_price=round(target, 4),
            size_multiplier=round(size_mult, 2),
            checks=checks,
        )

    # ── Long scoring ──────────────────────────────────────────────────────────

    def _score_long(
        self, symbol, df_5m, df_1m, ind, state, state_conf, preferred
    ) -> tuple[float, str, float, float, str]:
        """Score the BUY setup. Returns (score, reason, stop, target, strategy)."""
        close  = ind["close"]
        vwap   = ind["vwap"]
        ema9   = ind["ema9"]
        ema21  = ind["ema21"]
        atr    = ind["atr"]
        rsi    = ind["rsi"]
        vol_r  = ind["vol_ratio"]

        score = 0.0
        reasons: list[str] = []
        strategy = "EMAMomentum"

        # ── Market alignment (biggest weight) ─────────────────────────────────
        if preferred:
            score += 0.30
            reasons.append(f"{state} regime favours longs")
        else:
            score -= 0.15
            reasons.append(f"{state} regime disfavours longs")

        # ── Stock vs VWAP ─────────────────────────────────────────────────────
        if close > vwap:
            score += 0.20
            reasons.append(f"above VWAP ({vwap:.2f})")
        else:
            dist_pct = (vwap - close) / vwap * 100
            # VWAP reversion setup: close is below VWAP but bouncing
            if state == CHOPPY and dist_pct < 0.5:
                score += 0.10
                reasons.append(f"VWAP reversion setup ({dist_pct:.2f}% below)")
                strategy = "VWAPMeanReversion"
            else:
                score -= 0.10

        # ── EMA trend ─────────────────────────────────────────────────────────
        if ema9 > ema21:
            score += 0.15
            reasons.append(f"EMA9 ({ema9:.2f}) > EMA21 ({ema21:.2f}) bullish")
        else:
            score -= 0.08

        # ── ORB breakout check ────────────────────────────────────────────────
        orb_hi = ind.get("orb_high", 0)
        if orb_hi > 0 and close > orb_hi * 1.001 and state in (TREND_UP,):
            score += 0.10
            reasons.append(f"ORB breakout above {orb_hi:.2f}")
            strategy = "ORBBreakout"

        # ── Bollinger squeeze breakout (long) ─────────────────────────────────
        bb_upper = ind.get("bb_upper", 0)
        bb_squeeze = ind.get("bb_squeeze", False)
        if bb_upper > 0 and close > bb_upper and bb_squeeze:
            score += 0.08
            reasons.append(f"BB squeeze breakout above {bb_upper:.2f}")
            strategy = "BollingerMomentum"

        # ── Supertrend pullback (long) ─────────────────────────────────────────
        st_line = ind.get("st_line", 0)
        st_dir  = ind.get("st_dir", 0)
        if st_line > 0 and st_dir == 1 and state in (TREND_UP,):
            dist_to_st = close - st_line
            atr_val = ind.get("atr", 1)
            if 0 < dist_to_st < atr_val * 1.0:   # close but not too far above ST
                score += 0.07
                reasons.append(f"Supertrend pullback, ST={st_line:.2f}")
                strategy = "SupertrendTrend"

        # ── RSI ───────────────────────────────────────────────────────────────
        if 45 <= rsi <= 70:
            score += 0.08
        elif rsi > 75:
            score -= 0.08
            reasons.append(f"RSI {rsi:.0f} overbought")

        # ── Volume ────────────────────────────────────────────────────────────
        if vol_r >= 1.5:
            score += 0.12
            reasons.append(f"rel vol {vol_r:.1f}x — elevated")
        elif vol_r < 0.5:
            score -= 0.10
            reasons.append(f"rel vol {vol_r:.1f}x — too thin")

        # ── State confidence modifier ─────────────────────────────────────────
        if state_conf < 0.35:
            score *= 0.7    # low-confidence state → dampen all scores

        # ── Stop / target ─────────────────────────────────────────────────────
        stop   = round(close - atr * 1.2, 4) if atr > 0 else close * 0.985
        target = round(close + atr * 2.0, 4) if atr > 0 else close * 1.025

        reason_str = f"BUY {symbol}: " + ", ".join(reasons[:4])
        return max(0.0, min(1.0, score)), reason_str, stop, target, strategy

    # ── Short scoring ─────────────────────────────────────────────────────────

    def _score_short(
        self, symbol, df_5m, df_1m, ind, state, state_conf, preferred
    ) -> tuple[float, str, float, float, str]:
        """Score the SELL_SHORT setup. Returns (score, reason, stop, target, strategy)."""
        close  = ind["close"]
        vwap   = ind["vwap"]
        ema9   = ind["ema9"]
        ema21  = ind["ema21"]
        atr    = ind["atr"]
        rsi    = ind["rsi"]
        vol_r  = ind["vol_ratio"]

        score = 0.0
        reasons: list[str] = []
        strategy = "EMAMomentum"

        # ── Market alignment ──────────────────────────────────────────────────
        if preferred:
            score += 0.30
            reasons.append(f"{state} regime favours shorts")
        else:
            score -= 0.15

        # ── Stock vs VWAP ─────────────────────────────────────────────────────
        if close < vwap:
            score += 0.20
            reasons.append(f"below VWAP ({vwap:.2f})")
        else:
            score -= 0.10

        # ── EMA trend ─────────────────────────────────────────────────────────
        if ema9 < ema21:
            score += 0.15
            reasons.append(f"EMA9 ({ema9:.2f}) < EMA21 ({ema21:.2f}) bearish")
        else:
            score -= 0.08

        # ── ORB breakdown ─────────────────────────────────────────────────────
        orb_lo = ind.get("orb_low", 0)
        if orb_lo > 0 and close < orb_lo * 0.999 and state in (TREND_DOWN,):
            score += 0.10
            reasons.append(f"ORB breakdown below {orb_lo:.2f}")
            strategy = "ORBBreakout"

        # ── Bollinger squeeze breakdown (short) ───────────────────────────────
        bb_lower = ind.get("bb_lower", 0)
        bb_squeeze = ind.get("bb_squeeze", False)
        if bb_lower > 0 and close < bb_lower and bb_squeeze:
            score += 0.08
            reasons.append(f"BB squeeze breakdown below {bb_lower:.2f}")
            strategy = "BollingerMomentum"

        # ── Supertrend pullback (short) ────────────────────────────────────────
        st_line = ind.get("st_line", 0)
        st_dir  = ind.get("st_dir", 0)
        if st_line > 0 and st_dir == -1 and state in (TREND_DOWN,):
            dist_to_st = st_line - close
            atr_val = ind.get("atr", 1)
            if 0 < dist_to_st < atr_val * 1.0:
                score += 0.07
                reasons.append(f"Supertrend pullback short, ST={st_line:.2f}")
                strategy = "SupertrendTrend"

        # ── RSI ───────────────────────────────────────────────────────────────
        if 30 <= rsi <= 55:
            score += 0.08
        elif rsi < 25:
            score -= 0.08
            reasons.append(f"RSI {rsi:.0f} oversold — bounce risk")

        # ── Volume ────────────────────────────────────────────────────────────
        if vol_r >= 1.5:
            score += 0.12
            reasons.append(f"rel vol {vol_r:.1f}x")
        elif vol_r < 0.5:
            score -= 0.10

        # ── State confidence modifier ─────────────────────────────────────────
        if state_conf < 0.35:
            score *= 0.7

        # ── Stop / target ─────────────────────────────────────────────────────
        stop   = round(close + atr * 1.2, 4) if atr > 0 else close * 1.015
        target = round(close - atr * 2.0, 4) if atr > 0 else close * 0.975

        reason_str = f"SELL_SHORT {symbol}: " + ", ".join(reasons[:4])
        return max(0.0, min(1.0, score)), reason_str, stop, target, strategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_trade(reason: str) -> EntryDecision:
    return EntryDecision(
        action="NO_TRADE",
        confidence=0.0,
        chosen_strategy="",
        entry_reason=reason,
        stop_price=0.0,
        target_price=0.0,
        size_multiplier=0.0,
    )


def _compute_indicators(df_5m: pd.DataFrame) -> dict[str, float] | None:
    """Compute all indicators needed for entry scoring from 5m bars."""
    try:
        import ta.momentum as tam
        import ta.trend as tat
        import ta.volatility as tav

        today = df_5m.copy()
        if today.empty or len(today) < 10:
            return None

        closes = today["Close"]
        close = float(closes.iloc[-1])

        # VWAP (session reset built into compute_vwap)
        vwap_series = compute_vwap(today)
        vwap = float(vwap_series.iloc[-1])

        # EMA 9 and 21
        ema9  = float(tat.EMAIndicator(closes, window=9).ema_indicator().iloc[-1])
        ema21 = float(tat.EMAIndicator(closes, window=21).ema_indicator().iloc[-1])

        # ATR (14)
        atr_ind = tav.AverageTrueRange(today["High"], today["Low"], closes, window=14)
        atr_val = atr_ind.average_true_range()
        atr = float(atr_val.iloc[-1]) if not pd.isna(atr_val.iloc[-1]) else 0.0

        # RSI (14)
        rsi_val = tam.RSIIndicator(closes, window=14).rsi()
        rsi = float(rsi_val.iloc[-1]) if not pd.isna(rsi_val.iloc[-1]) else 50.0

        # Relative volume (current bar vs 20-bar rolling avg)
        vol_avg = float(today["Volume"].rolling(20).mean().iloc[-1]) if len(today) >= 20 else float(today["Volume"].mean())
        vol_ratio = float(today["Volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

        # Opening range (first 3 bars = 15m)
        orb_bars = today.iloc[:3]
        orb_high = float(orb_bars["High"].max())
        orb_low  = float(orb_bars["Low"].min())

        # Bollinger Bands (20, 2.0) + squeeze detection
        bb_upper_val = bb_lower_val = 0.0
        bb_squeeze = False
        if len(today) >= 20:
            bb_ind = tav.BollingerBands(closes, window=20, window_dev=2.0)
            bb_upper_series = bb_ind.bollinger_hband()
            bb_lower_series = bb_ind.bollinger_lband()
            bb_mid_series   = bb_ind.bollinger_mavg()
            bb_upper_val = float(bb_upper_series.iloc[-1]) if not pd.isna(bb_upper_series.iloc[-1]) else 0.0
            bb_lower_val = float(bb_lower_series.iloc[-1]) if not pd.isna(bb_lower_series.iloc[-1]) else 0.0
            mid_val = float(bb_mid_series.iloc[-1])
            if mid_val > 0:
                bb_width_series = (bb_upper_series - bb_lower_series) / bb_mid_series
                if len(bb_width_series.dropna()) >= 10:
                    current_width = float(bb_width_series.iloc[-1])
                    lookback = bb_width_series.dropna().iloc[-40:]
                    bb_squeeze = current_width <= float(lookback.quantile(0.20))

        # Supertrend (5m, length=10, multiplier=3.0)
        st_line_val = 0.0
        st_dir_val  = 0
        if len(today) >= 12:
            try:
                from app.services.strategy.daytrading.strategies.supertrend_trend import _compute_supertrend
                st_df = _compute_supertrend(today, length=10, multiplier=3.0)
                if st_df is not None and not st_df.empty:
                    st_line_val = float(st_df["supertrend"].iloc[-1])
                    st_dir_val  = int(st_df["direction"].iloc[-1])
            except Exception:
                pass

        return {
            "close": close,
            "vwap": vwap,
            "ema9": ema9,
            "ema21": ema21,
            "atr": atr,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "bb_upper": bb_upper_val,
            "bb_lower": bb_lower_val,
            "bb_squeeze": bb_squeeze,
            "st_line": st_line_val,
            "st_dir": st_dir_val,
        }
    except Exception as e:
        logger.debug("indicator compute failed: %s", e)
        return None


def _compute_size_mult(
    score: float,
    vol_ratio: float,
    state: str,
    state_conf: float,
) -> float:
    """Size multiplier: full size only when all conditions strongly align."""
    mult = score  # base: proportional to confidence

    # Penalise low vol
    if vol_ratio < 0.8:
        mult *= 0.75

    # State-based scaling
    if state in (HIGH_VOL,):
        mult *= 0.5
    elif state == CHOPPY:
        mult *= 0.75

    # Low-confidence state → smaller size
    if state_conf < 0.4:
        mult *= 0.7

    return round(max(0.25, min(1.0, mult)), 2)
