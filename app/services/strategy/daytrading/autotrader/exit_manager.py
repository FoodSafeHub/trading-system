"""
ExitManager — evaluates every bar and decides whether to close the position.

Exit trigger hierarchy (first match wins):
  1. Hard stop hit                   — non-negotiable
  2. EOD force-flatten               — 3:45 PM ET
  3. Stale data guard                — no new bar for > 2 bars worth of time
  4. Time-based stop-out             — trade hasn't worked after N bars
  5. Profit target hit               — close 100% at target
  6. Trailing stop hit               — trail stop touched
  7. Regime-aware momentum fade      — requires 2-of-3 confirmation signals
  8. Winner protection               — was at +1R, now fading back to entry

Regime management profiles:
  TREND_UP/DOWN : let winners run — wider trail, ignore minor vol fade,
                  momentum fade requires all 3 signals
  CHOPPY        : take profits faster — at +1R or with any 2 fade signals
  HIGH_VOL      : ultra-tight management, exit on first reversal signal
  NEWS_RISK     : immediately flatten if open when regime detected
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Literal

import pandas as pd

from app.services.strategy.daytrading.autotrader.trade_state import TradeStateMachine
from app.services.strategy.daytrading.market_open import ET, compute_vwap, now_et

logger = logging.getLogger(__name__)

_EOD_FORCE_FLAT  = time(15, 45)
_EOD_WARN_TIME   = time(15, 30)
_MAX_HOLD_BARS   = 48
_PARABOLIC_MULT  = 2.5

# Regime-specific profiles
_REGIME_PROFILES: dict[str, dict] = {
    "TREND_UP": {
        "quick_profit_r": 2.0,       # only quick-exit at +2R
        "fade_signals_needed": 3,    # need all 3 signals to exit on momentum fade
        "eod_trail_atr": 0.75,       # looser EOD trail
        "winner_protect_mfe_mult": 1.2,  # only protect if was +1.2R
    },
    "TREND_DOWN": {
        "quick_profit_r": 2.0,
        "fade_signals_needed": 3,
        "eod_trail_atr": 0.75,
        "winner_protect_mfe_mult": 1.2,
    },
    "CHOPPY": {
        "quick_profit_r": 1.0,       # take profits at +1R
        "fade_signals_needed": 2,    # any 2 fade signals = exit
        "eod_trail_atr": 0.4,
        "winner_protect_mfe_mult": 0.8,
    },
    "HIGH_VOL": {
        "quick_profit_r": 0.8,       # exit early
        "fade_signals_needed": 1,    # single signal is enough
        "eod_trail_atr": 0.5,
        "winner_protect_mfe_mult": 0.7,
    },
    "NEWS_RISK": {
        "quick_profit_r": 0.0,       # flatten immediately
        "fade_signals_needed": 1,
        "eod_trail_atr": 0.3,
        "winner_protect_mfe_mult": 0.5,
    },
    "UNKNOWN": {
        "quick_profit_r": 1.5,
        "fade_signals_needed": 2,
        "eod_trail_atr": 0.5,
        "winner_protect_mfe_mult": 1.0,
    },
}

# Strategy-specific hold behaviour (overrides regime profile for some params)
_STRATEGY_PROFILES: dict[str, dict] = {
    "ORBBreakout": {
        "min_bars_before_fade_exit": 3,  # don't cut ORB runners too early
        "quick_profit_r_bonus": 0.5,     # add to regime's quick_profit_r
    },
    "VWAPMeanReversion": {
        "min_bars_before_fade_exit": 1,
        "quick_profit_r_bonus": -0.25,   # mean-reversion: take it faster
    },
    "EMAMomentum": {
        "min_bars_before_fade_exit": 2,
        "quick_profit_r_bonus": 0.0,
    },
}


@dataclass
class ExitDecision:
    """What the ExitManager wants to do this bar."""
    action: Literal["HOLD", "PARTIAL_EXIT", "FULL_EXIT", "MOVE_STOP"]
    reason: str
    exit_price: float = 0.0
    new_stop: float | None = None
    urgency: Literal["low", "medium", "high"] = "low"
    fade_signals: list[str] = field(default_factory=list)   # for UI explainability

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "exit_price": round(self.exit_price, 4),
            "new_stop": round(self.new_stop, 4) if self.new_stop else None,
            "urgency": self.urgency,
            "fade_signals": self.fade_signals,
        }


class ExitManager:
    """
    Runs exit checks against the current bar and returns an ExitDecision.

    Does NOT execute — only returns instructions. SingleStockTrader acts.

    Parameters
    ----------
    max_hold_bars : exit the trade if it hasn't closed after this many 5m bars.
    use_momentum_exit : check for momentum fade signals (default True).
    quick_profit_mode : take profits faster on choppy days (default True).
    stale_bar_timeout_s : seconds without a new bar before treating data as stale.
    """

    def __init__(
        self,
        max_hold_bars: int = _MAX_HOLD_BARS,
        use_momentum_exit: bool = True,
        quick_profit_mode: bool = True,
        stale_bar_timeout_s: int = 600,   # 10 min = 2 missed 5m bars
    ):
        self.max_hold_bars = max_hold_bars
        self.use_momentum_exit = use_momentum_exit
        self.quick_profit_mode = quick_profit_mode
        self.stale_bar_timeout_s = stale_bar_timeout_s
        self._hold_bars = 0
        self._last_bar_time: pd.Timestamp | None = None

    def reset(self) -> None:
        self._hold_bars = 0
        self._last_bar_time = None

    def evaluate(
        self,
        tsm: TradeStateMachine,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame | None,
        market_state_str: str = "UNKNOWN",
    ) -> ExitDecision:
        """
        Evaluate one bar and return exit/hold instructions.
        Call on every new 5m (and optionally 1m) bar close.
        """
        if not tsm.has_position:
            return ExitDecision(action="HOLD", reason="No active position")

        self._hold_bars += 1

        close = _last_close(df_5m)
        if close <= 0:
            return ExitDecision(action="HOLD", reason="No price data")

        # Track last bar time for stale-data detection
        if not df_5m.empty:
            self._last_bar_time = df_5m.index[-1]

        side         = tsm.side
        entry        = tsm.entry_price
        current_stop = tsm.current_stop
        first_target = tsm.first_target
        initial_stop = tsm.initial_stop
        strategy     = tsm.strategy or "EMAMomentum"

        risk_unit = (
            (entry - initial_stop) if side == "LONG"
            else (initial_stop - entry)
        )
        r_multiple = (
            (close - entry) / risk_unit if side == "LONG"
            else (entry - close) / risk_unit
        ) if risk_unit > 0 else 0.0

        # Look up regime and strategy profiles
        profile = _REGIME_PROFILES.get(market_state_str, _REGIME_PROFILES["UNKNOWN"])
        strat_profile = _STRATEGY_PROFILES.get(strategy, _STRATEGY_PROFILES["EMAMomentum"])
        min_bars_before_fade = strat_profile.get("min_bars_before_fade_exit", 2)

        # ── 1. Hard stop ──────────────────────────────────────────────────────
        if side == "LONG" and close <= current_stop:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Hard stop hit: close {close:.2f} <= stop {current_stop:.2f}",
                exit_price=current_stop,
                urgency="high",
            )
        if side == "SHORT" and close >= current_stop:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Hard stop hit: close {close:.2f} >= stop {current_stop:.2f}",
                exit_price=current_stop,
                urgency="high",
            )

        # ── 2. EOD force-flatten ──────────────────────────────────────────────
        now_time = now_et().time()
        if now_time >= _EOD_FORCE_FLAT:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"EOD force flatten at {now_time.strftime('%H:%M')} ET",
                exit_price=close,
                urgency="high",
            )

        # ── 2b. NEWS_RISK — flatten immediately if open ───────────────────────
        if market_state_str == "NEWS_RISK" and r_multiple > 0:
            return ExitDecision(
                action="FULL_EXIT",
                reason="Regime turned NEWS_RISK with open position — flattening",
                exit_price=close,
                urgency="high",
            )

        # ── 3. Stale data guard ───────────────────────────────────────────────
        if self._last_bar_time is not None:
            try:
                last_ts = self._last_bar_time
                now_ts = pd.Timestamp.now(tz=last_ts.tzinfo)
                elapsed_s = (now_ts - last_ts).total_seconds()
                if elapsed_s > self.stale_bar_timeout_s and r_multiple < 0:
                    return ExitDecision(
                        action="FULL_EXIT",
                        reason=f"Stale data ({elapsed_s/60:.0f}m since last bar) with losing trade — safety exit",
                        exit_price=close,
                        urgency="high",
                    )
            except Exception:
                pass

        # ── 4. Time-based stop-out ────────────────────────────────────────────
        if self._hold_bars >= self.max_hold_bars:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Trade expired after {self._hold_bars} bars (no follow-through)",
                exit_price=close,
                urgency="medium",
            )

        # ── 5. Profit target ──────────────────────────────────────────────────
        if side == "LONG" and close >= first_target:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Profit target hit: close {close:.2f} >= target {first_target:.2f} (+{r_multiple:.1f}R)",
                exit_price=first_target,
                urgency="medium",
            )
        if side == "SHORT" and close <= first_target:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Profit target hit: close {close:.2f} <= target {first_target:.2f} (+{r_multiple:.1f}R)",
                exit_price=first_target,
                urgency="medium",
            )

        # ── EOD approach: tighten stops ───────────────────────────────────────
        if now_time >= _EOD_WARN_TIME and r_multiple > 0:
            atr = _quick_atr(df_5m)
            if atr > 0:
                eod_mult = profile.get("eod_trail_atr", 0.5)
                tight_stop = (
                    round(close - atr * eod_mult, 4) if side == "LONG"
                    else round(close + atr * eod_mult, 4)
                )
                if _is_better_stop(tight_stop, current_stop, side):
                    return ExitDecision(
                        action="MOVE_STOP",
                        reason=f"EOD approach ({now_time.strftime('%H:%M')}) [{market_state_str}] — tightening to {tight_stop:.2f}",
                        new_stop=tight_stop,
                        urgency="medium",
                    )

        # ── 6. Trailing stop ──────────────────────────────────────────────────
        if tsm.state.value in ("TRAILING",):
            trail = tsm.trailing_stop
            if trail > 0:
                if side == "LONG" and close <= trail:
                    return ExitDecision(
                        action="FULL_EXIT",
                        reason=f"Trailing stop hit: close {close:.2f} <= trail {trail:.2f}",
                        exit_price=trail,
                        urgency="medium",
                    )
                if side == "SHORT" and close >= trail:
                    return ExitDecision(
                        action="FULL_EXIT",
                        reason=f"Trailing stop hit: close {close:.2f} >= trail {trail:.2f}",
                        exit_price=trail,
                        urgency="medium",
                    )

        # ── 7. Winner protection ──────────────────────────────────────────────
        mfe_mult = profile.get("winner_protect_mfe_mult", 1.0)
        if tsm.max_favorable_excursion > risk_unit * mfe_mult and r_multiple < 0.1:
            return ExitDecision(
                action="FULL_EXIT",
                reason=(
                    f"Winner protection [{market_state_str}]: was up "
                    f"{tsm.max_favorable_excursion:.2f} but now +{r_multiple:.2f}R"
                ),
                exit_price=close,
                urgency="medium",
            )

        # ── 8. Momentum fade (regime-aware, confirmation required) ────────────
        if self.use_momentum_exit and self._hold_bars >= min_bars_before_fade:
            fade_signals_needed = profile.get("fade_signals_needed", 2)
            fade = self._check_momentum_fade_confirmed(
                tsm, df_5m, df_1m, close, r_multiple,
                market_state_str, fade_signals_needed,
            )
            if fade is not None:
                return fade

        # ── Quick profit mode: regime-specific R threshold ────────────────────
        if self.quick_profit_mode:
            qp_r = profile.get("quick_profit_r", 1.5)
            qp_r += strat_profile.get("quick_profit_r_bonus", 0.0)
            if r_multiple >= qp_r and market_state_str in ("CHOPPY", "HIGH_VOL", "NEWS_RISK"):
                return ExitDecision(
                    action="FULL_EXIT",
                    reason=f"Quick profit [{market_state_str}]: +{r_multiple:.1f}R >= {qp_r:.1f}R threshold",
                    exit_price=close,
                    urgency="low",
                )

        return ExitDecision(
            action="HOLD",
            reason=f"Holding +{r_multiple:.2f}R [{market_state_str}] bars={self._hold_bars}",
            urgency="low",
        )

    # ── Confirmed momentum fade ───────────────────────────────────────────────

    def _check_momentum_fade_confirmed(
        self,
        tsm: TradeStateMachine,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame | None,
        close: float,
        r_multiple: float,
        market_state_str: str,
        signals_needed: int,
    ) -> ExitDecision | None:
        """
        Collect momentum fade signals; only exit when >= signals_needed fire.

        Signals:
          A. VWAP loss (with prior bar above)
          B. EMA9 rollover (price crosses below + EMA slope negative)
          C. Volume dry-up (< 35% of avg after extended move)
          D. Parabolic reversal candle (> 2.5x ATR)

        r_multiple threshold: only check if trade has profit to protect.
        """
        if r_multiple < 0.3:
            return None

        side = tsm.side
        atr = _quick_atr(df_5m)
        fired: list[str] = []

        # ── Signal A: VWAP loss with prior-bar confirmation ──────────────────
        try:
            vwap_series = compute_vwap(df_5m)
            vwap = float(vwap_series.iloc[-1])
            if len(df_5m) >= 2 and not pd.isna(vwap_series.iloc[-2]):
                prior_vwap = float(vwap_series.iloc[-2])
                prior_close = float(df_5m["Close"].iloc[-2])
                vwap_slope = vwap - prior_vwap   # positive = VWAP rising

                if side == "LONG" and close < vwap and r_multiple > 0.5:
                    if prior_close > prior_vwap:  # was above VWAP
                        # Extra: VWAP slope should also be turning flat/down
                        if vwap_slope <= 0:
                            fired.append(f"VWAP lost + slope flat/down ({vwap:.2f})")
                        elif r_multiple > 1.0:
                            fired.append(f"VWAP lost ({vwap:.2f})")

                if side == "SHORT" and close > vwap and r_multiple > 0.5:
                    if prior_close < prior_vwap:
                        if vwap_slope >= 0:
                            fired.append(f"VWAP reclaimed + slope rising ({vwap:.2f})")
                        elif r_multiple > 1.0:
                            fired.append(f"VWAP reclaimed ({vwap:.2f})")
        except Exception:
            pass

        # ── Signal B: EMA9 cross + slope confirms rollover ───────────────────
        try:
            import ta.trend as tat
            closes = df_5m["Close"]
            ema9 = tat.EMAIndicator(closes, window=9).ema_indicator()
            if len(ema9) >= 3 and not pd.isna(ema9.iloc[-1]):
                curr_ema = float(ema9.iloc[-1])
                prev_ema = float(ema9.iloc[-2])
                prev2_ema = float(ema9.iloc[-3])
                curr_c = float(closes.iloc[-1])
                prev_c = float(closes.iloc[-2])

                ema_turning_down = curr_ema < prev_ema < prev2_ema   # EMA rolling over
                ema_turning_up   = curr_ema > prev_ema > prev2_ema

                if side == "LONG" and prev_c > prev_ema and curr_c < curr_ema and r_multiple > 0.5:
                    if ema_turning_down:
                        fired.append(f"EMA9 cross below + rollover ({curr_ema:.2f})")
                    elif r_multiple > 1.2:
                        fired.append(f"EMA9 cross below ({curr_ema:.2f})")

                if side == "SHORT" and prev_c < prev_ema and curr_c > curr_ema and r_multiple > 0.5:
                    if ema_turning_up:
                        fired.append(f"EMA9 cross above + curl up ({curr_ema:.2f})")
                    elif r_multiple > 1.2:
                        fired.append(f"EMA9 cross above ({curr_ema:.2f})")
        except Exception:
            pass

        # ── Signal C: Volume dry-up after move ───────────────────────────────
        if len(df_5m) >= 5:
            vol_recent = float(df_5m["Volume"].iloc[-3:].mean())
            vol_avg_window = 20 if len(df_5m) >= 20 else len(df_5m)
            vol_avg = float(df_5m["Volume"].rolling(vol_avg_window).mean().iloc[-1])
            if vol_avg > 0 and vol_recent / vol_avg < 0.35 and r_multiple > 0.8:
                fired.append(f"Volume dried up ({vol_recent/vol_avg:.0%} of avg)")

        # ── Signal D: Parabolic reversal candle ───────────────────────────────
        if atr > 0 and len(df_5m) >= 2:
            last_range = abs(float(df_5m["High"].iloc[-1]) - float(df_5m["Low"].iloc[-1]))
            if last_range > atr * _PARABOLIC_MULT:
                is_bearish = float(df_5m["Close"].iloc[-1]) < float(df_5m["Open"].iloc[-1])
                is_bullish = float(df_5m["Close"].iloc[-1]) > float(df_5m["Open"].iloc[-1])
                if side == "LONG" and is_bearish:
                    fired.append(f"Parabolic reversal candle ({last_range/atr:.1f}x ATR)")
                if side == "SHORT" and is_bullish:
                    fired.append(f"Parabolic reversal candle ({last_range/atr:.1f}x ATR)")

        if len(fired) >= signals_needed:
            return ExitDecision(
                action="FULL_EXIT",
                reason=f"Momentum fade [{market_state_str}] — {len(fired)} signals: {', '.join(fired)}",
                exit_price=close,
                urgency="medium" if len(fired) >= 2 else "low",
                fade_signals=fired,
            )

        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_close(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])


def _quick_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        import ta.volatility as tav
        atr = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period)
        val = atr.average_true_range().iloc[-1]
        return float(val) if not pd.isna(val) else 0.0
    except Exception:
        return 0.0


def _is_better_stop(new_stop: float, current_stop: float, side: str) -> bool:
    return new_stop > current_stop if side == "LONG" else new_stop < current_stop
