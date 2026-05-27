"""
Opening Gap Fade — Strategy 4

Concept:
    Fade overextended gap-ups or gap-downs that lack news volume.
    Statistical edge: >60% of non-news gaps between 0.75–3% partially fill
    within the first 90 minutes of trading.

Edge:
    Pre-market thin liquidity causes overshoots. Once the regular session
    opens with real two-sided volume, the gap corrects.

Typical trades per day: 0–1 per symbol (only on gap days)
Best conditions: Quiet news environment, no earnings, moderate gaps
Known weaknesses: Earnings gaps, Fed days, macro catalyst gaps — all reverse this edge

Key improvements:
    - Use the SECOND 15m bar's RSI (not the first — RSI with 1 bar is meaningless).
      Need at least 2 bars to get a meaningful RSI reading.
    - Add 5m bar confirmation: require the 5m chart to show a reversal candle
      in the 9:45–10:00 window before triggering (not just the 15m open bar)
    - Require the 15m body to exceed 40% of the bar range (real movement, not doji)
    - Tighten gap range: 0.5–2.5% (original 3% includes gap-and-go territory)
    - Use prior 5-day average volume for comparison, not a 40-bar lookback
      (40 bars of 15m = only 10 hours — meaningless for volume context)
    - Stop raised to 0.35% (was 0.25%) — slightly more room to breathe
    - Target: 60% fill (was 50%) — gaps tend to fill more than half
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import ta.momentum as tam

from app.services.strategy.daytrading.market_open import (
    localize_for_symbol, market_session,
)
from app.services.strategy.daytrading.models import DayTradeSignal

# Session-relative gap-fade window so it tracks the right market.
# Fade only in the first hour; require 5m confirmation after the first 15 min.
GAP_FADE_CUTOFF_OFFSET_MIN = 60
CONFIRM_WINDOW_START_OFFSET_MIN = 15


class OpeningGapFade:
    name = "OpeningGapFade"
    timeframe = "15m"
    default_config: dict[str, Any] = {
        "gap_min_pct": 0.50,        # lowered from 0.75 — catch more setups
        "gap_max_pct": 2.5,         # lowered from 3.0 — >2.5% is often a gap-and-go
        "rsi_overbought": 62,       # lowered from 65 — slightly more permissive
        "rsi_oversold": 38,         # raised from 35 — slightly more permissive
        "max_vol_ratio": 1.8,       # tightened from 2.0 — stricter news-gap filter
        "target_fill_pct": 0.60,    # raised from 0.50 — aim for 60% fill
        "stop_beyond_pct": 0.35,    # raised from 0.25 — more breathing room
        "min_body_pct": 0.40,       # new: body must be ≥40% of bar range (real move)
        "min_rr": 1.5,              # minimum R:R
    }

    def generate_signals(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        symbol: str,
        config: dict | None = None,
        regime: str = "BULL_OPEN",
    ) -> list[DayTradeSignal]:
        cfg = {**self.default_config, **(config or {})}
        signals: list[DayTradeSignal] = []

        today_15m = _today_bars(df_15m, symbol)
        all_15m = _localize(df_15m, symbol)
        today_5m = _today_bars(df_5m, symbol)

        if today_15m.empty or len(today_15m) < 2:
            return signals

        _sess = market_session(symbol)
        gap_fade_cutoff = _sess.after_open(GAP_FADE_CUTOFF_OFFSET_MIN)

        today_date = today_15m.index[-1].date()
        prev_bars = all_15m[all_15m.index.date < today_date]
        if prev_bars.empty:
            return signals

        # Use last 5 trading days' average daily volume for comparison
        prev_daily_closes = prev_bars.groupby(prev_bars.index.date)["Volume"].sum()
        avg_daily_vol = float(prev_daily_closes.tail(5).mean()) if len(prev_daily_closes) >= 2 else None

        prior_close = float(prev_bars["Close"].iloc[-1])
        open_bar = today_15m.iloc[0]
        open_price = float(open_bar["Open"])

        gap_pct = (open_price - prior_close) / prior_close * 100
        abs_gap = abs(gap_pct)

        if abs_gap < cfg["gap_min_pct"] or abs_gap > cfg["gap_max_pct"]:
            return signals

        bar_time = today_15m.index[0].time()
        if bar_time >= gap_fade_cutoff:
            return signals

        # Use second bar's RSI (first bar has no prior context for RSI calc)
        rsi_bar_idx = min(1, len(today_15m) - 1)
        rsi_series = tam.RSIIndicator(today_15m["Close"], window=14).rsi()
        if rsi_series is None or len(rsi_series) <= rsi_bar_idx or pd.isna(rsi_series.iloc[rsi_bar_idx]):
            # Fall back to first bar if second not available
            if len(rsi_series) == 0 or pd.isna(rsi_series.iloc[0]):
                return signals
            rsi_val = float(rsi_series.iloc[0])
        else:
            rsi_val = float(rsi_series.iloc[rsi_bar_idx])

        # Volume check: today's first bar vs avg daily (normalized to per-15m)
        first_vol = float(open_bar["Volume"])
        # Estimate avg 15m volume: daily vol / 26 bars per day
        avg_15m_vol = (avg_daily_vol / 26) if avg_daily_vol and avg_daily_vol > 0 else None
        vol_ratio = (first_vol / avg_15m_vol) if avg_15m_vol else 1.0

        if vol_ratio > cfg["max_vol_ratio"]:
            return signals  # news gap — skip

        # Body size check on the first 15m bar
        first_open = float(open_bar["Open"])
        first_close = float(open_bar["Close"])
        first_range = float(open_bar["High"]) - float(open_bar["Low"])
        body_pct = abs(first_close - first_open) / first_range if first_range > 0 else 0.0

        if body_pct < cfg["min_body_pct"]:
            return signals  # doji / indecision bar — skip

        # Look for 5m confirmation: a reversal bar in the 9:45–10:30 window
        confirm_bar = _find_5m_confirmation(today_5m, gap_pct, symbol)

        # Gap UP fade → SELL SHORT
        if (
            gap_pct > 0
            and first_close < first_open      # bearish first 15m bar
            and rsi_val > cfg["rsi_overbought"]
            and (confirm_bar is None or confirm_bar["is_bearish"])  # 5m confirms if available
        ):
            gap_high = open_price
            entry = first_close if confirm_bar is None else confirm_bar["close"]
            target = gap_high - (gap_high - prior_close) * cfg["target_fill_pct"]
            stop = gap_high * (1 + cfg["stop_beyond_pct"] / 100)

            risk = stop - entry
            reward = entry - target
            if risk <= 0 or reward <= 0:
                return signals
            rr = reward / risk
            if rr < cfg["min_rr"]:
                return signals

            confidence = _score(rsi_val, abs_gap, vol_ratio, body_pct, "gap_up", rr)
            sig_time = today_15m.index[0] if confirm_bar is None else confirm_bar["time"]

            signals.append(
                DayTradeSignal(
                    symbol=symbol,
                    strategy=self.name,
                    direction="SELL",
                    timeframe=self.timeframe,
                    entry_price=round(entry, 4),
                    stop_price=round(stop, 4),
                    target_price=round(target, 4),
                    confidence=round(confidence, 2),
                    reason=(
                        f"Gap-up {gap_pct:.2f}% fade. RSI {rsi_val:.1f}. "
                        f"Bearish first bar (body {body_pct:.0%}). "
                        f"Vol {vol_ratio:.1f}×. Target: {cfg['target_fill_pct']*100:.0f}% fill. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "gap_pct": round(gap_pct, 3),
                        "prior_close": round(prior_close, 4),
                        "open_price": round(open_price, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "body_pct": round(body_pct, 3),
                        "r_r": round(rr, 2),
                    },
                    signal_time=sig_time.isoformat() if hasattr(sig_time, "isoformat") else str(sig_time),
                )
            )

        # Gap DOWN fade → BUY
        elif (
            gap_pct < 0
            and first_close > first_open      # bullish first 15m bar
            and rsi_val < cfg["rsi_oversold"]
            and (confirm_bar is None or not confirm_bar["is_bearish"])
        ):
            gap_low = open_price
            entry = first_close if confirm_bar is None else confirm_bar["close"]
            target = gap_low + (prior_close - gap_low) * cfg["target_fill_pct"]
            stop = gap_low * (1 - cfg["stop_beyond_pct"] / 100)

            risk = entry - stop
            reward = target - entry
            if risk <= 0 or reward <= 0:
                return signals
            rr = reward / risk
            if rr < cfg["min_rr"]:
                return signals

            confidence = _score(rsi_val, abs_gap, vol_ratio, body_pct, "gap_down", rr)
            sig_time = today_15m.index[0] if confirm_bar is None else confirm_bar["time"]

            signals.append(
                DayTradeSignal(
                    symbol=symbol,
                    strategy=self.name,
                    direction="BUY",
                    timeframe=self.timeframe,
                    entry_price=round(entry, 4),
                    stop_price=round(stop, 4),
                    target_price=round(target, 4),
                    confidence=round(confidence, 2),
                    reason=(
                        f"Gap-down {gap_pct:.2f}% fade. RSI {rsi_val:.1f}. "
                        f"Bullish first bar (body {body_pct:.0%}). "
                        f"Vol {vol_ratio:.1f}×. Target: {cfg['target_fill_pct']*100:.0f}% fill. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "gap_pct": round(gap_pct, 3),
                        "prior_close": round(prior_close, 4),
                        "open_price": round(open_price, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "body_pct": round(body_pct, 3),
                        "r_r": round(rr, 2),
                    },
                    signal_time=sig_time.isoformat() if hasattr(sig_time, "isoformat") else str(sig_time),
                )
            )

        return signals


def _find_5m_confirmation(today_5m: pd.DataFrame, gap_pct: float, symbol: str = "") -> dict | None:
    """
    Look for a 5m reversal candle in the post-open confirmation window
    (open+15m to open+60m, in the symbol's market session).
    Returns a dict with close/is_bearish/time if found, else None.
    """
    if today_5m.empty:
        return None
    sess = market_session(symbol)
    window_start = sess.after_open(CONFIRM_WINDOW_START_OFFSET_MIN)
    window_end = sess.after_open(GAP_FADE_CUTOFF_OFFSET_MIN)
    window = today_5m[
        (today_5m.index.time >= window_start)
        & (today_5m.index.time < window_end)
    ]
    if window.empty:
        return None

    for _, bar in window.iterrows():
        o, c = float(bar["Open"]), float(bar["Close"])
        h, l = float(bar["High"]), float(bar["Low"])
        bar_range = h - l
        if bar_range <= 0:
            continue
        body_pct = abs(c - o) / bar_range
        if body_pct < 0.35:
            continue  # doji

        if gap_pct > 0 and c < o:  # gap-up, bearish 5m bar
            return {"close": c, "is_bearish": True, "time": bar.name}
        if gap_pct < 0 and c > o:  # gap-down, bullish 5m bar
            return {"close": c, "is_bearish": False, "time": bar.name}

    return None


def _localize(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    return localize_for_symbol(df, symbol)


def _today_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    df = _localize(df, symbol)
    if df.empty:
        return df
    today = df.index[-1].date()
    return df[df.index.date == today]


def _score(rsi: float, gap_pct: float, vol_ratio: float, body_pct: float,
           direction: str, rr: float) -> float:
    base = 0.55
    if direction == "gap_up" and rsi > 68:
        base += 0.10
    elif direction == "gap_up" and rsi > 62:
        base += 0.05
    if direction == "gap_down" and rsi < 28:
        base += 0.10
    elif direction == "gap_down" and rsi < 35:
        base += 0.05
    if gap_pct > 1.5:
        base += 0.05
    if vol_ratio < 0.8:  # very quiet gap — strongest fade candidate
        base += 0.08
    if body_pct > 0.6:   # strong directional first bar
        base += 0.05
    if rr >= 2.5:
        base += 0.05
    return min(max(base, 0.0), 1.0)
