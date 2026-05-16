"""
Opening Range Breakout (ORB) — Strategy 1

Concept:
    Define the opening range as the high/low of the first 15 minutes (3×5m bars).
    Enter only when price CLOSES above ORB high by at least 0.1% buffer — this
    eliminates the majority of fakeout entries that reverse immediately.

Edge:
    Institutional order flow concentrates at open. A decisive volume-confirmed
    break above the range signals directional intent for the session.

Typical trades per day: 0.5–1 per symbol
Best conditions: BULL_OPEN, gap-and-go days, high relative volume at open
Known weaknesses: Choppy/low-volatility days, large opening ranges (risk too wide)

Key improvements over naive ORB:
    - Entry buffer (0.1%) above ORB high eliminates fakeouts
    - ATR-based stop instead of full ORB range (tighter, better R:R)
    - ORB height / ATR ratio check — skip wide, untradeable ranges
    - Require higher low in the last 2 bars before breakout (momentum confirm)
    - Reject entries after 11:30 AM ET (late breakouts rarely follow through)
    - Minimum 2.0 R:R required before entry
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, LAST_ENTRY_TIME
from app.services.strategy.daytrading.models import DayTradeSignal

# No new ORB entries after 11:30 AM — late breakouts have poor follow-through
ORB_LAST_ENTRY = time(11, 30)


class ORBBreakout:
    name = "ORBBreakout"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "orb_minutes": 15,          # opening range window
        "entry_buffer_pct": 0.10,   # must close this % above ORB high (eliminates fakeouts)
        "vol_multiple": 1.8,        # volume must exceed this × 20-bar avg (raised from 1.5)
        "tp_multiplier": 2.0,       # target = entry + orb_height × this (raised for better R:R)
        "atr_stop_mult": 1.0,       # stop = entry - atr × this (tighter than full ORB range)
        "rsi_period": 14,
        "rsi_min": 52,              # slightly above 50 — confirmed momentum
        "max_orb_atr_ratio": 2.5,   # skip if ORB height > 2.5×ATR (range too wide to trade)
        "min_rr": 2.0,              # minimum R:R required to take the trade
        "max_hold_bars": 48,        # 4 hours max (was 60)
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

        if regime == "BEAR_OPEN":
            return signals

        today_bars = _today_bars(df_5m)
        if today_bars.empty or len(today_bars) < 6:
            return signals

        orb_bars = max(1, cfg["orb_minutes"] // 5)
        orb_df = today_bars.iloc[:orb_bars]
        orb_high = float(orb_df["High"].max())
        orb_low = float(orb_df["Low"].min())
        orb_height = orb_high - orb_low

        if orb_height <= 0:
            return signals

        df_ind = today_bars.copy()
        df_ind["rsi"] = tam.RSIIndicator(df_ind["Close"], window=cfg["rsi_period"]).rsi()
        atr_ind = tav.AverageTrueRange(df_ind["High"], df_ind["Low"], df_ind["Close"], window=14)
        df_ind["atr"] = atr_ind.average_true_range()
        vol_avg = df_ind["Volume"].rolling(20).mean()

        open_signal_seen = False

        for i in range(orb_bars, len(df_ind)):
            if open_signal_seen:
                break

            bar = df_ind.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None

            # ORB entries only work in the first 2 hours
            if bar_time and bar_time >= ORB_LAST_ENTRY:
                break

            if pd.isna(bar["rsi"]) or pd.isna(bar["atr"]) or pd.isna(vol_avg.iloc[i]):
                continue

            close = float(bar["Close"])
            volume = float(bar["Volume"])
            avg_vol = float(vol_avg.iloc[i])
            rsi_val = float(bar["rsi"])
            atr_val = float(bar["atr"])

            if atr_val <= 0:
                continue

            # Skip untradeable wide ranges.
            # Compare ORB height against multi-bar expected range (orb_bars * ATR),
            # not a single 5m bar ATR. A 3-bar ORB covering 3x ATR is normal.
            expected_orb_range = orb_bars * atr_val
            if expected_orb_range > 0 and orb_height > cfg["max_orb_atr_ratio"] * expected_orb_range:
                break

            # Entry buffer: close must be meaningfully above ORB high, not just touching it
            breakout_threshold = orb_high * (1 + cfg["entry_buffer_pct"] / 100)
            if close <= breakout_threshold:
                continue

            # Volume confirmation
            if volume <= cfg["vol_multiple"] * avg_vol:
                continue

            # RSI momentum confirmation
            if rsi_val <= cfg["rsi_min"]:
                continue

            # Higher low structure: previous bar's low > bar before that's low
            # Confirms momentum is building into the breakout, not exhausting
            if i >= 2:
                prev_low = float(df_ind.iloc[i - 1]["Low"])
                prev2_low = float(df_ind.iloc[i - 2]["Low"])
                if prev_low < prev2_low:
                    continue  # lower low before breakout — likely a fakeout

            # ATR-based stop (tighter than full ORB range)
            entry = close
            stop = entry - cfg["atr_stop_mult"] * atr_val
            # Never let stop go above ORB low — use whichever is tighter
            stop = max(stop, orb_low)
            target = entry + orb_height * cfg["tp_multiplier"]

            risk = entry - stop
            if risk <= 0:
                continue
            rr = (target - entry) / risk
            if rr < cfg["min_rr"]:
                continue

            confidence = _score(rsi_val, volume / avg_vol, regime, rr)

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
                        f"ORB breakout: close {close:.2f} > threshold {breakout_threshold:.2f}. "
                        f"ORB range {orb_height:.2f}. Vol {volume/avg_vol:.1f}× avg. "
                        f"RSI {rsi_val:.1f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "orb_high": round(orb_high, 4),
                        "orb_low": round(orb_low, 4),
                        "orb_height": round(orb_height, 4),
                        "atr": round(atr_val, 4),
                        "orb_atr_ratio": round(orb_height / atr_val, 2),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(volume / avg_vol, 2),
                        "r_r": round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                )
            )
            open_signal_seen = True

        return signals


def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    df = df.copy()
    df.index = idx
    today = idx[-1].date()
    return df[idx.date == today]


def _score(rsi: float, vol_ratio: float, regime: str, rr: float) -> float:
    base = 0.55
    if rsi > 60:
        base += 0.08
    if vol_ratio > 2.5:
        base += 0.10
    elif vol_ratio > 2.0:
        base += 0.05
    if rr >= 3.0:
        base += 0.08
    elif rr >= 2.5:
        base += 0.04
    if regime == "BULL_OPEN":
        base += 0.08
    elif regime == "CHOPPY":
        base -= 0.10
    return min(max(base, 0.0), 1.0)
