"""
Volume Spike Reversal — Strategy 5

Concept:
    Catches capitulation reversals at unusual volume spikes with RSI extremes
    and rejection candle evidence.

Edge:
    Panic selling or euphoric buying exhausts at volume spikes.
    The reversal is swift — this is a scalp, not a swing.

Typical trades per day: 0.5–2 per symbol
Best conditions: All regimes
Known weaknesses: Genuine breakdowns can persist despite 3× volume spikes

Reality calibration:
    On SPY, RSI(14) rarely goes below 25 intraday.
    The realistic oversold level on a 5m SPY bar is 32–38 with a big volume spike.
    Volume close spikes (bars 76-77 = 3:45-4pm) must be excluded — they are
    end-of-day portfolio rebalancing, not tradeable reversals.
    Wick ratio of 0.50 is often not met on large-range capitulation candles
    where close is near the low — use 0.35 for buys.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET
from app.services.strategy.daytrading.models import DayTradeSignal

# Exclude the last 45 min — close volume spikes are not reversals
VSR_LAST_ENTRY = time(15, 0)
VSR_MIN_START = time(9, 45)   # need enough bars for volume average


class VolumeSpikeReversal:
    name = "VolumeSpikeReversal"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "spike_multiple": 2.5,       # lowered from 3.0 — catches more real spikes
        "rsi_period": 14,
        "rsi_max": 38,               # raised from 25/30 — realistic for SPY intraday
        "rsi_min_short": 62,         # lowered from 70/75 — realistic overbought
        "decline_pct": 0.4,          # 3-bar decline threshold (lowered from 0.8)
        "wick_ratio_min": 0.35,      # lowered from 0.50 — more realistic
        "min_bar_atr_mult": 0.6,     # bar range > 0.6×ATR (real move, not micro-doji)
        "atr_stop_mult": 1.0,
        "atr_tp_mult": 2.0,
        "require_largest_vol": True, # must be peak volume bar in last 10 bars
        "max_hold_bars": 10,
        "min_rr": 1.6,
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

        today = _today_bars(df_5m)
        if today.empty or len(today) < 22:
            return signals

        today = today.copy()
        today["rsi"] = tam.RSIIndicator(today["Close"], window=cfg["rsi_period"]).rsi()
        today["atr"] = tav.AverageTrueRange(
            today["High"], today["Low"], today["Close"], window=14
        ).average_true_range()
        vol_avg = today["Volume"].rolling(20).mean()

        # 15m RSI for multi-timeframe confirmation
        today_15m = _today_bars(df_15m)
        rsi15_curr: float | None = None
        rsi15_prev: float | None = None
        if not today_15m.empty and len(today_15m) >= 5:
            rsi15 = tam.RSIIndicator(today_15m["Close"], window=14).rsi()
            if rsi15 is not None and len(rsi15) >= 2 and not pd.isna(rsi15.iloc[-1]):
                rsi15_curr = float(rsi15.iloc[-1])
                rsi15_prev = float(rsi15.iloc[-2]) if not pd.isna(rsi15.iloc[-2]) else None

        open_signal_seen = False

        for i in range(10, len(today)):
            if open_signal_seen:
                break

            bar = today.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None

            # Exclude close-volume spikes and pre-market noise
            if bar_time and (bar_time < VSR_MIN_START or bar_time >= VSR_LAST_ENTRY):
                continue

            if pd.isna(bar["rsi"]) or pd.isna(bar["atr"]) or pd.isna(vol_avg.iloc[i]):
                continue

            close = float(bar["Close"])
            high = float(bar["High"])
            low = float(bar["Low"])
            volume = float(bar["Volume"])
            avg_vol = float(vol_avg.iloc[i])
            rsi_val = float(bar["rsi"])
            atr_val = float(bar["atr"])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            bar_range = high - low

            # Filter doji spikes
            if bar_range < cfg["min_bar_atr_mult"] * atr_val:
                continue

            close_3_ago = float(today.iloc[i - 3]["Close"])
            price_move_pct = abs(close - close_3_ago) / close_3_ago * 100
            lower_wick = (close - low) / bar_range if bar_range > 0 else 0.0
            upper_wick = (high - close) / bar_range if bar_range > 0 else 0.0
            vol_spike = volume > cfg["spike_multiple"] * avg_vol

            if not vol_spike:
                continue

            # Must be peak volume in last 10 bars (real capitulation, not random spike)
            if cfg["require_largest_vol"]:
                if volume < float(today.iloc[i - 9: i + 1]["Volume"].max()):
                    continue

            # ── BUY reversal ─────────────────────────────────────────
            if (
                rsi_val < cfg["rsi_max"]
                and price_move_pct >= cfg["decline_pct"]
                and close < close_3_ago
                and lower_wick > cfg["wick_ratio_min"]
            ):
                # 15m RSI confirmation: oversold or improving
                if rsi15_curr is not None:
                    if rsi15_curr > 55:       # 15m not oversold enough
                        continue
                    if rsi15_prev is not None and rsi15_curr < rsi15_prev - 5:
                        continue  # 15m still deteriorating sharply

                entry = close
                stop = entry - cfg["atr_stop_mult"] * atr_val
                target = entry + cfg["atr_tp_mult"] * atr_val

                risk = entry - stop
                if risk <= 0:
                    continue
                rr = (target - entry) / risk
                if rr < cfg["min_rr"]:
                    continue

                confidence = _score(rsi_val, volume / avg_vol, price_move_pct, lower_wick, "buy", rr)

                signals.append(DayTradeSignal(
                    symbol=symbol, strategy=self.name,
                    direction="BUY", timeframe=self.timeframe,
                    entry_price=round(entry, 4), stop_price=round(stop, 4),
                    target_price=round(target, 4), confidence=round(confidence, 2),
                    reason=(
                        f"Capitulation spike {volume/avg_vol:.1f}× avg vol. "
                        f"RSI {rsi_val:.1f} oversold. Drop {price_move_pct:.2f}% in 3 bars. "
                        f"Lower wick {lower_wick:.0%}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "vol_ratio": round(volume / avg_vol, 2),
                        "rsi_5m": round(rsi_val, 2),
                        "rsi_15m": round(rsi15_curr, 2) if rsi15_curr else None,
                        "price_move_pct": round(price_move_pct, 3),
                        "lower_wick": round(lower_wick, 3),
                        "bar_atr_ratio": round(bar_range / atr_val, 2),
                        "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                ))
                open_signal_seen = True

            # ── SELL SHORT (BEAR_OPEN) ────────────────────────────────
            elif (
                not open_signal_seen
                and regime == "BEAR_OPEN"
                and rsi_val > cfg["rsi_min_short"]
                and price_move_pct >= cfg["decline_pct"]
                and close > close_3_ago
                and upper_wick > cfg["wick_ratio_min"]
            ):
                if rsi15_curr is not None:
                    if rsi15_curr < 45:
                        continue
                    if rsi15_prev is not None and rsi15_curr > rsi15_prev + 5:
                        continue

                entry = close
                stop = entry + cfg["atr_stop_mult"] * atr_val
                target = entry - cfg["atr_tp_mult"] * atr_val
                risk = stop - entry
                if risk <= 0:
                    continue
                rr = (entry - target) / risk
                if rr < cfg["min_rr"]:
                    continue

                confidence = _score(rsi_val, volume / avg_vol, price_move_pct, upper_wick, "sell", rr)

                signals.append(DayTradeSignal(
                    symbol=symbol, strategy=self.name,
                    direction="SELL", timeframe=self.timeframe,
                    entry_price=round(entry, 4), stop_price=round(stop, 4),
                    target_price=round(target, 4), confidence=round(confidence, 2),
                    reason=(
                        f"Exhaustion spike {volume/avg_vol:.1f}× avg vol. "
                        f"RSI {rsi_val:.1f} overbought. Rise {price_move_pct:.2f}% in 3 bars. "
                        f"Upper wick {upper_wick:.0%}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "vol_ratio": round(volume / avg_vol, 2),
                        "rsi_5m": round(rsi_val, 2),
                        "rsi_15m": round(rsi15_curr, 2) if rsi15_curr else None,
                        "price_move_pct": round(price_move_pct, 3),
                        "upper_wick": round(upper_wick, 3),
                        "atr": round(atr_val, 4), "r_r": round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                ))
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


def _score(rsi: float, vol_ratio: float, move_pct: float, wick: float, direction: str, rr: float) -> float:
    base = 0.50
    if direction == "buy":
        if rsi < 25:
            base += 0.15
        elif rsi < 32:
            base += 0.08
    else:
        if rsi > 75:
            base += 0.15
        elif rsi > 68:
            base += 0.08
    if vol_ratio > 4.0:
        base += 0.10
    elif vol_ratio > 3.0:
        base += 0.05
    if wick > 0.60:
        base += 0.06
    if move_pct > 1.0:
        base += 0.05
    if rr >= 2.5:
        base += 0.06
    return min(max(base, 0.0), 1.0)
