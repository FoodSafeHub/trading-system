"""
VWAP Mean Reversion — Strategy 2

Concept:
    Price tends to revert to VWAP after extended deviations.
    Enter when price drops below VWAP with oversold RSI and reversal evidence.

Edge:
    Institutional algorithms use VWAP as a benchmark — they buy when price
    drops below it, creating the reversion force we trade.

Typical trades per day: 0.5–2 per symbol
Best conditions: BULL_OPEN, liquid names (SPY, QQQ, AAPL, MSFT)
Known weaknesses: Trending days where price never returns to VWAP

Reality calibration:
    SPY intraday moves 0.1–0.3% from VWAP on most days.
    A 0.6% VWAP deviation on SPY is rare (strong trend day).
    A 0.3% deviation with RSI < 38 is tradeable and occurs ~2x per week.
    Thresholds must scale to the symbol's typical intraday ATR%.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, compute_vwap
from app.services.strategy.daytrading.models import DayTradeSignal

VWAP_MR_LAST_ENTRY = time(14, 30)
VWAP_MR_MIN_START = time(9, 45)   # skip first 15 min


class VWAPMeanReversion:
    name = "VWAPMeanReversion"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        # ATR-relative distance: price must be this many ATR units below VWAP
        # 0.4×ATR works on SPY (~0.2–0.3% on a normal day) and scales to TSLA/NVDA
        "vwap_atr_distance": 0.4,
        "rsi_period": 14,
        "rsi_oversold": 38,         # realistic for SPY (25 almost never happens)
        "wick_ratio_min": 0.40,     # lower wick > 40% of range
        "vol_ratio_min": 1.3,       # modest volume uptick
        "atr_stop_mult": 1.0,
        "atr_target_mult": 1.8,     # target = entry + 1.8×ATR (past VWAP)
        "max_hold_bars": 20,
        "min_rr": 1.5,
        # VWAP slope: reject if VWAP is dropping faster than this % per bar
        "max_vwap_drop_pct": 0.05,
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

        if regime != "BULL_OPEN":
            return signals

        today = _today_bars(df_5m)
        if today.empty or len(today) < 6:  # NaN guard in loop handles rolling-20 warmup; was 20 (blocked until 10:10 AM)
            return signals

        today = today.copy()
        today["vwap"] = compute_vwap(today)
        today["rsi"] = tam.RSIIndicator(today["Close"], window=cfg["rsi_period"]).rsi()
        atr_ind = tav.AverageTrueRange(today["High"], today["Low"], today["Close"], window=14)
        today["atr"] = atr_ind.average_true_range()
        vol_avg = today["Volume"].rolling(20).mean()

        open_signal_seen = False

        for i in range(3, len(today)):
            if open_signal_seen:
                break

            bar = today.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None

            if bar_time and (bar_time < VWAP_MR_MIN_START or bar_time >= VWAP_MR_LAST_ENTRY):
                continue

            if any(pd.isna(bar[c]) for c in ["rsi", "vwap", "atr"]):
                continue
            if pd.isna(vol_avg.iloc[i]):
                continue

            close = float(bar["Close"])
            open_p = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            vwap = float(bar["vwap"])
            rsi_val = float(bar["rsi"])
            atr_val = float(bar["atr"])
            volume = float(bar["Volume"])
            avg_vol = float(vol_avg.iloc[i])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            # ATR-relative distance — adapts to each symbol's volatility
            vwap_dist_atr = (vwap - close) / atr_val
            vwap_dist_pct = (vwap - close) / vwap * 100

            if vwap_dist_atr < cfg["vwap_atr_distance"]:
                continue

            bar_range = high - low
            wick_ratio = (close - low) / bar_range if bar_range > 0 else 0.0
            vol_ratio = volume / avg_vol

            # Prior bar also below VWAP
            prev_close = float(today.iloc[i - 1]["Close"])
            prev_vwap = float(today.iloc[i - 1]["vwap"])
            if prev_close >= prev_vwap:
                continue

            # VWAP must not be in freefall
            if i >= 3:
                vwap_3_ago = float(today.iloc[i - 3]["vwap"])
                if vwap_3_ago > 0:
                    vwap_drop_pct = (vwap_3_ago - vwap) / vwap_3_ago * 100
                    if vwap_drop_pct > cfg["max_vwap_drop_pct"]:
                        continue

            # Entry bar must close green (buyers responding)
            if close <= open_p:
                continue

            if (
                rsi_val < cfg["rsi_oversold"]
                and wick_ratio > cfg["wick_ratio_min"]
                and vol_ratio > cfg["vol_ratio_min"]
            ):
                entry = close
                stop = entry - cfg["atr_stop_mult"] * atr_val
                target = vwap + cfg["atr_target_mult"] * atr_val * 0.5  # target past VWAP

                risk = entry - stop
                if risk <= 0:
                    continue
                rr = (target - entry) / risk
                if rr < cfg["min_rr"]:
                    continue

                confidence = _score(rsi_val, vwap_dist_atr, vol_ratio, rr)

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
                            f"Price {vwap_dist_pct:.2f}% ({vwap_dist_atr:.1f}×ATR) below VWAP {vwap:.2f}. "
                            f"RSI {rsi_val:.1f}. Wick {wick_ratio:.0%}. Vol {vol_ratio:.1f}×. R:R {rr:.1f}."
                        ),
                        regime=regime,
                        indicators={
                            "vwap": round(vwap, 4),
                            "vwap_dist_pct": round(vwap_dist_pct, 3),
                            "vwap_dist_atr": round(vwap_dist_atr, 2),
                            "rsi": round(rsi_val, 2),
                            "wick_ratio": round(wick_ratio, 3),
                            "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr_val, 4),
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


def _score(rsi: float, dist_atr: float, vol_ratio: float, rr: float) -> float:
    base = 0.50
    if rsi < 28:
        base += 0.15
    elif rsi < 33:
        base += 0.08
    if dist_atr > 0.8:
        base += 0.10
    elif dist_atr > 0.6:
        base += 0.05
    if vol_ratio > 2.0:
        base += 0.08
    if rr >= 2.5:
        base += 0.07
    return min(max(base, 0.0), 1.0)
