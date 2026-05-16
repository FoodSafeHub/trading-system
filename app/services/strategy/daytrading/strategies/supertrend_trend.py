"""
Supertrend Trend-Following — Strategy 7

Concept:
    The 15m Supertrend establishes the macro bias for the session. A 5m
    pullback to the Supertrend line or EMA20 with a bullish reclaim candle
    offers a low-risk entry in the direction of the dominant trend.

Edge:
    The Supertrend filters out counter-trend trades, dramatically improving
    win rate. Pullback entries give tighter stops vs chasing breakouts.
    Trailing the 5m Supertrend line manages the exit dynamically.

Typical trades per day: 0–2 per symbol
Best conditions: BULL_OPEN / BEAR_OPEN (trending days); poor in CHOPPY
Known weaknesses: Whipsaws when 15m Supertrend flips mid-day

Entry filters (LONG):
    - 15m Supertrend bullish (price above 15m ST line)
    - 5m close pulls back to within 0.5 × ATR of the 5m ST line or EMA20
    - Bullish reclaim candle: close > open on the trigger bar
    - RSI 45–70
    - Relative volume >= 1.1x

Stop: just below 5m Supertrend line (1.5 × ATR below entry)
Target: 2.0R from entry

Supertrend is computed manually (ATR-based) since the `ta` library does not
include it natively.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, compute_vwap
from app.services.strategy.daytrading.models import DayTradeSignal

ST_LAST_ENTRY = time(15, 0)


class SupertrendTrend:
    name = "SupertrendTrend"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "st_length": 10,
        "st_multiplier": 3.0,
        "ema_pullback": 20,
        "rsi_period": 14,
        "rsi_min_long": 45,
        "rsi_max_long": 70,
        "rsi_min_short": 30,
        "rsi_max_short": 55,
        "vol_rel_min": 1.1,
        "atr_stop_mult": 1.5,
        "r_multiple_target": 2.0,
        "pullback_atr_dist": 0.5,   # how close to ST/EMA qualifies as "pullback"
        "max_hold_bars": 80,
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

        # Supertrend needs at least st_length + a few bars
        min_bars = cfg["st_length"] + cfg["ema_pullback"] + 5
        today_5m = _today_bars(df_5m)
        if today_5m.empty or len(today_5m) < min_bars:
            return signals

        # ── 15m context: compute Supertrend and determine macro bias ────────────
        today_15m = _today_bars(df_15m) if df_15m is not None and not df_15m.empty else pd.DataFrame()
        if today_15m.empty or len(today_15m) < cfg["st_length"] + 2:
            # Fall back to 5m if 15m unavailable
            macro_bullish = regime in ("BULL_OPEN",)
            macro_bearish = regime in ("BEAR_OPEN",)
        else:
            st15 = _compute_supertrend(today_15m, cfg["st_length"], cfg["st_multiplier"])
            if st15 is None or st15.empty:
                macro_bullish = regime in ("BULL_OPEN",)
                macro_bearish = regime in ("BEAR_OPEN",)
            else:
                last_direction = int(st15["direction"].iloc[-1])
                macro_bullish = last_direction == 1
                macro_bearish = last_direction == -1

        # ── 5m indicators ───────────────────────────────────────────────────────
        df = today_5m.copy()
        st5 = _compute_supertrend(df, cfg["st_length"], cfg["st_multiplier"])
        if st5 is None or st5.empty:
            return signals
        df["st_line"]  = st5["supertrend"]
        df["st_dir"]   = st5["direction"]

        df["ema_pb"] = tat.EMAIndicator(df["Close"], window=cfg["ema_pullback"]).ema_indicator()
        df["rsi"]    = tam.RSIIndicator(df["Close"], window=cfg["rsi_period"]).rsi()
        atr_ind      = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
        df["atr"]    = atr_ind.average_true_range()
        vol_avg      = df["Volume"].rolling(20).mean()

        signal_seen = False

        for i in range(cfg["ema_pullback"] + cfg["st_length"], len(df)):
            if signal_seen:
                break

            bar = df.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None
            if bar_time and bar_time >= ST_LAST_ENTRY:
                break

            if any(pd.isna(bar[c]) for c in ["st_line", "st_dir", "ema_pb", "rsi", "atr"]):
                continue
            if pd.isna(vol_avg.iloc[i]):
                continue

            close    = float(bar["Close"])
            open_    = float(bar["Open"])
            high_    = float(bar["High"])
            low_     = float(bar["Low"])
            st_line  = float(bar["st_line"])
            st_dir   = int(bar["st_dir"])
            ema_val  = float(bar["ema_pb"])
            rsi_val  = float(bar["rsi"])
            atr_val  = float(bar["atr"])
            volume   = float(bar["Volume"])
            avg_vol  = float(vol_avg.iloc[i])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            vol_ratio = volume / avg_vol
            pb_dist   = cfg["pullback_atr_dist"] * atr_val

            # ── LONG: 15m bullish + 5m ST bullish + pullback to ST/EMA ──────────
            if (
                macro_bullish
                and st_dir == 1                                         # 5m ST also bullish
                and cfg["rsi_min_long"] <= rsi_val <= cfg["rsi_max_long"]
                and vol_ratio >= cfg["vol_rel_min"]
                and close > open_                                        # bullish reclaim candle
                and (
                    abs(low_ - st_line) <= pb_dist                      # touched near ST line
                    or abs(low_ - ema_val) <= pb_dist                   # or touched near EMA
                )
            ):
                entry  = close
                stop   = st_line - 0.1 * atr_val                        # just below ST line
                stop   = min(stop, entry - cfg["atr_stop_mult"] * atr_val)
                risk   = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * cfg["r_multiple_target"]
                rr     = (target - entry) / risk
                if rr < 1.5:
                    continue

                confidence = _score(rsi_val, vol_ratio, regime, rr, direction="LONG")
                signals.append(DayTradeSignal(
                    symbol=symbol,
                    strategy=self.name,
                    direction="BUY",
                    timeframe=self.timeframe,
                    entry_price=round(entry, 4),
                    stop_price=round(stop, 4),
                    target_price=round(target, 4),
                    confidence=round(confidence, 2),
                    reason=(
                        f"Supertrend LONG pullback: 15m macro bullish, 5m ST={st_line:.2f} bullish. "
                        f"Pullback bar low {low_:.2f} near ST/EMA20={ema_val:.2f}. "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "st_line_5m": round(st_line, 4),
                        "st_direction": st_dir,
                        "ema_pullback": round(ema_val, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                        "macro_bullish": macro_bullish,
                    },
                    signal_time=bar.name.isoformat(),
                ))
                signal_seen = True
                continue

            # ── SHORT: 15m bearish + 5m ST bearish + pullback to ST/EMA ─────────
            if (
                macro_bearish
                and st_dir == -1                                         # 5m ST also bearish
                and cfg["rsi_min_short"] <= rsi_val <= cfg["rsi_max_short"]
                and vol_ratio >= cfg["vol_rel_min"]
                and close < open_                                         # bearish candle
                and (
                    abs(high_ - st_line) <= pb_dist
                    or abs(high_ - ema_val) <= pb_dist
                )
            ):
                entry  = close
                stop   = st_line + 0.1 * atr_val
                stop   = max(stop, entry + cfg["atr_stop_mult"] * atr_val)
                risk   = stop - entry
                if risk <= 0:
                    continue
                target = entry - risk * cfg["r_multiple_target"]
                rr     = (entry - target) / risk
                if rr < 1.5:
                    continue

                confidence = _score(rsi_val, vol_ratio, regime, rr, direction="SHORT")
                signals.append(DayTradeSignal(
                    symbol=symbol,
                    strategy=self.name,
                    direction="SELL",
                    timeframe=self.timeframe,
                    entry_price=round(entry, 4),
                    stop_price=round(stop, 4),
                    target_price=round(target, 4),
                    confidence=round(confidence, 2),
                    reason=(
                        f"Supertrend SHORT pullback: 15m macro bearish, 5m ST={st_line:.2f} bearish. "
                        f"Pullback bar high {high_:.2f} near ST/EMA20={ema_val:.2f}. "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "st_line_5m": round(st_line, 4),
                        "st_direction": st_dir,
                        "ema_pullback": round(ema_val, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                        "macro_bearish": macro_bearish,
                    },
                    signal_time=bar.name.isoformat(),
                ))
                signal_seen = True

        return signals


# ── Supertrend computation ────────────────────────────────────────────────────

def _compute_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame | None:
    """
    ATR-based Supertrend indicator.
    Returns DataFrame with columns: supertrend, direction (1=bullish, -1=bearish).
    """
    try:
        if len(df) < length + 1:
            return None

        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)
        n     = len(close)

        # Wilder's ATR
        tr = np.maximum(high - low,
             np.maximum(np.abs(high - np.roll(close, 1)),
                        np.abs(low  - np.roll(close, 1))))
        tr[0] = high[0] - low[0]

        atr = np.zeros(n)
        # seed with simple average
        atr[length - 1] = tr[:length].mean()
        for j in range(length, n):
            atr[j] = (atr[j - 1] * (length - 1) + tr[j]) / length

        hl2    = (high + low) / 2
        upper  = hl2 + multiplier * atr
        lower  = hl2 - multiplier * atr

        supertrend = np.zeros(n)
        direction  = np.ones(n, dtype=int)   # 1 = bullish

        for j in range(1, n):
            # Upper band
            if upper[j] < upper[j - 1] or close[j - 1] > upper[j - 1]:
                upper[j] = upper[j]
            else:
                upper[j] = upper[j - 1]

            # Lower band
            if lower[j] > lower[j - 1] or close[j - 1] < lower[j - 1]:
                lower[j] = lower[j]
            else:
                lower[j] = lower[j - 1]

            if supertrend[j - 1] == upper[j - 1]:
                if close[j] <= upper[j]:
                    supertrend[j] = upper[j]
                    direction[j]  = -1
                else:
                    supertrend[j] = lower[j]
                    direction[j]  = 1
            else:
                if close[j] >= lower[j]:
                    supertrend[j] = lower[j]
                    direction[j]  = 1
                else:
                    supertrend[j] = upper[j]
                    direction[j]  = -1

        # Seed first bar
        supertrend[0] = upper[0] if close[0] < hl2[0] else lower[0]
        direction[0]  = -1 if close[0] < hl2[0] else 1

        return pd.DataFrame(
            {"supertrend": supertrend, "direction": direction},
            index=df.index,
        )
    except Exception:
        return None


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


def _score(rsi: float, vol_ratio: float, regime: str, rr: float, direction: str) -> float:
    base = 0.52
    if vol_ratio > 2.0:
        base += 0.10
    elif vol_ratio > 1.5:
        base += 0.05
    if rr >= 3.0:
        base += 0.08
    elif rr >= 2.0:
        base += 0.04
    if direction == "LONG":
        if 55 <= rsi <= 65:
            base += 0.06
        if regime == "BULL_OPEN":
            base += 0.10
        elif regime == "BEAR_OPEN":
            base -= 0.20
    else:
        if 35 <= rsi <= 45:
            base += 0.06
        if regime == "BEAR_OPEN":
            base += 0.10
        elif regime == "BULL_OPEN":
            base -= 0.20
    if regime == "CHOPPY":
        base -= 0.10
    return min(max(base, 0.0), 1.0)
