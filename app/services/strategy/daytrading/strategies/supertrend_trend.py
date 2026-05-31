"""
Supertrend Trend-Following — Strategy 7
Canonical spec: docs/strategies_spec.md § "Strategy 7 — SupertrendTrend"

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

from typing import Any

import numpy as np
import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import (
    compute_vwap, localize_for_symbol, market_session,
)
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.risk_templates import (
    get_symbol_bucket, supertrend_exit_plan,
)

# No new entries in the last ~45 min before the close (US 15:00 ET, NSE 14:45 IST).
ST_LAST_ENTRY_BEFORE_CLOSE_MIN = 45


class SupertrendTrend:
    name = "SupertrendTrend"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "st_length": 10,
        "st_multiplier": 3.0,       # US; NSE uses 2.5 (see _NSE_ST_MULT)
        "ema_pullback": 13,
        "rsi_period": 14,
        "rsi_min_long": 45,
        "rsi_max_long": 70,
        "rsi_min_short": 30,
        "rsi_max_short": 55,
        "vol_rel_min": 1.1,
        # Stop: ST_line ± 0.1×ATR, capped at entry ± 1.2×ATR.
        # If ST line is further than cap, the trade is skipped (too much risk).
        "st_stop_atr_buffer": 0.10,
        "stop_cap_atr_mult": 1.20,  # cap; NSE uses 1.0×ATR
        "atr_stop_mult": 1.0,       # legacy fallback
        "r_multiple_target": 2.0,
        "pullback_atr_dist": 0.8,
        "max_hold_bars": 60,        # reduced from 80
    }

    _NSE_ST_MULT   = 2.5   # tighter multiplier for NSE's thinner orderbook
    _NSE_STOP_CAP  = 1.00  # tighter stop cap for NSE

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

        bucket = get_symbol_bucket(symbol, market="NSE" if "." in symbol else "US")
        is_nse = "NSE" in bucket

        # NSE: use tighter ST multiplier and stop cap
        st_mult = self._NSE_ST_MULT if is_nse else cfg["st_multiplier"]
        stop_cap_mult = self._NSE_STOP_CAP if is_nse else cfg["stop_cap_atr_mult"]

        min_bars = cfg["st_length"] + cfg["ema_pullback"] + 5
        today_5m = _today_bars(df_5m, symbol)
        if today_5m.empty or len(today_5m) < min_bars:
            return signals

        st_last_entry = market_session(symbol).before_close(ST_LAST_ENTRY_BEFORE_CLOSE_MIN)

        # ── 15m context: compute Supertrend and determine macro bias ────────────
        today_15m = _today_bars(df_15m, symbol) if df_15m is not None and not df_15m.empty else pd.DataFrame()
        if today_15m.empty or len(today_15m) < cfg["st_length"] + 2:
            macro_bullish = regime in ("BULL_OPEN",)
            macro_bearish = regime in ("BEAR_OPEN",)
        else:
            st15 = _compute_supertrend(today_15m, cfg["st_length"], st_mult)
            if st15 is None or st15.empty:
                macro_bullish = regime in ("BULL_OPEN",)
                macro_bearish = regime in ("BEAR_OPEN",)
            else:
                last_direction = int(st15["direction"].iloc[-1])
                # ── Relaxed 15m filter: ST bullish OR EMA21 rising 3+ bars ──
                # Old rule required BOTH; first 2-3 pullbacks after a new trend
                # are now capturable without waiting for 15m ST to flip.
                st15_bullish = last_direction == 1
                st15_bearish = last_direction == -1
                # EMA21 slope check on 15m as alternative macro confirmation
                ema21_15m = tat.EMAIndicator(today_15m["Close"], window=21).ema_indicator()
                ema21_rising = (
                    len(ema21_15m) >= 3
                    and not pd.isna(ema21_15m.iloc[-1])
                    and float(ema21_15m.iloc[-1]) > float(ema21_15m.iloc[-2]) > float(ema21_15m.iloc[-3])
                )
                ema21_falling = (
                    len(ema21_15m) >= 3
                    and not pd.isna(ema21_15m.iloc[-1])
                    and float(ema21_15m.iloc[-1]) < float(ema21_15m.iloc[-2]) < float(ema21_15m.iloc[-3])
                )
                macro_bullish = st15_bullish or ema21_rising
                macro_bearish = st15_bearish or ema21_falling

        # ── 5m indicators ───────────────────────────────────────────────────────
        df = today_5m.copy()
        st5 = _compute_supertrend(df, cfg["st_length"], st_mult)
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
            if bar_time and bar_time >= st_last_entry:
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
                and st_dir == 1
                and cfg["rsi_min_long"] <= rsi_val <= cfg["rsi_max_long"]
                and vol_ratio >= cfg["vol_rel_min"]
                and close > open_
                and (
                    abs(low_ - st_line) <= pb_dist
                    or abs(low_ - ema_val) <= pb_dist
                )
            ):
                entry = close
                # ── Stop: ST line − buffer×ATR, capped at entry − cap×ATR ──
                st_stop     = st_line - cfg["st_stop_atr_buffer"] * atr_val
                cap_stop    = entry - stop_cap_mult * atr_val
                stop        = max(st_stop, cap_stop)   # tighter of the two
                # If ST line is much further than cap, risk is too wide — skip
                if st_stop < cap_stop - 0.5 * atr_val:
                    continue
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * cfg["r_multiple_target"]
                rr     = (target - entry) / risk
                if rr < 1.5:
                    continue

                exit_plan = supertrend_exit_plan(
                    bucket=bucket, entry=entry, stop=stop, direction="BUY",
                )
                confidence = _score(rsi_val, vol_ratio, regime, rr, direction="LONG")
                signals.append(DayTradeSignal(
                    symbol=symbol, strategy=self.name,
                    direction="BUY", timeframe=self.timeframe,
                    entry_price=round(entry, 4), stop_price=round(stop, 4),
                    target_price=round(target, 4), confidence=round(confidence, 2),
                    reason=(
                        f"Supertrend LONG pullback: macro bullish (ST/EMA21), "
                        f"5m ST={st_line:.2f} dir=+1. "
                        f"Low {low_:.2f} near ST/EMA={ema_val:.2f}. "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}. [{bucket}]"
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
                        "st_multiplier": st_mult,
                        "bucket": bucket,
                        "exit_plan": exit_plan.to_dict(),
                    },
                    signal_time=bar.name.isoformat(),
                    exit_plan=exit_plan,
                ))
                signal_seen = True
                continue

            # ── SHORT: macro bearish + 5m ST bearish + pullback ──────────────
            if (
                macro_bearish
                and st_dir == -1
                and cfg["rsi_min_short"] <= rsi_val <= cfg["rsi_max_short"]
                and vol_ratio >= cfg["vol_rel_min"]
                and close < open_
                and (
                    abs(high_ - st_line) <= pb_dist
                    or abs(high_ - ema_val) <= pb_dist
                )
            ):
                entry    = close
                st_stop  = st_line + cfg["st_stop_atr_buffer"] * atr_val
                cap_stop = entry + stop_cap_mult * atr_val
                stop     = min(st_stop, cap_stop)
                if st_stop > cap_stop + 0.5 * atr_val:
                    continue
                risk = stop - entry
                if risk <= 0:
                    continue
                target = entry - risk * cfg["r_multiple_target"]
                rr     = (entry - target) / risk
                if rr < 1.5:
                    continue

                exit_plan = supertrend_exit_plan(
                    bucket=bucket, entry=entry, stop=stop, direction="SELL",
                )
                confidence = _score(rsi_val, vol_ratio, regime, rr, direction="SHORT")
                signals.append(DayTradeSignal(
                    symbol=symbol, strategy=self.name,
                    direction="SELL", timeframe=self.timeframe,
                    entry_price=round(entry, 4), stop_price=round(stop, 4),
                    target_price=round(target, 4), confidence=round(confidence, 2),
                    reason=(
                        f"Supertrend SHORT pullback: macro bearish (ST/EMA21), "
                        f"5m ST={st_line:.2f} dir=-1. "
                        f"High {high_:.2f} near ST/EMA={ema_val:.2f}. "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}. [{bucket}]"
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
                        "st_multiplier": st_mult,
                        "bucket": bucket,
                        "exit_plan": exit_plan.to_dict(),
                    },
                    signal_time=bar.name.isoformat(),
                    exit_plan=exit_plan,
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


def _today_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    df = localize_for_symbol(df, symbol)
    idx = df.index
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
