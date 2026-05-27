"""
EMA Momentum — Strategy 3 (9/21 EMA intraday trend)

Concept:
    Two entry modes on 15m bars:
    A) Fresh EMA9/21 crossover with positive MACD histogram
    B) EMA9 bounce — EMAs already bullish aligned, price pulls back to EMA9
       and closes above it (trend continuation). Far more frequent than A.

Edge:
    EMA9 on 15m is a widely-watched institutional level. Both the crossover
    and the pullback-to-hold generate consistent intraday momentum trades.

Typical trades per day: 1–3 per symbol
Best conditions: Trending sessions post-10 AM
Known weaknesses: Choppy/range-bound days, first 30 min of session

Reality calibration:
    EMA9/21 crossovers on 15m SPY happen only a few times per month.
    The EMA9 bounce fires multiple times per week and is the primary driver
    of this strategy's trade frequency.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import (
    compute_vwap, localize_for_symbol, market_session,
)
from app.services.strategy.daytrading.models import DayTradeSignal

# Entry window as offsets from the session open (works on both US and NSE):
# skip first 30 min; no new entries after open+4.5h.
EMA_START_OFFSET_MIN = 30
EMA_LAST_ENTRY_OFFSET_MIN = 270


class EMAMomentum:
    name = "EMAMomentum"
    timeframe = "15m"
    default_config: dict[str, Any] = {
        "ema_fast": 9,
        "ema_slow": 21,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_low": 40,
        "rsi_high": 70,
        "atr_stop_mult": 1.2,
        "atr_tp_mult": 2.5,
        "min_rr": 1.8,
        "max_hold_bars": 16,
        "ema9_bounce_atr": 0.3,   # low must come within 0.3×ATR of EMA9
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

        # Compute indicators on the full multi-day 15m history so EMAs are
        # warmed up even during the first hour of today's session.
        # BUG FIX: using today-only bars meant ema_slow=21 required 24 today-bars
        # = 6 hours into the session before any signal was possible.
        all_15m = _localize(df_15m, symbol)
        if all_15m.empty or len(all_15m) < cfg["ema_slow"] + 3:
            return signals

        _sess = market_session(symbol)
        ema_min_start = _sess.after_open(EMA_START_OFFSET_MIN)
        ema_last_entry = _sess.after_open(EMA_LAST_ENTRY_OFFSET_MIN)

        today_date = all_15m.index[-1].date()

        all_15m = all_15m.copy()
        all_15m["ema_fast"] = tat.EMAIndicator(all_15m["Close"], window=cfg["ema_fast"]).ema_indicator()
        all_15m["ema_slow"] = tat.EMAIndicator(all_15m["Close"], window=cfg["ema_slow"]).ema_indicator()

        # Restrict the scan window to today's bars only
        today = all_15m[all_15m.index.date == today_date]
        if today.empty:
            return signals

        today = today.copy()
        # Slice the pre-computed multi-day series to today's index
        today["ema_fast"] = all_15m["ema_fast"].reindex(today.index)
        today["ema_slow"] = all_15m["ema_slow"].reindex(today.index)
        # RSI, MACD, ATR computed on full history; VWAP resets per day so today-only is correct
        all_15m["rsi"] = tam.RSIIndicator(all_15m["Close"], window=cfg["rsi_period"]).rsi()
        macd_ind = tat.MACD(all_15m["Close"],
                            window_fast=cfg["macd_fast"],
                            window_slow=cfg["macd_slow"],
                            window_sign=cfg["macd_signal"])
        all_15m["macd_hist"] = macd_ind.macd_diff()
        all_15m["atr"] = tav.AverageTrueRange(
            all_15m["High"], all_15m["Low"], all_15m["Close"], window=14
        ).average_true_range()

        today["rsi"]       = all_15m["rsi"].reindex(today.index)
        today["macd_hist"] = all_15m["macd_hist"].reindex(today.index)
        today["atr"]       = all_15m["atr"].reindex(today.index)
        today["vwap"]      = compute_vwap(today)

        open_signal_seen = False

        for i in range(2, len(today)):
            if open_signal_seen:
                break

            bar = today.iloc[i]
            prev = today.iloc[i - 1]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None
            if bar_time and (bar_time < ema_min_start or bar_time >= ema_last_entry):
                continue

            cols = ["ema_fast", "ema_slow", "rsi", "macd_hist", "vwap", "atr"]
            if any(pd.isna(bar[c]) for c in cols):
                continue
            if any(pd.isna(prev[c]) for c in ["ema_fast", "ema_slow", "macd_hist"]):
                continue

            ema_f = float(bar["ema_fast"])
            ema_s = float(bar["ema_slow"])
            prev_ef = float(prev["ema_fast"])
            prev_es = float(prev["ema_slow"])
            rsi_val = float(bar["rsi"])
            hist_val = float(bar["macd_hist"])
            vwap = float(bar["vwap"])
            close = float(bar["Close"])
            low = float(bar["Low"])
            high = float(bar["High"])
            atr_val = float(bar["atr"])
            prev_close = float(prev["Close"])
            prev_ef_val = float(prev["ema_fast"])

            if atr_val <= 0:
                continue

            # ── BUY setups ────────────────────────────────────────
            bullish_aligned = ema_f > ema_s
            ema_cross_up = prev_ef <= prev_es and ema_f > ema_s

            # Setup A: Fresh crossover
            setup_a = (
                ema_cross_up
                and hist_val > 0
                and cfg["rsi_low"] <= rsi_val <= cfg["rsi_high"]
                and close > vwap
            )

            # Setup B: EMA9 bounce (trend continuation)
            setup_b = (
                bullish_aligned
                and not ema_cross_up
                and low <= ema_f + cfg["ema9_bounce_atr"] * atr_val
                and close > ema_f
                and prev_close > prev_ef_val
                and hist_val > 0
                and cfg["rsi_low"] <= rsi_val <= cfg["rsi_high"]
                and close > vwap
            )

            if regime in ("BULL_OPEN", "CHOPPY") and (setup_a or setup_b):
                entry = close
                stop = ema_f - cfg["atr_stop_mult"] * atr_val
                target = entry + cfg["atr_tp_mult"] * atr_val
                risk = entry - stop
                if risk <= 0:
                    continue
                rr = (target - entry) / risk
                if rr < cfg["min_rr"]:
                    continue

                reason_tag = "Fresh EMA9/21 crossover" if setup_a else "EMA9 bounce — trend continuation"
                confidence = _score(rsi_val, ema_f - ema_s, hist_val, regime, rr, setup_a)

                signals.append(DayTradeSignal(
                    symbol=symbol, strategy=self.name,
                    direction="BUY", timeframe=self.timeframe,
                    entry_price=round(entry, 4), stop_price=round(stop, 4),
                    target_price=round(target, 4), confidence=round(confidence, 2),
                    reason=f"{reason_tag}. MACD hist {hist_val:.4f}. RSI {rsi_val:.1f}. Above VWAP. R:R {rr:.1f}.",
                    regime=regime,
                    indicators={
                        "ema_fast": round(ema_f, 4), "ema_slow": round(ema_s, 4),
                        "ema_spread": round(ema_f - ema_s, 4),
                        "rsi": round(rsi_val, 2), "macd_hist": round(hist_val, 5),
                        "vwap": round(vwap, 4), "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                        "setup": "crossover" if setup_a else "ema9_bounce",
                    },
                    signal_time=bar.name.isoformat(),
                ))
                open_signal_seen = True

            # ── SELL SHORT (BEAR_OPEN) ──────────────────────────────
            elif not open_signal_seen and regime == "BEAR_OPEN":
                bearish_aligned = ema_f < ema_s
                ema_cross_dn = prev_ef >= prev_es and ema_f < ema_s

                setup_short_a = (
                    ema_cross_dn and hist_val < 0
                    and 30 <= rsi_val <= 60 and close < vwap
                )
                setup_short_b = (
                    bearish_aligned and not ema_cross_dn
                    and high >= ema_f - cfg["ema9_bounce_atr"] * atr_val
                    and close < ema_f and prev_close < prev_ef_val
                    and hist_val < 0 and 30 <= rsi_val <= 60 and close < vwap
                )

                if setup_short_a or setup_short_b:
                    entry = close
                    stop = ema_f + cfg["atr_stop_mult"] * atr_val
                    target = entry - cfg["atr_tp_mult"] * atr_val
                    risk = stop - entry
                    if risk <= 0:
                        continue
                    rr = (entry - target) / risk
                    if rr < cfg["min_rr"]:
                        continue

                    reason_tag = "Fresh EMA9/21 bearish crossover" if setup_short_a else "EMA9 resistance bounce"
                    confidence = _score(rsi_val, ema_s - ema_f, abs(hist_val), regime, rr, setup_short_a)

                    signals.append(DayTradeSignal(
                        symbol=symbol, strategy=self.name,
                        direction="SELL", timeframe=self.timeframe,
                        entry_price=round(entry, 4), stop_price=round(stop, 4),
                        target_price=round(target, 4), confidence=round(confidence, 2),
                        reason=f"{reason_tag}. MACD hist {hist_val:.4f}. RSI {rsi_val:.1f}. Below VWAP. R:R {rr:.1f}.",
                        regime=regime,
                        indicators={
                            "ema_fast": round(ema_f, 4), "ema_slow": round(ema_s, 4),
                            "rsi": round(rsi_val, 2), "macd_hist": round(hist_val, 5),
                            "vwap": round(vwap, 4), "atr": round(atr_val, 4), "r_r": round(rr, 2),
                        },
                        signal_time=bar.name.isoformat(),
                    ))
                    open_signal_seen = True

        return signals


def _localize(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    return localize_for_symbol(df, symbol)


def _today_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    df = _localize(df, symbol)
    if df.empty:
        return df
    today = df.index[-1].date()
    return df[df.index.date == today]


def _score(rsi: float, ema_spread: float, hist_val: float, regime: str, rr: float, is_crossover: bool) -> float:
    base = 0.52
    if ema_spread > 1.0:
        base += 0.08
    elif ema_spread > 0.5:
        base += 0.04
    if abs(hist_val) > 0.05:
        base += 0.08
    if is_crossover:
        base += 0.05
    if rr >= 3.0:
        base += 0.08
    elif rr >= 2.0:
        base += 0.04
    if regime == "BULL_OPEN":
        base += 0.05
    elif regime == "CHOPPY":
        base -= 0.08
    return min(max(base, 0.0), 1.0)
