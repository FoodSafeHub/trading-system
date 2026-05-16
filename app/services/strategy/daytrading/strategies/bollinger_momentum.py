"""
Bollinger Momentum Breakout — Strategy 6
Canonical spec: docs/strategies_spec.md § "Strategy 6 — BollingerMomentum"

Concept:
    Volatility contraction (BB squeeze) followed by a decisive close above
    the upper band signals an explosive directional move.

Edge:
    Low-volatility coiling periods concentrate energy. When the bands are
    narrowest relative to recent history, a breakout above the upper band
    with confirming RSI and volume has significantly higher follow-through.

Typical trades per day: 0–1 per symbol
Best conditions: BULL_OPEN or CHOPPY (volatility expansion days)
Known weaknesses: Whipsaws on news spikes; false breakouts when contraction
    resolves sideways rather than directionally

Entry filters:
    - BB band width in lowest 20% of last 40 bars (squeeze confirmed)
    - Close above upper band (momentum direction)
    - EMA9 slope positive (trend alignment)
    - RSI 50–70 for longs, 30–50 for shorts (momentum but not overbought)
    - Relative volume >= 1.2x (participation confirms move)
    - Above VWAP for longs, below for shorts

Uses ta.volatility.BollingerBands (same library as existing strategies).
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, compute_vwap, LAST_ENTRY_TIME
from app.services.strategy.daytrading.models import DayTradeSignal

BB_LAST_ENTRY = time(14, 30)


class BollingerMomentum:
    name = "BollingerMomentum"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "bb_length": 20,
        "bb_std": 2.0,
        "contraction_lookback": 40,      # bars to measure squeeze context
        "contraction_percentile": 0.20,  # band width must be in lowest 20%
        "ema_fast": 9,
        "rsi_period": 14,
        "rsi_min_long": 50,
        "rsi_max_long": 70,
        "rsi_min_short": 30,
        "rsi_max_short": 50,
        "vol_rel_min": 1.2,
        "atr_stop_mult": 1.0,
        "r_multiple_target": 2.0,
        "max_hold_bars": 60,
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

        today_bars = _today_bars(df_5m)
        if today_bars.empty or len(today_bars) < cfg["bb_length"] + 4:
            return signals

        df = today_bars.copy()

        # ── Indicators ──────────────────────────────────────────────────────────
        bb = tav.BollingerBands(
            df["Close"],
            window=cfg["bb_length"],
            window_dev=cfg["bb_std"],
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, float("nan"))

        ema_fast = tat.EMAIndicator(df["Close"], window=cfg["ema_fast"]).ema_indicator()
        df["ema_fast"] = ema_fast

        df["rsi"] = tam.RSIIndicator(df["Close"], window=cfg["rsi_period"]).rsi()

        atr_ind = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
        df["atr"] = atr_ind.average_true_range()

        vol_avg = df["Volume"].rolling(20).mean()

        vwap_series = compute_vwap(df)
        df["vwap"] = vwap_series

        signal_seen = False

        for i in range(cfg["contraction_lookback"], len(df)):
            if signal_seen:
                break

            bar = df.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None
            if bar_time and bar_time >= BB_LAST_ENTRY:
                break

            # Skip bars with NaN indicators
            if any(pd.isna(bar[c]) for c in ["bb_upper", "bb_lower", "bb_width", "ema_fast", "rsi", "atr"]):
                continue
            if pd.isna(vol_avg.iloc[i]):
                continue

            close    = float(bar["Close"])
            bb_upper = float(bar["bb_upper"])
            bb_lower = float(bar["bb_lower"])
            bb_width = float(bar["bb_width"])
            ema_val  = float(bar["ema_fast"])
            rsi_val  = float(bar["rsi"])
            atr_val  = float(bar["atr"])
            vwap_val = float(bar["vwap"]) if not pd.isna(bar["vwap"]) else 0.0
            volume   = float(bar["Volume"])
            avg_vol  = float(vol_avg.iloc[i])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            # ── Squeeze filter: band width must be in bottom percentile ─────────
            lookback_widths = df["bb_width"].iloc[i - cfg["contraction_lookback"]: i].dropna()
            if len(lookback_widths) < 10:
                continue
            squeeze_threshold = lookback_widths.quantile(cfg["contraction_percentile"])
            in_squeeze = bb_width <= squeeze_threshold

            vol_ratio = volume / avg_vol

            # ── EMA slope (current bar vs prior) ────────────────────────────────
            ema_prev = float(df["ema_fast"].iloc[i - 1]) if i >= 1 else ema_val

            # ── LONG: close above upper band ────────────────────────────────────
            if (
                in_squeeze
                and close > bb_upper
                and ema_val > ema_prev                          # EMA rising
                and cfg["rsi_min_long"] <= rsi_val <= cfg["rsi_max_long"]
                and vol_ratio >= cfg["vol_rel_min"]
                and (vwap_val <= 0 or close > vwap_val)        # above VWAP
            ):
                entry  = close
                stop   = float(bar["Low"]) - 0.25 * atr_val
                stop   = min(stop, entry - atr_val * cfg["atr_stop_mult"])
                risk   = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * cfg["r_multiple_target"]
                rr     = (target - entry) / risk
                if rr < 1.5:
                    continue

                confidence = _score(rsi_val, vol_ratio, regime, rr, in_squeeze, direction="LONG")
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
                        f"BB squeeze breakout LONG: close {close:.2f} > upper {bb_upper:.2f}. "
                        f"BW {bb_width:.4f} (squeeze threshold {squeeze_threshold:.4f}). "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "bb_upper": round(bb_upper, 4),
                        "bb_lower": round(bb_lower, 4),
                        "bb_width": round(bb_width, 6),
                        "squeeze_threshold": round(squeeze_threshold, 6),
                        "ema_fast": round(ema_val, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                ))
                signal_seen = True
                continue

            # ── SHORT: close below lower band ───────────────────────────────────
            if (
                in_squeeze
                and close < bb_lower
                and ema_val < ema_prev                          # EMA falling
                and cfg["rsi_min_short"] <= rsi_val <= cfg["rsi_max_short"]
                and vol_ratio >= cfg["vol_rel_min"]
                and (vwap_val <= 0 or close < vwap_val)        # below VWAP
            ):
                entry  = close
                stop   = float(bar["High"]) + 0.25 * atr_val
                stop   = max(stop, entry + atr_val * cfg["atr_stop_mult"])
                risk   = stop - entry
                if risk <= 0:
                    continue
                target = entry - risk * cfg["r_multiple_target"]
                rr     = (entry - target) / risk
                if rr < 1.5:
                    continue

                confidence = _score(rsi_val, vol_ratio, regime, rr, in_squeeze, direction="SHORT")
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
                        f"BB squeeze breakout SHORT: close {close:.2f} < lower {bb_lower:.2f}. "
                        f"BW {bb_width:.4f} (squeeze threshold {squeeze_threshold:.4f}). "
                        f"Vol {vol_ratio:.1f}x. RSI {rsi_val:.1f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "bb_upper": round(bb_upper, 4),
                        "bb_lower": round(bb_lower, 4),
                        "bb_width": round(bb_width, 6),
                        "squeeze_threshold": round(squeeze_threshold, 6),
                        "ema_fast": round(ema_val, 4),
                        "rsi": round(rsi_val, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr_val, 4),
                        "r_r": round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                ))
                signal_seen = True

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


def _score(rsi: float, vol_ratio: float, regime: str, rr: float, in_squeeze: bool, direction: str) -> float:
    base = 0.50
    if in_squeeze:
        base += 0.08
    if vol_ratio > 2.0:
        base += 0.10
    elif vol_ratio > 1.5:
        base += 0.05
    if rr >= 3.0:
        base += 0.08
    elif rr >= 2.0:
        base += 0.04
    if direction == "LONG":
        if rsi > 60:
            base += 0.05
        if regime == "BULL_OPEN":
            base += 0.08
        elif regime == "BEAR_OPEN":
            base -= 0.15
    else:
        if rsi < 40:
            base += 0.05
        if regime == "BEAR_OPEN":
            base += 0.08
        elif regime == "BULL_OPEN":
            base -= 0.15
    if regime == "CHOPPY":
        base -= 0.05
    return min(max(base, 0.0), 1.0)
