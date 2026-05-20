"""Four candlestick-pattern momentum strategies (5m/15m daytrading engine).

These strategies look for textbook chart patterns on the most recent closed
bars and fire only when the momentum regime gate (SPY structure + VIX +
breadth) agrees with the direction. Pattern detection lives in
:mod:`app.services.strategy.candle_patterns` and is shared with the Perplexity
swing engine, so a fix to a detector improves both engines at once.

Strategies
----------
* :class:`EngulfingVolumeSurge` — engulfing reversal confirmed by 1.5×+ avg
  volume. Long-only in BULL regimes, short-only in BEAR_MOMENTUM.
* :class:`NarrowRangeBreakout`  — NR4/NR7 compression that resolves on a
  decisive close beyond the prior bar's extreme.
* :class:`ThreeBarPush`         — three consecutive same-direction expanding
  bars closing near their extremes (institutional accumulation/distribution).
* :class:`HammerShootingStar`   — single-bar reversal with long tail at the
  extreme of a multi-bar move. Highest evidence requirement of the four.

All four use the shared :func:`evaluate_tiered_exit_long`-style stops via the
exit manager — entries here only set the *initial* stop (most recent swing).
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam

from app.services.strategy import candle_patterns as cp
from app.services.strategy.daytrading.market_open import ET, LAST_ENTRY_TIME
from app.services.strategy.daytrading.models import DayTradeSignal


# Patterns are unreliable in the first 30m noise window; gate accordingly.
MIN_ENTRY_TIME = time(10, 0)


def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    out = df.copy()
    out.index = idx
    return out[idx.date == idx[-1].date()]


def _bar_ok(bar_time: time | None) -> bool:
    if bar_time is None:
        return True
    return MIN_ENTRY_TIME <= bar_time < LAST_ENTRY_TIME


def _regime_allows_long(regime: str) -> bool:
    return regime in ("BULL_OPEN", "TREND_UP", "BULL_MOMENTUM", "BULL_CAUTION")


def _regime_allows_short(regime: str) -> bool:
    return regime in ("BEAR_OPEN", "TREND_DOWN", "BEAR_MOMENTUM")


# ══════════════════════════════════════════════════════════════════════════════
# 1) Engulfing + Volume Surge
# ══════════════════════════════════════════════════════════════════════════════
class EngulfingVolumeSurge:
    name = "EngulfingVolumeSurge"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "vol_multiple": 1.5,        # current bar volume vs 20-bar avg
        "rsi_period": 14,
        "rsi_long_min": 40,         # don't buy overbought engulfings; not so tight we kill the setup
        "rsi_long_max": 75,
        "rsi_short_min": 25,
        "rsi_short_max": 60,
        "atr_stop_mult": 1.2,
        "tp_atr_mult": 2.4,         # 2R target
        "min_rr": 1.8,
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

        bars = _today_bars(df_5m)
        if len(bars) < 25:
            return signals

        bar_time = bars.index[-1].time() if hasattr(bars.index[-1], "time") else None
        if not _bar_ok(bar_time):
            return signals

        rsi = tam.RSIIndicator(bars["Close"], window=cfg["rsi_period"]).rsi()
        if pd.isna(rsi.iloc[-1]):
            return signals
        rsi_val = float(rsi.iloc[-1])

        vol_ratio = cp.volume_surge_ratio(bars, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return signals

        atr = cp.current_atr(bars, 14)
        if atr <= 0:
            return signals
        close = float(bars["Close"].iloc[-1])

        # Long setup
        if (
            _regime_allows_long(regime)
            and cp.is_bullish_engulfing(bars, -1)
            and cfg["rsi_long_min"] <= rsi_val <= cfg["rsi_long_max"]
        ):
            entry = close
            # Structural stop = engulfing bar low − small ATR buffer. If that's
            # too tight (< 0.5×ATR away), widen to 0.5×ATR so noise doesn't
            # take us out instantly. Cap at atr_stop_mult so risk stays bounded.
            bar_low = float(bars["Low"].iloc[-1])
            stop = max(bar_low - 0.1 * atr, entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "BUY", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"], {"rsi": rsi_val, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"Bullish engulfing on {vol_ratio:.1f}× vol; RSI {rsi_val:.1f}")
            if sig:
                signals.append(sig)

        # Short setup
        elif (
            _regime_allows_short(regime)
            and cp.is_bearish_engulfing(bars, -1)
            and cfg["rsi_short_min"] <= rsi_val <= cfg["rsi_short_max"]
        ):
            entry = close
            bar_high = float(bars["High"].iloc[-1])
            stop = min(bar_high + 0.1 * atr, entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "SELL", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"], {"rsi": rsi_val, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"Bearish engulfing on {vol_ratio:.1f}× vol; RSI {rsi_val:.1f}")
            if sig:
                signals.append(sig)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 2) NR4 / NR7 Breakout
# ══════════════════════════════════════════════════════════════════════════════
class NarrowRangeBreakout:
    """Inside-day / NR4 / NR7 compression resolving on a decisive close.

    The signal fires when the *previous* bar was NR4/NR7 (or inside) and the
    *current* bar closes beyond that bar's high/low with volume — the textbook
    Toby Crabel setup, ported to 5m intraday.
    """
    name = "NarrowRangeBreakout"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "use_nr7": True,
        "vol_multiple": 1.4,
        "atr_stop_mult": 1.5,
        "tp_atr_mult": 3.0,
        "min_rr": 1.8,
        "min_breakout_pct": 0.05,   # current close must clear prior extreme by this % of prior range
    }

    def generate_signals(self, df_5m, df_15m, symbol, config=None, regime="BULL_OPEN"):
        cfg = {**self.default_config, **(config or {})}
        signals: list[DayTradeSignal] = []

        bars = _today_bars(df_5m)
        if len(bars) < 25:
            return signals
        bar_time = bars.index[-1].time() if hasattr(bars.index[-1], "time") else None
        if not _bar_ok(bar_time):
            return signals

        # Compression bar = the bar *before* the current breakout bar.
        is_compressed = (
            cp.is_nr7(bars, -2) if cfg["use_nr7"] else
            (cp.is_nr4(bars, -2) or cp.is_inside_bar(bars, -2))
        )
        if not is_compressed:
            return signals

        prev = bars.iloc[-2]
        cur = bars.iloc[-1]
        atr = cp.current_atr(bars, 14)
        if atr <= 0:
            return signals
        prev_range = float(prev["High"]) - float(prev["Low"])
        if prev_range <= 0:
            return signals

        vol_ratio = cp.volume_surge_ratio(bars, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return signals

        close = float(cur["Close"])
        long_break = close > float(prev["High"]) + prev_range * (cfg["min_breakout_pct"] / 100)
        short_break = close < float(prev["Low"])  - prev_range * (cfg["min_breakout_pct"] / 100)

        if long_break and _regime_allows_long(regime):
            entry = close
            stop = max(float(prev["Low"]) - 0.1 * atr, entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "BUY", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"prev_range": prev_range, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"NR{'7' if cfg['use_nr7'] else '4'} breakout > {prev['High']:.2f}, vol {vol_ratio:.1f}×")
            if sig:
                signals.append(sig)
        elif short_break and _regime_allows_short(regime):
            entry = close
            stop = min(float(prev["High"]) + 0.1 * atr, entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "SELL", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"prev_range": prev_range, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"NR{'7' if cfg['use_nr7'] else '4'} breakdown < {prev['Low']:.2f}, vol {vol_ratio:.1f}×")
            if sig:
                signals.append(sig)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 3) Three-Bar Momentum Push
# ══════════════════════════════════════════════════════════════════════════════
class ThreeBarPush:
    """Three consecutive same-direction bars with expanding ranges and closes
    near the extremes — institutional thrust. We enter on the close of bar 3,
    not on a pullback (those rarely come on real thrusts).
    """
    name = "ThreeBarPush"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "vol_multiple": 1.3,
        "rsi_period": 14,
        "rsi_long_max": 80,    # let strong momentum run; only block blow-off
        "rsi_short_min": 20,
        "atr_stop_mult": 1.5,
        "tp_atr_mult": 3.0,
        "min_rr": 1.8,
    }

    def generate_signals(self, df_5m, df_15m, symbol, config=None, regime="BULL_OPEN"):
        cfg = {**self.default_config, **(config or {})}
        signals: list[DayTradeSignal] = []

        bars = _today_bars(df_5m)
        if len(bars) < 25:
            return signals
        bar_time = bars.index[-1].time() if hasattr(bars.index[-1], "time") else None
        if not _bar_ok(bar_time):
            return signals

        rsi = tam.RSIIndicator(bars["Close"], window=cfg["rsi_period"]).rsi()
        if pd.isna(rsi.iloc[-1]):
            return signals
        rsi_val = float(rsi.iloc[-1])
        vol_ratio = cp.volume_surge_ratio(bars, -1, 20)
        atr = cp.current_atr(bars, 14)
        if atr <= 0 or vol_ratio < cfg["vol_multiple"]:
            return signals

        close = float(bars["Close"].iloc[-1])

        if (
            _regime_allows_long(regime)
            and cp.is_three_bar_push_up(bars, -1)
            and rsi_val < cfg["rsi_long_max"]
        ):
            entry = close
            stop = max(float(bars["Low"].iloc[-3:].min()) - 0.1 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "BUY", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"rsi": rsi_val, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"3-bar push up; RSI {rsi_val:.1f}; vol {vol_ratio:.1f}×")
            if sig:
                signals.append(sig)

        elif (
            _regime_allows_short(regime)
            and cp.is_three_bar_push_down(bars, -1)
            and rsi_val > cfg["rsi_short_min"]
        ):
            entry = close
            stop = min(float(bars["High"].iloc[-3:].max()) + 0.1 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "SELL", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"rsi": rsi_val, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"3-bar push down; RSI {rsi_val:.1f}; vol {vol_ratio:.1f}×")
            if sig:
                signals.append(sig)

        return signals


# ══════════════════════════════════════════════════════════════════════════════
# 4) Hammer / Shooting Star Reversal
# ══════════════════════════════════════════════════════════════════════════════
class HammerShootingStar:
    """Single-bar tail reversal at the extreme of a multi-bar move.

    Requires CONTEXT: a hammer is only meaningful at the bottom of a pullback
    (oversold RSI + prior down-bars). A shooting star only matters at the top
    of an extension. Without context we skip — the pattern alone is too noisy.
    """
    name = "HammerShootingStar"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "rsi_period": 14,
        "rsi_oversold": 38,        # hammer needs prior weakness
        "rsi_overbought": 62,      # shooting star needs prior strength
        "prior_down_bars": 3,      # of the last N bars, majority must be red for hammer
        "prior_up_bars": 3,
        "tail_to_body": 2.0,
        "vol_multiple": 1.2,
        "atr_stop_mult": 1.2,
        "tp_atr_mult": 2.5,
        "min_rr": 1.8,
    }

    def generate_signals(self, df_5m, df_15m, symbol, config=None, regime="BULL_OPEN"):
        cfg = {**self.default_config, **(config or {})}
        signals: list[DayTradeSignal] = []

        bars = _today_bars(df_5m)
        if len(bars) < 25:
            return signals
        bar_time = bars.index[-1].time() if hasattr(bars.index[-1], "time") else None
        if not _bar_ok(bar_time):
            return signals

        rsi = tam.RSIIndicator(bars["Close"], window=cfg["rsi_period"]).rsi()
        if pd.isna(rsi.iloc[-1]):
            return signals
        rsi_val = float(rsi.iloc[-1])
        atr = cp.current_atr(bars, 14)
        if atr <= 0:
            return signals
        vol_ratio = cp.volume_surge_ratio(bars, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return signals

        # Count direction of the prior N bars (excluding current).
        prior = bars.iloc[-(cfg["prior_down_bars"] + 1):-1]
        red_count = int((prior["Close"] < prior["Open"]).sum())
        green_count = int((prior["Close"] > prior["Open"]).sum())
        close = float(bars["Close"].iloc[-1])

        if (
            _regime_allows_long(regime)
            and cp.is_hammer(bars, -1, tail_to_body=cfg["tail_to_body"])
            and rsi_val < cfg["rsi_oversold"]
            and red_count >= cfg["prior_down_bars"] - 1
        ):
            entry = close
            stop = max(float(bars["Low"].iloc[-1]) - 0.05 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.4 * atr:
                stop = entry - 0.4 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "BUY", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"rsi": rsi_val, "red_bars": red_count, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"Hammer after {red_count} red bars; RSI {rsi_val:.1f}")
            if sig:
                signals.append(sig)

        elif (
            _regime_allows_short(regime)
            and cp.is_shooting_star(bars, -1, tail_to_body=cfg["tail_to_body"])
            and rsi_val > cfg["rsi_overbought"]
            and green_count >= cfg["prior_up_bars"] - 1
        ):
            entry = close
            stop = min(float(bars["High"].iloc[-1]) + 0.05 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.4 * atr:
                stop = entry + 0.4 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            sig = _build_signal(self.name, "SELL", symbol, entry, stop, target, regime, bars,
                                cfg["min_rr"],
                                {"rsi": rsi_val, "green_bars": green_count, "vol_ratio": vol_ratio, "atr": atr},
                                reason=f"Shooting star after {green_count} green bars; RSI {rsi_val:.1f}")
            if sig:
                signals.append(sig)

        return signals


# ── shared signal builder (R:R gate + confidence scoring) ──────────────────────

def _build_signal(
    strategy: str,
    direction: str,
    symbol: str,
    entry: float,
    stop: float,
    target: float,
    regime: str,
    bars: pd.DataFrame,
    min_rr: float,
    indicators: dict,
    *,
    reason: str,
) -> DayTradeSignal | None:
    if direction == "BUY":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    if rr < min_rr:
        return None

    conf = 0.55
    if rr >= 3.0:
        conf += 0.10
    elif rr >= 2.2:
        conf += 0.05
    vr = indicators.get("vol_ratio", 0)
    if vr >= 2.5:
        conf += 0.10
    elif vr >= 1.8:
        conf += 0.05
    if regime in ("BULL_MOMENTUM", "BEAR_MOMENTUM"):
        conf += 0.08
    elif regime == "BULL_CAUTION":
        conf -= 0.05

    return DayTradeSignal(
        symbol=symbol,
        strategy=strategy,
        direction=direction,
        timeframe="5m",
        entry_price=round(entry, 4),
        stop_price=round(stop, 4),
        target_price=round(target, 4),
        confidence=round(min(max(conf, 0.0), 1.0), 2),
        reason=f"{reason}. R:R {rr:.1f}",
        regime=regime,
        indicators={k: round(v, 4) if isinstance(v, float) else v for k, v in indicators.items()} | {"r_r": round(rr, 2)},
        signal_time=bars.index[-1].isoformat(),
    )
