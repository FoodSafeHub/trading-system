"""Daily-candle momentum strategies for the Perplexity swing engine.

Four pattern-based strategies that mirror the daytrading 5m versions, applied
to daily bars. They share the detectors in
:mod:`app.services.strategy.candle_patterns` so any tweak to a pattern shows up
in both engines.

Strategies
----------
* :class:`PerpEngulfingVolumeSurge`  — daily engulfing + above-average volume
* :class:`PerpNarrowRangeBreakout`   — NR4/NR7 daily-range compression
* :class:`PerpThreeBarPush`          — 3 expanding daily bars closing strong
* :class:`PerpHammerShootingStar`    — hammer/star at the extreme of a swing

All four use the *momentum* regime gate (BULL_MOMENTUM / BULL_CAUTION /
BEAR_MOMENTUM) rather than the simpler MarketRegime — momentum strategies
need the tighter filter (SPY > 50DMA, VIX < 25, breadth ≥ 50%) before they
fire long, otherwise they get chopped up in transitional tape.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.services.market_regime import MarketRegime
from app.services.market_regime_advanced import (
    MomentumRegime,
    get_momentum_regime,
)
from app.services.markets import is_india_symbol
from app.services.strategy import candle_patterns as cp
from app.services.strategy.perplexity.base import (
    PerplexitySignal,
    PerplexityStrategy,
)


# ── Regime gating helpers ─────────────────────────────────────────────────────

def _momentum_snapshot(symbol: str | None = None):
    """Regime snapshot for the symbol's home market.

    India symbols gate on Nifty 50 / India VIX; everything else on SPY / ^VIX.
    Picking the wrong benchmark silently corrupts the gate (e.g. an Indian
    stock being suppressed because the US tape is weak), so route by symbol.
    """
    try:
        market = "india" if (symbol and is_india_symbol(symbol)) else "us"
        return get_momentum_regime(market=market)
    except Exception:
        return None


def _regime_allows_long(snap) -> bool:
    if snap is None:
        return True  # degrade gracefully if regime fetch failed
    return snap.regime in (MomentumRegime.BULL_MOMENTUM, MomentumRegime.BULL_CAUTION)


def _regime_allows_short(snap) -> bool:
    if snap is None:
        return False
    return snap.regime == MomentumRegime.BEAR_MOMENTUM


def _confidence_adjust(snap, base: float) -> float:
    """Trim confidence in BULL_CAUTION, boost in clean BULL_MOMENTUM."""
    if snap is None:
        return base
    if snap.regime == MomentumRegime.BULL_MOMENTUM:
        return min(0.95, base + 0.05)
    if snap.regime == MomentumRegime.BULL_CAUTION:
        return max(0.40, base - 0.10)
    if snap.regime == MomentumRegime.BEAR_MOMENTUM:
        return min(0.92, base + 0.03)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# 1) Daily Engulfing + Volume Surge
# ══════════════════════════════════════════════════════════════════════════════
class PerpEngulfingVolumeSurge(PerplexityStrategy):
    name = "Daily_Engulfing_Volume"
    # KEEP (5y re-evaluation 2026-06-02): net +$2,756 / 9 trades / WR 66.7%
    # over 5y. Short-side engine path now wired; this strategy's short branch
    # is still 0-trade because momentum_strategies' regime gate uses LIVE
    # state in backtest (separate follow-up), but the long-side edge is
    # consistent enough to keep live.
    # Artifact: reports/perplexity_5y_research_verdicts.md

    config: dict = {
        "min_data_bars":   60,
        "vol_multiple":    1.5,
        "atr_stop_mult":   1.5,
        "tp_atr_mult":     3.0,
        "min_rr":          1.8,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol)
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")
        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")
        close = float(df["Close"].iloc[-1])

        if cp.is_bullish_engulfing(df, -1) and _regime_allows_long(snap):
            entry = close
            stop = max(float(df["Low"].iloc[-1]) - 0.1 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            rr = (target - entry) / (entry - stop)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.65 + min(0.15, (vol_ratio - 1.5) * 0.10)), 2),
                reason=f"Daily bullish engulfing on {vol_ratio:.1f}× vol; R:R {rr:.1f}",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        if cp.is_bearish_engulfing(df, -1) and _regime_allows_short(snap):
            entry = close
            stop = min(float(df["High"].iloc[-1]) + 0.1 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            rr = (entry - target) / (stop - entry)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.65 + min(0.15, (vol_ratio - 1.5) * 0.10)), 2),
                reason=f"Daily bearish engulfing on {vol_ratio:.1f}× vol; R:R {rr:.1f}",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════════════════════
# 2) Daily NR4/NR7 Breakout
# ══════════════════════════════════════════════════════════════════════════════
class PerpNarrowRangeBreakout(PerplexityStrategy):
    name = "Daily_NR_Breakout"
    # KEEP -- BORDERLINE (5y re-evaluation 2026-06-02): net +$1,392 / 13
    # trades / WR 61.5% over 5y. 13 trades is just under the 20-trade
    # comfort threshold; trades pay (PF 1.83) and direction is consistent.
    # Artifact: reports/perplexity_5y_research_verdicts.md

    config: dict = {
        "min_data_bars":   60,
        "use_nr7":         True,
        "vol_multiple":    1.3,
        "atr_stop_mult":   1.5,
        "tp_atr_mult":     3.5,
        "min_rr":          2.0,
        "min_breakout_pct": 0.05,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol)
        is_compressed = (
            cp.is_nr7(df, -2) if cfg["use_nr7"] else (cp.is_nr4(df, -2) or cp.is_inside_bar(df, -2))
        )
        if not is_compressed:
            return self._hold(symbol, "no NR compression on prior day")

        prev = df.iloc[-2]; cur = df.iloc[-1]
        atr = cp.current_atr(df, 14)
        prev_range = float(prev["High"]) - float(prev["Low"])
        if atr <= 0 or prev_range <= 0:
            return self._hold(symbol, "degenerate range")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")

        close = float(cur["Close"])
        long_break = close > float(prev["High"]) + prev_range * (cfg["min_breakout_pct"] / 100)
        short_break = close < float(prev["Low"])  - prev_range * (cfg["min_breakout_pct"] / 100)

        if long_break and _regime_allows_long(snap):
            entry = close
            stop = max(float(prev["Low"]) - 0.1 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            rr = (target - entry) / (entry - stop)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.62), 2),
                reason=f"NR{'7' if cfg['use_nr7'] else '4'} breakout > {prev['High']:.2f} on {vol_ratio:.1f}× vol",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2),
                            "prev_range": round(prev_range, 2)},
            )
        if short_break and _regime_allows_short(snap):
            entry = close
            stop = min(float(prev["High"]) + 0.1 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            rr = (entry - target) / (stop - entry)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.62), 2),
                reason=f"NR{'7' if cfg['use_nr7'] else '4'} breakdown < {prev['Low']:.2f} on {vol_ratio:.1f}× vol",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2),
                            "prev_range": round(prev_range, 2)},
            )
        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════════════════════
# 3) Daily 3-Bar Momentum Push
# ══════════════════════════════════════════════════════════════════════════════
class PerpThreeBarPush(PerplexityStrategy):
    name = "Daily_Three_Bar_Push"
    # KEEP (5y re-evaluation 2026-06-02): net +$6,511 / 26 trades / WR 65.4%
    # over 5y. Most-improved on more data -- the 2y -$567 verdict was
    # sample noise. Strongest of the four daily-candle patterns.
    # Artifact: reports/perplexity_5y_research_verdicts.md

    config: dict = {
        "min_data_bars":  60,
        "vol_multiple":   1.2,
        "atr_stop_mult":  2.0,
        "tp_atr_mult":    4.0,
        "min_rr":         1.8,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol)
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")
        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")
        close = float(df["Close"].iloc[-1])

        if cp.is_three_bar_push_up(df, -1) and _regime_allows_long(snap):
            entry = close
            stop = max(float(df["Low"].iloc[-3:].min()) - 0.1 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.5 * atr:
                stop = entry - 0.5 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            rr = (target - entry) / (entry - stop)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.68), 2),
                reason=f"3-bar daily push up on {vol_ratio:.1f}× vol; R:R {rr:.1f}",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        if cp.is_three_bar_push_down(df, -1) and _regime_allows_short(snap):
            entry = close
            stop = min(float(df["High"].iloc[-3:].max()) + 0.1 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.5 * atr:
                stop = entry + 0.5 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            rr = (entry - target) / (stop - entry)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.68), 2),
                reason=f"3-bar daily push down on {vol_ratio:.1f}× vol; R:R {rr:.1f}",
                indicators={"vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════════════════════
# 4) Daily Hammer / Shooting Star Reversal
# ══════════════════════════════════════════════════════════════════════════════
class PerpHammerShootingStar(PerplexityStrategy):
    name = "Daily_Hammer_Star"
    # KEEP -- BORDERLINE (5y re-evaluation 2026-06-02): net +$896 / 11
    # trades / WR 63.6% over 5y. 11 trades is the thinnest of the kept set;
    # PF 1.57 / hold 3.7d / direction consistent. On the right side of zero.
    # Artifact: reports/perplexity_5y_research_verdicts.md

    config: dict = {
        "min_data_bars":    60,
        "rsi_period":       14,
        "rsi_oversold":     35,
        "rsi_overbought":   65,
        "prior_bars":       3,
        "tail_to_body":     2.0,
        "vol_multiple":     1.1,
        "atr_stop_mult":    1.5,
        "tp_atr_mult":      3.0,
        "min_rr":           1.8,
    }

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(alpha=1 / period, adjust=False).mean()
        al = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = ag / al.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol)
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")
        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")

        rsi_now = float(self._rsi(df["Close"], cfg["rsi_period"]).iloc[-1])
        prior = df.iloc[-(cfg["prior_bars"] + 1):-1]
        red_count = int((prior["Close"] < prior["Open"]).sum())
        green_count = int((prior["Close"] > prior["Open"]).sum())
        close = float(df["Close"].iloc[-1])

        if (
            cp.is_hammer(df, -1, tail_to_body=cfg["tail_to_body"])
            and rsi_now < cfg["rsi_oversold"]
            and red_count >= cfg["prior_bars"] - 1
            and _regime_allows_long(snap)
        ):
            entry = close
            stop = max(float(df["Low"].iloc[-1]) - 0.05 * atr,
                       entry - cfg["atr_stop_mult"] * atr)
            if entry - stop < 0.4 * atr:
                stop = entry - 0.4 * atr
            target = entry + cfg["tp_atr_mult"] * atr
            rr = (target - entry) / (entry - stop)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.70), 2),
                reason=f"Daily hammer after {red_count} red bars; RSI {rsi_now:.1f}",
                indicators={"rsi": round(rsi_now, 1), "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        if (
            cp.is_shooting_star(df, -1, tail_to_body=cfg["tail_to_body"])
            and rsi_now > cfg["rsi_overbought"]
            and green_count >= cfg["prior_bars"] - 1
            and _regime_allows_short(snap)
        ):
            entry = close
            stop = min(float(df["High"].iloc[-1]) + 0.05 * atr,
                       entry + cfg["atr_stop_mult"] * atr)
            if stop - entry < 0.4 * atr:
                stop = entry + 0.4 * atr
            target = entry - cfg["tp_atr_mult"] * atr
            rr = (entry - target) / (stop - entry)
            if rr < cfg["min_rr"]:
                return self._hold(symbol, f"R:R {rr:.1f} below min")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=round(entry, 2), stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(_confidence_adjust(snap, 0.70), 2),
                reason=f"Daily shooting star after {green_count} green bars; RSI {rsi_now:.1f}",
                indicators={"rsi": round(rsi_now, 1), "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr, 2), "r_r": round(rr, 2)},
            )
        return self._hold(symbol)


__all__ = [
    "PerpEngulfingVolumeSurge",
    "PerpNarrowRangeBreakout",
    "PerpThreeBarPush",
    "PerpHammerShootingStar",
]
