from __future__ import annotations

"""India-tuned daily swing strategies.

Replaces the US-locked Perplexity swing set. The old mean-reversion strategies
gated on ``MarketRegime.BULL`` which the backtest engine derives from SPY — so
Indian names were silently gated on US tape (see project_india_regime_gate).

These three strategies instead gate via the *momentum snapshot* the engine
injects per-bar (``kwargs["momentum_snapshot"]``). For Indian symbols that
snapshot is computed from ^NSEI / ^INDIAVIX (see market_regime_advanced
_MARKET_CFG), so the gate reads Indian tape. The live runner falls back to a
fresh per-market snapshot via ``_momentum_snapshot`` when none is injected.

All three are LONG-ONLY (India cash-segment reality; shorting is intraday/F&O).
They share the regime helpers and ATR math with momentum_strategies.py so the
two engines stay consistent.

Strategies
----------
* :class:`NiftyLeaderPullback`      — trend pullback to the 20-EMA in a rising
                                       50/200-EMA stack. NSE leaders trend hard.
* :class:`FiftyTwoWeekHighBreakout` — Donchian 252-day-high breakout + volume.
* :class:`VcpContractionBreakout`   — volatility contraction near highs then a
                                       volume breakout (Minervini VCP-lite).

Shipped ``research_only=True`` — the backtest engine evaluates them but the live
runner skips them until the Nifty-100+midcap backtest justifies flipping each
``research_only`` flag to False.
"""

import pandas as pd

from app.services.market_regime import MarketRegime
from app.services.market_regime_advanced import MomentumRegime, get_momentum_regime
from app.services.markets import is_india_symbol
from app.services.strategy import candle_patterns as cp
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


# ── Regime gating (mirrors momentum_strategies.py so the two engines agree) ───

def _momentum_snapshot(symbol: str | None = None, injected=None):
    """Regime snapshot for the symbol's home market.

    India → Nifty 50 / India VIX; everything else → SPY / ^VIX. During backtests
    the engine injects a point-in-time snapshot; we use it verbatim (never call
    the live helper inside a backtest loop or every bar leaks today's tape).
    """
    if injected is not None:
        return injected
    try:
        market = "india" if (symbol and is_india_symbol(symbol)) else "us"
        return get_momentum_regime(market=market)
    except Exception:
        return None


def _regime_allows_long(snap) -> bool:
    if snap is None:
        return True  # degrade gracefully if regime fetch failed
    return snap.regime in (MomentumRegime.BULL_MOMENTUM, MomentumRegime.BULL_CAUTION)


def _confidence_adjust(snap, base: float) -> float:
    if snap is None:
        return base
    if snap.regime == MomentumRegime.BULL_MOMENTUM:
        return min(0.95, base + 0.05)
    if snap.regime == MomentumRegime.BULL_CAUTION:
        return max(0.40, base - 0.10)
    return base


# ── Shared indicator helpers ──────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


# ══════════════════════════════════════════════════════════════════════════════
# 1) Nifty Leader Pullback — trend pullback to the 20-EMA
# ══════════════════════════════════════════════════════════════════════════════
class NiftyLeaderPullback(PerplexityStrategy):
    """Buy a shallow pullback to the 20-EMA inside an established uptrend.

    BUY : EMA(50) > EMA(200) AND EMA(50) rising (uptrend stack)
          + close > EMA(200) (don't catch broken trends)
          + price pulled back to within ``ema_proximity_pct`` of the 20-EMA
          + RSI(14) in [rsi_min, rsi_max] AND turning up (today > prior)
          + regime allows long.
    Stop  : entry − atr_stop_mult × ATR (and below the pullback low).
    Target: entry + atr_tp_mult × ATR.
    """
    name = "India_Leader_Pullback"
    research_only: bool = True  # flip after Nifty-100+midcap backtest justifies it

    config: dict = {
        "min_data_bars":     220,
        "ema_fast":          20,
        "ema_mid":           50,
        "ema_slow":          200,
        "ema_slope_bars":    10,
        "ema_proximity_pct": 3.0,    # close within ±3% of EMA(20)
        "rsi_period":        14,
        "rsi_min":           40,
        "rsi_max":           60,
        "atr_stop_mult":     2.0,
        "atr_tp_mult":       4.0,
        "min_rr":            1.5,
        "vol_min":           0.7,    # don't require a surge — pullbacks are quiet
        "max_hold_bars":     20,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        ema20 = _ema(close, cfg["ema_fast"])
        ema50 = _ema(close, cfg["ema_mid"])
        ema200 = _ema(close, cfg["ema_slow"])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e50_prev = float(ema50.iloc[-cfg["ema_slope_bars"]])
        e200 = float(ema200.iloc[-1])

        # Uptrend stack: 50 > 200, 50 rising, price above 200.
        if not (e50 > e200 and e50 > e50_prev and c_now > e200):
            return self._hold(symbol, "no rising 50>200 uptrend")

        # Pullback proximity to the 20-EMA.
        prox = abs(c_now - e20) / e20 * 100 if e20 > 0 else 99.0
        if prox > cfg["ema_proximity_pct"]:
            return self._hold(symbol, f"{prox:.1f}% from EMA20 — not a pullback")

        rsi = _rsi(close, cfg["rsi_period"])
        rsi_now = float(rsi.iloc[-1])
        rsi_prev = float(rsi.iloc[-2])
        if not (cfg["rsi_min"] <= rsi_now <= cfg["rsi_max"]):
            return self._hold(symbol, f"RSI {rsi_now:.0f} outside [{cfg['rsi_min']},{cfg['rsi_max']}]")
        if rsi_now <= rsi_prev:
            return self._hold(symbol, "RSI not turning up")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_min"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_min']}×")

        entry = c_now
        pullback_low = float(df["Low"].iloc[-3:].min())
        stop = min(entry - cfg["atr_stop_mult"] * atr, pullback_low - 0.1 * atr)
        if entry - stop < 0.5 * atr:
            stop = entry - 0.5 * atr
        target = entry + cfg["atr_tp_mult"] * atr
        rr = (target - entry) / (entry - stop) if entry > stop else 0.0
        if rr < cfg["min_rr"]:
            return self._hold(symbol, f"R:R {rr:.1f} below min")

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.66), 2),
            reason=(f"Pullback to EMA20 in rising 50>200 stack; "
                    f"RSI {rsi_prev:.0f}→{rsi_now:.0f}, R:R {rr:.1f}"),
            indicators={"ema20": round(e20, 2), "ema50": round(e50, 2),
                        "ema200": round(e200, 2), "rsi": round(rsi_now, 1),
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2) 52-Week-High Momentum Breakout (Donchian)
# ══════════════════════════════════════════════════════════════════════════════
class FiftyTwoWeekHighBreakout(PerplexityStrategy):
    """Buy a breakout to a new 52-week (252-bar) high with volume confirmation.

    BUY : close > highest high of the prior ``donchian_bars`` bars (excluding
          today) by at least ``breakout_buffer_pct``
          + close > EMA(200) (only break out in a structural uptrend)
          + volume ≥ vol_multiple × 20-bar avg
          + regime allows long.
    Stop  : entry − atr_stop_mult × ATR.
    Target: entry + atr_tp_mult × ATR (momentum runners — wide target).
    """
    name = "India_52wk_Breakout"
    research_only: bool = True

    config: dict = {
        "min_data_bars":       270,
        "donchian_bars":       252,   # ~52 weeks
        "breakout_buffer_pct": 0.1,   # close must clear the prior high by 0.1%
        "ema_slow":            200,
        "rsi_period":          14,
        "rsi_min":             55,    # momentum, not just a marginal poke
        "vol_multiple":        1.3,
        "atr_stop_mult":       2.5,
        "atr_tp_mult":         6.0,
        "min_rr":              2.0,
        "max_hold_bars":       40,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        # Prior-N-bar high, EXCLUDING today (no same-bar lookahead).
        n = cfg["donchian_bars"]
        prior_high = float(df["High"].iloc[-(n + 1):-1].max())
        buffer = prior_high * (cfg["breakout_buffer_pct"] / 100)
        if c_now <= prior_high + buffer:
            return self._hold(symbol, f"{c_now:.2f} not above {n}-bar high {prior_high:.2f}")

        e200 = float(_ema(close, cfg["ema_slow"]).iloc[-1])
        if c_now <= e200:
            return self._hold(symbol, "below EMA200 — not a structural uptrend")

        rsi_now = float(_rsi(close, cfg["rsi_period"]).iloc[-1])
        if rsi_now < cfg["rsi_min"]:
            return self._hold(symbol, f"RSI {rsi_now:.0f} < {cfg['rsi_min']}")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")

        entry = c_now
        stop = entry - cfg["atr_stop_mult"] * atr
        target = entry + cfg["atr_tp_mult"] * atr
        rr = (target - entry) / (entry - stop) if entry > stop else 0.0
        if rr < cfg["min_rr"]:
            return self._hold(symbol, f"R:R {rr:.1f} below min")

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.68), 2),
            reason=(f"52wk-high breakout > {prior_high:.2f} on {vol_ratio:.1f}× vol; "
                    f"RSI {rsi_now:.0f}, R:R {rr:.1f}"),
            indicators={"prior_high": round(prior_high, 2), "rsi": round(rsi_now, 1),
                        "vol_ratio": round(vol_ratio, 2), "atr": round(atr, 2),
                        "r_r": round(rr, 2)},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3) VCP-lite Volatility-Contraction Breakout
# ══════════════════════════════════════════════════════════════════════════════
class VcpContractionBreakout(PerplexityStrategy):
    """Minervini VCP-lite: a tightening range near the highs that resolves up.

    Detects volatility contraction (recent ATR materially below its longer
    baseline) while price holds near a recent high, then buys the breakout above
    the contraction range on volume.

    BUY : ATR(recent) ≤ contraction_ratio × ATR(baseline)  (range tightened)
          + close within ``near_high_pct`` of the contraction-window high
          + close > EMA(50) (uptrend)
          + close > high of the prior ``range_bars`` bars (breakout)
          + volume ≥ vol_multiple × 20-bar avg
          + regime allows long.
    Stop  : low of the contraction range (tight — that's the VCP edge).
    Target: entry + atr_tp_mult × ATR.
    """
    name = "India_VCP_Breakout"
    research_only: bool = True

    config: dict = {
        "min_data_bars":     220,
        "ema_trend":         50,
        "range_bars":        10,    # contraction window
        "baseline_bars":     50,    # longer ATR baseline
        "contraction_ratio": 0.75,  # recent ATR ≤ 75% of baseline ATR
        "near_high_pct":     5.0,   # price within 5% of the window high
        "breakout_buffer_pct": 0.1,
        "vol_multiple":      1.3,
        "atr_tp_mult":       4.0,
        "min_rr":            1.8,
        "max_hold_bars":     25,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        # Volatility contraction: recent ATR vs longer baseline.
        atr_recent = float(cp.atr_series(df, cfg["range_bars"]).iloc[-1])
        atr_base = float(cp.atr_series(df, cfg["baseline_bars"]).iloc[-1])
        if atr_base <= 0 or atr_recent > cfg["contraction_ratio"] * atr_base:
            return self._hold(symbol, "no volatility contraction")

        # Trend filter.
        e50 = float(_ema(close, cfg["ema_trend"]).iloc[-1])
        if c_now <= e50:
            return self._hold(symbol, "below EMA50")

        # Near the contraction-window high.
        rb = cfg["range_bars"]
        window_high = float(df["High"].iloc[-(rb + 1):-1].max())
        window_low = float(df["Low"].iloc[-(rb + 1):-1].min())
        near = (window_high - c_now) / window_high * 100 if window_high > 0 else 99.0
        if near > cfg["near_high_pct"]:
            return self._hold(symbol, f"{near:.1f}% below window high — not coiled near highs")

        # Breakout above the contraction range.
        buffer = window_high * (cfg["breakout_buffer_pct"] / 100)
        if c_now <= window_high + buffer:
            return self._hold(symbol, "no breakout above contraction range")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_multiple"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_multiple']}×")

        entry = c_now
        stop = min(window_low, entry - 0.5 * atr)  # tight VCP stop
        if entry - stop <= 0:
            return self._hold(symbol, "degenerate stop")
        target = entry + cfg["atr_tp_mult"] * atr
        rr = (target - entry) / (entry - stop)
        if rr < cfg["min_rr"]:
            return self._hold(symbol, f"R:R {rr:.1f} below min")

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.64), 2),
            reason=(f"VCP breakout: ATR{cfg['range_bars']} {atr_recent:.2f} ≤ "
                    f"{cfg['contraction_ratio']:.0%}×ATR{cfg['baseline_bars']}; "
                    f"break {window_high:.2f} on {vol_ratio:.1f}× vol, R:R {rr:.1f}"),
            indicators={"atr_recent": round(atr_recent, 2), "atr_base": round(atr_base, 2),
                        "window_high": round(window_high, 2), "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )


__all__ = [
    "NiftyLeaderPullback",
    "FiftyTwoWeekHighBreakout",
    "VcpContractionBreakout",
]
