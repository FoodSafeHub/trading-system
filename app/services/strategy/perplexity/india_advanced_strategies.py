from __future__ import annotations

"""Five advanced India swing/momentum strategies (daily bars).

Implemented as :class:`PerplexityStrategy` subclasses so they run through the
existing India-aware backtest engine (app.services.backtest.perplexity_engine)
and reuse its data provider, costs (INDIA_DEFAULT), position sizing, trailing
stops, and metrics. They gate on the injected momentum snapshot, which for NSE
symbols is computed from ^NSEI / ^INDIAVIX (see market_regime_advanced) — no US
tape leaks into Indian signals.

Strategies (per the spec)
-------------------------
1. :class:`MomentumBreakout`      — RS-leader breakout near 52wk high on volume.
2. :class:`TrendPullbackEma`      — pullback to the 20–50 EMA zone with a
                                    bullish engulfing / pinbar trigger.
3. :class:`TrendFollowingHHHL`    — golden-cross + higher-high/higher-low
                                    structure, RSI 50–70 (momentum-rank gated).
4. :class:`SupportResistanceBounce` — rebound off a detected support shelf with
                                    volume + RSI>50.
5. :class:`WyckoffSpringTest`     — range spring (false breakdown that closes
                                    back inside) then breakout above range high.

Relative Strength
-----------------
The spec wants RS vs Nifty 500. The per-symbol run() signature can't see the
whole universe, but the engine injects a point-in-time regime snapshot whose
``spy_close`` field is the benchmark index close AS-OF that bar (^NSEI for
India). We compute RS as the symbol's trailing return minus the benchmark's
trailing return over ``rs_lookback`` bars and map it to a 0–100 "RS rating"
proxy. This is PIT-safe (no lookahead) and needs no new engine plumbing. A true
cross-sectional RS≥87 percentile vs Nifty 500 can be layered later via the
rs_rotation service; the proxy is the runnable approximation agreed for now.

Fundamental filters (ROIC>15%, market cap, dividend yield, ₹3cr liquidity) are
configurable pass-through stubs (default off) — no free reliable NSE source.
Wire a fundamentals provider into ``_fundamentals_ok`` to enable them.
"""

from typing import Optional

import pandas as pd

from app.services.market_regime import MarketRegime
from app.services.market_regime_advanced import MomentumRegime, get_momentum_regime
from app.services.markets import is_india_symbol
from app.services.strategy import candle_patterns as cp
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


# ── Regime gating ─────────────────────────────────────────────────────────────

def _momentum_snapshot(symbol: str | None = None, injected=None):
    if injected is not None:
        return injected
    try:
        market = "india" if (symbol and is_india_symbol(symbol)) else "us"
        return get_momentum_regime(market=market)
    except Exception:
        return None


def _regime_allows_long(snap) -> bool:
    if snap is None:
        return True
    return snap.regime in (MomentumRegime.BULL_MOMENTUM, MomentumRegime.BULL_CAUTION)


def _confidence_adjust(snap, base: float) -> float:
    if snap is None:
        return base
    if snap.regime == MomentumRegime.BULL_MOMENTUM:
        return min(0.95, base + 0.05)
    if snap.regime == MomentumRegime.BULL_CAUTION:
        return max(0.40, base - 0.10)
    return base


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _trend_exit_signal(self, symbol: str, df: pd.DataFrame,
                       ema_period: int = 20, rsi_overbought: float = 80.0) -> Optional[PerplexitySignal]:
    """Shared exit: emit a SELL when the trend that justified the long breaks.

    A SELL fires when EITHER the close drops below the exit EMA (trend lost) OR
    RSI(14) is overbought (momentum exhausted). The backtest engine and live
    scheduler convert this SELL into Approach C's tight trailing stop — so the
    position rides the remaining move rather than dumping at market.

    Returns a SELL PerplexitySignal, or None when no exit condition is met.
    Called only while in a position (the engine asks run() for a SELL then).
    """
    close = df["Close"]
    if len(close) < ema_period + 2:
        return None
    c_now = float(close.iloc[-1])
    ema = float(_ema(close, ema_period).iloc[-1])
    rsi_now = float(_rsi(close, 14).iloc[-1])
    below_ema = c_now < ema
    overbought = rsi_now >= rsi_overbought
    if not (below_ema or overbought):
        return None
    reason = (f"close {c_now:.2f} < EMA{ema_period} {ema:.2f} — trend lost"
              if below_ema else f"RSI {rsi_now:.0f} ≥ {rsi_overbought:.0f} — overbought")
    return PerplexitySignal(
        symbol=symbol, strategy_name=self.name, direction="SELL",
        entry_price=round(c_now, 2), confidence=0.70,
        reason=f"Exit: {reason}",
        indicators={"close": round(c_now, 2), f"ema{ema_period}": round(ema, 2),
                    "rsi": round(rsi_now, 1)},
    )


def _rs_rating(symbol_close: pd.Series, lookback: int,
               benchmark_close: Optional[pd.Series] = None) -> Optional[float]:
    """RS rating (0–100): symbol's trailing return vs the index, SAME horizon.

    The spec wants RS vs Nifty 500 over a 40-day lookback. The backtest engine
    injects ``benchmark_close`` — the index (^NSEI) close series sliced as-of the
    current bar — so we measure both legs over the identical ``lookback`` window
    (no horizon mismatch, no lookahead).

    Mapping: a relative-return spread of −20%..+40% maps to 0..100 (clamped); a
    symbol matching the index (0 spread) rates ~33, a symbol +20% ahead rates
    ~67. ``rs_min`` is tuned against the backtest. Returns None when inputs are
    insufficient (caller decides how to treat a missing benchmark).
    """
    if len(symbol_close) <= lookback:
        return None
    sym_ret = float(symbol_close.iloc[-1]) / float(symbol_close.iloc[-lookback - 1]) - 1.0

    bench_ret = None
    if benchmark_close is not None and len(benchmark_close) > lookback:
        b_now = float(benchmark_close.iloc[-1])
        b_past = float(benchmark_close.iloc[-lookback - 1])
        if b_past > 0:
            bench_ret = b_now / b_past - 1.0
    if bench_ret is None:
        return None

    spread = sym_ret - bench_ret
    rating = (spread + 0.20) / 0.60 * 100.0
    return max(0.0, min(100.0, rating))


def _fundamentals_ok(symbol: str, cfg: dict) -> bool:
    """Pass-through stub for fundamental gates (ROIC / market cap / div yield).

    Returns True unless a fundamentals provider is wired in AND a gate is
    enabled in ``cfg``. No free reliable NSE source, so default is pass-through.
    To enable: set cfg["use_fundamentals"]=True and implement the lookups here.
    """
    if not cfg.get("use_fundamentals", False):
        return True
    # Placeholder — integrate a provider (screener/tijori/CSV) then gate on:
    #   roic >= cfg["min_roic"], market_cap in band, div_yield < cfg["max_div_yield"]
    return True


def _turnover_ok(df: pd.DataFrame, cfg: dict) -> bool:
    """₹-turnover liquidity gate: 30-day avg (Close×Volume) ≥ min_turnover_inr."""
    min_to = cfg.get("min_turnover_inr", 0.0)
    if min_to <= 0 or "Volume" not in df.columns or len(df) < 30:
        return True
    turnover = (df["Close"] * df["Volume"]).iloc[-30:].mean()
    return float(turnover) >= min_to


# ══════════════════════════════════════════════════════════════════════════════
# 1) Momentum Breakout — RS leader breaking out near the 52-week high
# ══════════════════════════════════════════════════════════════════════════════
class MomentumBreakout(PerplexityStrategy):
    """RS-leader breakout (Mark-Minervini / IBD style).

    BUY : RS rating ≥ rs_min vs the index
          + price 0–10% below its 52-week high (set-up zone)
          + close > prior ``breakout_bars`` high (the breakout) on volume ≥
            vol_mult × 20-day avg
          + close > SMA200 AND SMA150 AND SMA150 > SMA200 (stage-2 uptrend)
          + regime allows long.
    Stop : below the breakout level (prior swing high that was cleared),
           floored at atr_stop_mult × ATR.
    Trail: handled by the engine's 1R→0.5R trailing logic (≈ the 10/20/30 EMA
           trail intent). Exit also on close < EMA200 (proxy for the "2 closes
           below 200 EMA" rule the live runner can enforce separately).
    """
    name = "India_Momentum_Breakout"
    research_only: bool = True

    config: dict = {
        "min_data_bars":      270,
        "rs_lookback":        40,     # spec: 40-day RS
        "rs_min":             60.0,   # proxy threshold (≈ RS≥87 percentile)
        "high_lookback":      252,    # 52 weeks
        "near_high_max_pct":  10.0,   # within 0–10% of 52wk high
        "breakout_bars":      20,     # break the prior 20-bar high
        "vol_mult":           1.5,
        "sma_fast":           150,
        "sma_slow":           200,
        "atr_stop_mult":      2.0,
        "atr_tp_mult":        6.0,
        "min_rr":             2.0,
        "max_hold_bars":      60,
        "exit_ema":           20,    # SELL signal when close < EMA(20) → Approach C trail
        # liquidity / fundamentals (stubs)
        "min_turnover_inr":   0.0,
        "use_fundamentals":   False,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        # Exit FIRST: when in a position, a trend-break/overbought SELL lets the
        # engine + live scheduler arm Approach C's tight trail. Entry gates below
        # would otherwise mask the exit, so check it before them.
        _exit = _trend_exit_signal(self, symbol, df, ema_period=cfg.get("exit_ema", 20))
        if _exit is not None:
            return _exit

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")
        if not _turnover_ok(df, cfg) or not _fundamentals_ok(symbol, cfg):
            return self._hold(symbol, "liquidity/fundamentals gate")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        # Relative strength vs index (same-horizon, PIT benchmark from engine).
        rs = _rs_rating(close, cfg["rs_lookback"], kwargs.get("benchmark_close"))
        if rs is None or rs < cfg["rs_min"]:
            return self._hold(symbol, f"RS {rs if rs is None else round(rs)} < {cfg['rs_min']}")

        # Stage-2 uptrend: above rising 150/200 SMA stack.
        sma150 = float(_sma(close, cfg["sma_fast"]).iloc[-1])
        sma200 = float(_sma(close, cfg["sma_slow"]).iloc[-1])
        if not (c_now > sma200 and c_now > sma150 and sma150 > sma200):
            return self._hold(symbol, "not in stage-2 (150/200 SMA) uptrend")

        # Within 0–10% of the 52-week high.
        high_52w = float(df["High"].iloc[-cfg["high_lookback"]:].max())
        below_high_pct = (high_52w - c_now) / high_52w * 100 if high_52w > 0 else 99.0
        if not (0.0 <= below_high_pct <= cfg["near_high_max_pct"]):
            return self._hold(symbol, f"{below_high_pct:.1f}% below 52wk high — outside set-up zone")

        # Breakout above the prior N-bar high (excluding today).
        bb = cfg["breakout_bars"]
        breakout_level = float(df["High"].iloc[-(bb + 1):-1].max())
        if c_now <= breakout_level:
            return self._hold(symbol, "no breakout above prior high")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_mult"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_mult']}×")

        entry = c_now
        stop = min(breakout_level * 0.995, entry - cfg["atr_stop_mult"] * atr)
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
            confidence=round(_confidence_adjust(snap, 0.70), 2),
            reason=(f"RS {rs:.0f} leader breakout > {breakout_level:.2f} on {vol_ratio:.1f}× vol; "
                    f"{below_high_pct:.1f}% below 52wk high, R:R {rr:.1f}"),
            indicators={"rs": round(rs, 0), "breakout_level": round(breakout_level, 2),
                        "below_high_pct": round(below_high_pct, 1), "vol_ratio": round(vol_ratio, 2),
                        "sma150": round(sma150, 2), "sma200": round(sma200, 2),
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )

    # Exit helper for the LIVE runner: "close below EMA200 for 2 consecutive days".
    # The backtest engine uses its own stop/target/trailing; this SELL fires the
    # structural exit when the strategy is polled while in a position.
    def _structural_exit(self, df: pd.DataFrame) -> bool:
        close = df["Close"]
        if len(close) < 201:
            return False
        ema200 = _ema(close, 200)
        return bool(close.iloc[-1] < ema200.iloc[-1] and close.iloc[-2] < ema200.iloc[-2])


# ══════════════════════════════════════════════════════════════════════════════
# 2) Trend Pullback (EMA zone) — bullish engulfing / pinbar at 20–50 EMA
# ══════════════════════════════════════════════════════════════════════════════
class TrendPullbackEma(PerplexityStrategy):
    """Pullback into the 20–50 EMA zone of an uptrend with a reversal trigger.

    BUY : close > rising EMA(50)
          + price inside the EMA20..EMA50 band (the pullback zone)
          + pullback volume < impulse volume (recent down-leg quieter than the
            preceding up-leg)
          + bullish engulfing OR hammer/pinbar on the trigger bar
          + regime allows long.
    Stop : below the pullback low (or EMA50), floored at atr_stop_mult × ATR.
    Target: previous swing high (≥ 2R, else use atr_tp_mult × ATR).
    Trail: engine trailing (≈ 20-EMA trail intent).
    """
    name = "India_Trend_Pullback"
    research_only: bool = True

    config: dict = {
        "min_data_bars":     120,
        "ema_fast":          20,
        "ema_slow":          50,
        "ema_slope_bars":    10,
        "rsi_period":        14,
        "atr_stop_mult":     1.5,
        "atr_tp_mult":       3.0,
        "min_rr":            2.0,
        "swing_lookback":    30,
        "impulse_bars":      10,    # window for impulse vs pullback volume
        "pullback_bars":     3,
        "max_hold_bars":     20,
        "exit_ema":          50,    # SELL when close < EMA(50) → Approach C trail
        "min_turnover_inr":  0.0,
        "use_fundamentals":  False,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        # Exit FIRST (see MomentumBreakout): in a position, a trend-break/
        # overbought SELL lets the engine + scheduler arm Approach C's trail.
        _exit = _trend_exit_signal(self, symbol, df, ema_period=cfg.get("exit_ema", 20))
        if _exit is not None:
            return _exit

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")
        if not _turnover_ok(df, cfg):
            return self._hold(symbol, "liquidity gate")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        ema20 = _ema(close, cfg["ema_fast"])
        ema50 = _ema(close, cfg["ema_slow"])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e50_prev = float(ema50.iloc[-cfg["ema_slope_bars"]])

        if not (c_now > e50 and e50 > e50_prev):
            return self._hold(symbol, "not above a rising EMA50")

        # Pullback zone: price between EMA20 and EMA50 (inclusive band).
        lo_band, hi_band = min(e20, e50), max(e20, e50)
        # Allow a small tolerance around the band so wicks into it qualify.
        tol = 0.01 * c_now
        if not (lo_band - tol <= c_now <= hi_band + tol):
            return self._hold(symbol, "price not in EMA20–50 pullback zone")

        # Pullback volume < impulse volume.
        if "Volume" in df.columns and len(df) > cfg["impulse_bars"] + cfg["pullback_bars"]:
            pb = cfg["pullback_bars"]; imp = cfg["impulse_bars"]
            pullback_vol = float(df["Volume"].iloc[-pb:].mean())
            impulse_vol = float(df["Volume"].iloc[-(imp + pb):-pb].mean())
            if impulse_vol > 0 and pullback_vol >= impulse_vol:
                return self._hold(symbol, "pullback volume not below impulse volume")

        # Reversal trigger at the zone.
        trigger = cp.is_bullish_engulfing(df, -1) or cp.is_hammer(df, -1, tail_to_body=1.8)
        if not trigger:
            return self._hold(symbol, "no engulfing/pinbar trigger")

        rsi_now = float(_rsi(close, cfg["rsi_period"]).iloc[-1])

        entry = c_now
        pullback_low = float(df["Low"].iloc[-cfg["pullback_bars"]:].min())
        stop = min(pullback_low - 0.1 * atr, e50 - 0.1 * atr, entry - cfg["atr_stop_mult"] * atr)
        if entry - stop <= 0:
            return self._hold(symbol, "degenerate stop")

        # Target = previous swing high, with a 2R floor.
        swing_high = cp.structure_swing_high(df, lookback=cfg["swing_lookback"])
        risk = entry - stop
        target_2r = entry + cfg["min_rr"] * risk
        if swing_high and swing_high > target_2r:
            target = swing_high
        else:
            target = max(target_2r, entry + cfg["atr_tp_mult"] * atr)
        rr = (target - entry) / risk
        if rr < cfg["min_rr"]:
            return self._hold(symbol, f"R:R {rr:.1f} below min")

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.64), 2),
            reason=(f"Pullback to EMA20–50 with reversal trigger; RSI {rsi_now:.0f}, R:R {rr:.1f}"),
            indicators={"ema20": round(e20, 2), "ema50": round(e50, 2), "rsi": round(rsi_now, 1),
                        "swing_high": round(swing_high, 2) if swing_high else None,
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3) Trend Following — golden cross + higher-high/higher-low structure
# ══════════════════════════════════════════════════════════════════════════════
class TrendFollowingHHHL(PerplexityStrategy):
    """Buy a confirmed higher-low after a 50/200 golden cross, RSI 50–70.

    Approximates "Nifty 200 Momentum 30 membership" with a momentum-rank gate:
    RS rating must be strong (rs_min) — the same proxy used by MomentumBreakout.

    BUY : SMA50 > SMA200 AND a golden cross occurred within ``cross_lookback`` bars
          + RSI(14) in [50, 70]
          + structure: latest swing low > prior swing low (higher low) AND
            latest swing high > prior swing high (higher high)
          + price reclaiming after a higher-low (close > prior bar)
          + regime allows long.
    Exit : engine stop/target/trailing; live runner can add "close < SMA50 for
           2 days" via ``_structural_exit``.
    """
    name = "India_Trend_Following"
    research_only: bool = True

    config: dict = {
        "min_data_bars":   270,
        "sma_fast":        50,
        "sma_slow":        200,
        "cross_lookback":  60,     # golden cross within last N bars
        "rsi_period":      14,
        "rsi_min":         50,
        "rsi_max":         70,
        "rs_lookback":     40,
        "rs_min":          50.0,   # momentum-rank proxy for "Momentum 30"
        "swing_lookback":  10,
        "atr_stop_mult":   2.0,
        "atr_tp_mult":     5.0,
        "min_rr":          2.0,
        "max_hold_bars":   60,
        "exit_ema":        50,    # SELL when close < EMA(50) → Approach C trail
        "min_turnover_inr": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        # Exit FIRST (see MomentumBreakout): in a position, a trend-break/
        # overbought SELL lets the engine + scheduler arm Approach C's trail.
        _exit = _trend_exit_signal(self, symbol, df, ema_period=cfg.get("exit_ema", 20))
        if _exit is not None:
            return _exit

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")
        if not _turnover_ok(df, cfg):
            return self._hold(symbol, "liquidity gate")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        c_prev = float(close.iloc[-2])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        rs = _rs_rating(close, cfg["rs_lookback"], kwargs.get("benchmark_close"))
        if rs is None or rs < cfg["rs_min"]:
            return self._hold(symbol, "RS rank below momentum threshold")

        sma50 = _sma(close, cfg["sma_fast"])
        sma200 = _sma(close, cfg["sma_slow"])
        s50_now, s200_now = float(sma50.iloc[-1]), float(sma200.iloc[-1])
        if not (s50_now > s200_now):
            return self._hold(symbol, "50DMA not above 200DMA")

        # Golden cross within lookback (was below, now above).
        lb = min(cfg["cross_lookback"], len(sma50) - 1)
        crossed = any(
            float(sma50.iloc[-(k + 1)]) <= float(sma200.iloc[-(k + 1)])
            for k in range(1, lb)
        )
        if not crossed:
            return self._hold(symbol, "no recent golden cross")

        rsi_now = float(_rsi(close, cfg["rsi_period"]).iloc[-1])
        if not (cfg["rsi_min"] <= rsi_now <= cfg["rsi_max"]):
            return self._hold(symbol, f"RSI {rsi_now:.0f} outside [{cfg['rsi_min']},{cfg['rsi_max']}]")

        # Higher-high / higher-low structure across two recent swing windows.
        n = cfg["swing_lookback"]
        recent_low = cp.structure_swing_low(df, lookback=n)
        prior_low = cp.structure_swing_low(df.iloc[:-n], lookback=n) if len(df) > 2 * n else None
        recent_high = cp.structure_swing_high(df, lookback=n)
        prior_high = cp.structure_swing_high(df.iloc[:-n], lookback=n) if len(df) > 2 * n else None
        if not (recent_low and prior_low and recent_high and prior_high):
            return self._hold(symbol, "insufficient swing structure")
        if not (recent_low > prior_low and recent_high > prior_high):
            return self._hold(symbol, "no higher-high/higher-low structure")

        # Higher-low confirmation: price turning back up.
        if c_now <= c_prev:
            return self._hold(symbol, "no higher-low reclaim (close not up)")

        entry = c_now
        stop = min(recent_low - 0.1 * atr, entry - cfg["atr_stop_mult"] * atr)
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
            confidence=round(_confidence_adjust(snap, 0.66), 2),
            reason=(f"Golden-cross trend, HH/HL structure, RSI {rsi_now:.0f}, RS {rs:.0f}; R:R {rr:.1f}"),
            indicators={"rs": round(rs, 0), "rsi": round(rsi_now, 1),
                        "sma50": round(s50_now, 2), "sma200": round(s200_now, 2),
                        "recent_low": round(recent_low, 2), "prior_low": round(prior_low, 2),
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )

    def _structural_exit(self, df: pd.DataFrame) -> bool:
        close = df["Close"]
        if len(close) < 51:
            return False
        sma50 = _sma(close, 50)
        return bool(close.iloc[-1] < sma50.iloc[-1] and close.iloc[-2] < sma50.iloc[-2])


# ══════════════════════════════════════════════════════════════════════════════
# 4) Support/Resistance Bounce
# ══════════════════════════════════════════════════════════════════════════════
class SupportResistanceBounce(PerplexityStrategy):
    """Buy a rebound off a tested support shelf with volume + RSI>50.

    Support is the lowest swing low of the lookback window that has been touched
    ``min_touches`` times (within ``touch_tol_pct``). We require price to have
    dipped to that shelf on the prior bar and closed back above it today with a
    volume uptick and RSI>50.

    Stop : below support (3–5% / 1.5×ATR, whichever is wider but capped).
    Target: nearest resistance (swing high) above, ≥ min_rr else ATR target.
    """
    name = "India_SR_Bounce"
    research_only: bool = True

    config: dict = {
        "min_data_bars":    120,
        "sr_lookback":      60,
        "min_touches":      2,
        "touch_tol_pct":    1.5,
        "rsi_period":       14,
        "rsi_min":          50,
        "vol_mult":         1.1,
        "stop_below_pct":   4.0,   # 3–5% below support
        "atr_stop_mult":    1.5,
        "atr_tp_mult":      3.0,
        "min_rr":           2.0,
        "max_hold_bars":    20,
        "exit_ema":         20,    # SELL when close < EMA(20) → Approach C trail
        "min_turnover_inr": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        # Exit FIRST (see MomentumBreakout): in a position, a trend-break/
        # overbought SELL lets the engine + scheduler arm Approach C's trail.
        _exit = _trend_exit_signal(self, symbol, df, ema_period=cfg.get("exit_ema", 20))
        if _exit is not None:
            return _exit

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")
        if not _turnover_ok(df, cfg):
            return self._hold(symbol, "liquidity gate")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        c_prev = float(close.iloc[-2])
        low_prev = float(df["Low"].iloc[-2])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        rsi_now = float(_rsi(close, cfg["rsi_period"]).iloc[-1])
        if rsi_now <= cfg["rsi_min"]:
            return self._hold(symbol, f"RSI {rsi_now:.0f} not bullish (>50)")

        # Support shelf = lowest low of the window; count touches near it.
        lb = cfg["sr_lookback"]
        lows = df["Low"].iloc[-lb:]
        support = float(lows.min())
        tol = support * cfg["touch_tol_pct"] / 100
        touches = int((lows <= support + tol).sum())
        if touches < cfg["min_touches"]:
            return self._hold(symbol, f"support touched only {touches}× (<{cfg['min_touches']})")

        # Prior bar dipped to support, today closes back above it (rebound).
        dipped = low_prev <= support + tol
        reclaimed = c_now > support + tol and c_now > c_prev
        if not (dipped and reclaimed):
            return self._hold(symbol, "no rebound off support")

        vol_ratio = cp.volume_surge_ratio(df, -1, 20)
        if vol_ratio < cfg["vol_mult"]:
            return self._hold(symbol, f"vol {vol_ratio:.1f}× below {cfg['vol_mult']}×")

        entry = c_now
        stop_pct = entry * (1 - cfg["stop_below_pct"] / 100)
        stop = min(support * (1 - 0.005), stop_pct, entry - cfg["atr_stop_mult"] * atr)
        if entry - stop <= 0:
            return self._hold(symbol, "degenerate stop")

        # Resistance = nearest swing high above entry.
        resistance = cp.structure_swing_high(df, lookback=lb)
        risk = entry - stop
        target_2r = entry + cfg["min_rr"] * risk
        if resistance and resistance > target_2r:
            target = resistance
        else:
            target = max(target_2r, entry + cfg["atr_tp_mult"] * atr)
        rr = (target - entry) / risk
        if rr < cfg["min_rr"]:
            return self._hold(symbol, f"R:R {rr:.1f} below min")

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.60), 2),
            reason=(f"Bounce off support {support:.2f} ({touches} touches) on {vol_ratio:.1f}× vol; "
                    f"RSI {rsi_now:.0f}, R:R {rr:.1f}"),
            indicators={"support": round(support, 2), "touches": touches,
                        "resistance": round(resistance, 2) if resistance else None,
                        "rsi": round(rsi_now, 1), "vol_ratio": round(vol_ratio, 2),
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5) Wyckoff Spring / Test (VSA)
# ══════════════════════════════════════════════════════════════════════════════
class WyckoffSpringTest(PerplexityStrategy):
    """Wyckoff spring: a false breakdown below a trading range that closes back
    inside on high volume, followed by a low-volume test, then a breakout above
    the range high.

    BUY : a sideways range of ≥ range_bars exists (range width ≤ max_range_pct)
          + within ``spring_window`` bars a bar broke below range low intraday
            but closed back inside, on volume > vol_mult × 10-day avg (the spring)
          + a later bar re-tested lower on LOWER volume (the test)
          + today closes above the range high (the breakout/SOS)
          + regime allows long.
    Stop : below the spring low (or 1 ATR), whichever is tighter-but-valid.
    Targets: range mid → range high projection (we set target at range high +
             one range width; engine also trails).
    """
    name = "India_Wyckoff_Spring"
    research_only: bool = True

    config: dict = {
        "min_data_bars":   120,
        "range_bars":      20,      # ≥20 sessions sideways
        "max_range_pct":   15.0,    # range height ≤ 15% of mid
        "spring_window":   10,      # spring must be within last N bars
        "vol_mult":        1.2,     # spring vol > 1.2× 10-day avg
        "vol_avg_bars":    10,
        "atr_stop_mult":   1.0,
        "min_rr":          1.5,
        "max_hold_bars":   25,
        "exit_ema":        20,    # SELL when close < EMA(20) → Approach C trail
        "min_turnover_inr": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None, **kwargs) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        # Exit FIRST (see MomentumBreakout): in a position, a trend-break/
        # overbought SELL lets the engine + scheduler arm Approach C's trail.
        _exit = _trend_exit_signal(self, symbol, df, ema_period=cfg.get("exit_ema", 20))
        if _exit is not None:
            return _exit

        snap = _momentum_snapshot(symbol, injected=kwargs.get("momentum_snapshot"))
        if not _regime_allows_long(snap):
            return self._hold(symbol, "regime not long-friendly")
        if not _turnover_ok(df, cfg):
            return self._hold(symbol, "liquidity gate")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr = cp.current_atr(df, 14)
        if atr <= 0:
            return self._hold(symbol, "ATR=0")

        # Define the trading range over the window BEFORE the spring window, so
        # the range isn't redefined by the spring/breakout bars themselves.
        rb = cfg["range_bars"]
        sw = cfg["spring_window"]
        if len(df) < rb + sw + 2:
            return self._hold(symbol, "not enough bars for range+spring")

        range_slice = df.iloc[-(rb + sw):-sw]
        range_high = float(range_slice["High"].max())
        range_low = float(range_slice["Low"].min())
        range_mid = (range_high + range_low) / 2
        if range_mid <= 0:
            return self._hold(symbol, "degenerate range")
        range_pct = (range_high - range_low) / range_mid * 100
        if range_pct > cfg["max_range_pct"]:
            return self._hold(symbol, f"range {range_pct:.0f}% too wide — not consolidating")

        # Find a spring within the spring window: low < range_low but close back
        # inside, on elevated volume.
        vol_avg = float(df["Volume"].iloc[-(cfg["vol_avg_bars"] + sw):-sw].mean()) if "Volume" in df.columns else 0.0
        spring_idx = None
        spring_low = None
        for k in range(sw, 0, -1):
            bar = df.iloc[-k]
            if float(bar["Low"]) < range_low and float(bar["Close"]) > range_low:
                vol_ok = True
                if vol_avg > 0:
                    vol_ok = float(bar["Volume"]) > cfg["vol_mult"] * vol_avg
                if vol_ok:
                    spring_idx = k
                    spring_low = float(bar["Low"])
                    break
        if spring_idx is None:
            return self._hold(symbol, "no spring (false breakdown) in window")

        # Test: a later bar dips again on LOWER volume than the spring bar.
        spring_vol = float(df.iloc[-spring_idx]["Volume"]) if "Volume" in df.columns else 0.0
        test_ok = False
        for k in range(spring_idx - 1, 0, -1):
            bar = df.iloc[-k]
            if float(bar["Low"]) <= range_low * 1.01:  # re-tests near the lows
                if spring_vol <= 0 or float(bar["Volume"]) < spring_vol:
                    test_ok = True
                    break
        if not test_ok:
            return self._hold(symbol, "no low-volume test after spring")

        # Breakout: today closes above the range high (sign of strength).
        if c_now <= range_high:
            return self._hold(symbol, "no breakout above range high yet")

        entry = c_now
        stop = max(spring_low - 0.1 * atr, entry - cfg["atr_stop_mult"] * atr) \
            if spring_low else entry - cfg["atr_stop_mult"] * atr
        # Spring low should be BELOW entry; if the ATR floor put the stop above
        # the spring low, prefer the spring low for a structural stop.
        stop = min(stop, entry - 0.1 * atr)
        if entry - stop <= 0:
            return self._hold(symbol, "degenerate stop")

        # Target: project one range width above the breakout (range high → +width).
        width = range_high - range_low
        target = range_high + width
        risk = entry - stop
        if (target - entry) / risk < cfg["min_rr"]:
            target = entry + cfg["min_rr"] * risk
        rr = (target - entry) / risk

        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=round(_confidence_adjust(snap, 0.62), 2),
            reason=(f"Wyckoff spring+test then breakout > range high {range_high:.2f}; R:R {rr:.1f}"),
            indicators={"range_high": round(range_high, 2), "range_low": round(range_low, 2),
                        "range_mid": round(range_mid, 2), "spring_low": round(spring_low, 2) if spring_low else None,
                        "atr": round(atr, 2), "r_r": round(rr, 2)},
        )


__all__ = [
    "MomentumBreakout",
    "TrendPullbackEma",
    "TrendFollowingHHHL",
    "SupportResistanceBounce",
    "WyckoffSpringTest",
]
