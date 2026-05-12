from __future__ import annotations

"""
5 Perplexity swing trading strategies.

Strategy 1 — 20 EMA Mean Reversion in Uptrend
Strategy 2 — MA Crossover with RSI Confirmation
Strategy 3 — Breakout from Consolidation with Volume
Strategy 4 — Bollinger Band Mean Reversion in Uptrend
Strategy 5 — Support / Fibonacci Pullback in Trend

All operate on daily OHLCV bars.
All are long-only with configurable parameters exposed via the Config tab.

BULL MARKET PHILOSOPHY
─────────────────────
• Only trade when SMA50 > SMA200 (golden cross) — the structural uptrend is intact.
• Close > SMA200 required for all entries — no trading below the long-term trend line.
• ADX > 20 required for trend-following entries — avoids flat, choppy markets.
• Reversion strategies (EMA pullback, BB, Fib) use a lighter ADX gate because they
  BUY INTO weakness — but the golden cross must still be intact (not a bear market).
• BEAR MARKET EXIT: if SMA50 crosses below SMA200 while in a position, all strategies
  exit immediately — capital preservation over holding through a trend reversal.
"""

import pandas as pd

from app.services.indicators.atr import compute_atr
from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma
from app.services.indicators.ema import compute_ema
from app.services.market_regime import MarketRegime
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy

_INDEX_ETFS = {"SPY", "QQQ", "DIA", "IWM", "IVV", "VOO"}


# ── Shared helpers ────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _current_atr(df: pd.DataFrame, period: int = 14) -> float:
    s = _atr_series(df, period)
    v = float(s.iloc[-1])
    return v if not pd.isna(v) else float(df["Close"].iloc[-1]) * 0.02


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX > 20 = trending. ADX < 20 = choppy/sideways."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr_s    = _atr_series(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx_val  = float(dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1])
    return adx_val if not pd.isna(adx_val) else 0.0


def _bull_market_check(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Core bear-market filter shared by ALL strategies.
    Returns (is_bull, reason).
    Conditions (from strictest to loosest requirement):
      1. close > SMA(200) — not in long-term downtrend
      2. SMA(50) > SMA(200) — golden cross: structural bull market intact
    This is the minimum gate. Strategies add their own on top.
    """
    close = df["Close"]
    if len(df) < 200:
        return False, "not enough data for SMA(200)"
    c_now  = float(close.iloc[-1])
    sma50  = float(_sma(close, 50).iloc[-1])
    sma200 = float(_sma(close, 200).iloc[-1])
    if c_now <= sma200:
        return False, f"price below SMA(200) — bear market"
    if sma50 <= sma200:
        return False, f"SMA(50) ≤ SMA(200) — death cross / bear market structure"
    return True, "bull market"


def _is_death_cross(df: pd.DataFrame) -> bool:
    """
    Returns True if SMA50 just crossed below SMA200 (or is already below it).
    Used as an emergency exit trigger for all open positions.
    """
    close  = df["Close"]
    sma50  = _sma(close, 50)
    sma200 = _sma(close, 200)
    if len(sma50) < 2 or len(sma200) < 2:
        return False
    return float(sma50.iloc[-1]) < float(sma200.iloc[-1])


def _market_regime_trend(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Strict gate for trend-following entries (Breakout).
    Requires: bull market + close > SMA50 + ADX > 20.
    Close must be ABOVE SMA50 — we are trading with confirmed momentum, not buying dips.
    """
    is_bull, reason = _bull_market_check(df)
    if not is_bull:
        return False, reason
    close = df["Close"]
    c_now  = float(close.iloc[-1])
    sma50  = float(_sma(close, 50).iloc[-1])
    if c_now <= sma50:
        return False, f"price below SMA(50) — wait for price to reclaim medium-term trend"
    adx_val = _adx(df, 14)
    if adx_val < 20:
        return False, f"ADX={adx_val:.1f} — sideways/choppy market, no breakout edge"
    return True, f"bull trend confirmed (ADX={adx_val:.1f})"


def _market_regime_reversion(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Gate for pullback/reversion entries (EMA, BB, Fib, MA Crossover).
    Requires: bull market (golden cross + above SMA200).
    Does NOT require close > SMA50 or ADX > 20 — these strategies intentionally
    buy into temporary weakness while the overall bull structure is intact.
    A light ADX floor of 15 ensures we aren't in a dead-flat market.
    """
    is_bull, reason = _bull_market_check(df)
    if not is_bull:
        return False, reason
    adx_val = _adx(df, 14)
    if adx_val < 15:
        return False, f"ADX={adx_val:.1f} — market too flat, no directional edge"
    return True, f"bull market, pullback entry allowed (ADX={adx_val:.1f})"


def _swing_high_low(highs: pd.Series, lows: pd.Series, lookback: int = 30):
    """Return (swing_low_price, swing_high_price) from last `lookback` bars."""
    window_h = highs.iloc[-lookback:]
    window_l = lows.iloc[-lookback:]
    return float(window_l.min()), float(window_h.max())


# ══════════════════════════════════════════════════════════════
# STRATEGY 1 — 20 EMA Mean Reversion in Uptrend
# ══════════════════════════════════════════════════════════════
class EmaMeanReversionUptrend(PerplexityStrategy):
    """
    Buy pullbacks to the 20 EMA in an ongoing bull market uptrend.

    BUY : Golden cross intact + above SMA200 + price pulled back to EMA20
          + bullish reversal candle + RSI in recovery zone (45–68).
    SELL: RSI overbought OR N consecutive closes below EMA20 OR death cross.
    Stop : % below EMA20 (gives room for intraday wick through EMA).
    Target: entry + r_multiple × risk.
    """
    name = "EMA_Mean_Reversion"

    config: dict = {
        "ema_period":       20,
        "ema_distance_pct": 5.0,   # max % pullback from EMA to qualify (catches multi-week pullbacks)
        "min_ema_dist_pct": 0.5,   # minimum % from EMA — avoids entries already back at EMA with no pullback
        "stop_pct":         2.5,   # % below EMA for stop — 2.5% gives room on volatile stocks
        "r_multiple":       3.0,   # reward:risk target multiple
        "exit_bars_below":  3,     # consecutive closes below EMA to exit (3 bars = confirmed breakdown)
        "rsi_exit":         76,    # RSI overbought exit
        "min_data_bars":    220,
        # ── Calibration filters ──
        "filter_ema_dist_min": 0.0,
        "filter_vol_min":      0.0,
        "filter_bb_pos_min":   0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime == MarketRegime.DEEP_BEAR:
            return self._hold(symbol, "OFF in Deep Bear Regime")

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        opens = df["Open"]

        c_now = float(close.iloc[-1])
        sma200 = float(_sma(close, 200).iloc[-1])
        if regime == MarketRegime.BEAR and symbol not in _INDEX_ETFS and c_now <= sma200:
            return self._hold(symbol, "bear regime: only SPY/QQQ/DIA/IWM or symbols above SMA200")

        in_uptrend, regime_reason = _market_regime_reversion(df)
        if not in_uptrend:
            return self._hold(symbol, regime_reason)

        ema20  = _ema(close, cfg["ema_period"])
        rsi    = _rsi(close, 14)
        atr_v  = _current_atr(df, 14)

        c_now   = float(close.iloc[-1])
        o_now   = float(opens.iloc[-1])
        h_now   = float(high.iloc[-1])
        l_now   = float(low.iloc[-1])
        ema_now = float(ema20.iloc[-1])
        rsi_now = float(rsi.iloc[-1])

        ema_dist_pct = abs(c_now - ema_now) / ema_now * 100

        max_ema_distance = cfg["ema_distance_pct"]
        rsi_floor = 45
        if regime == MarketRegime.BEAR:
            max_ema_distance = min(max_ema_distance, 3.0)
            rsi_floor = 50

        bar_range      = h_now - l_now
        upper_half     = (c_now - l_now) / bar_range > 0.40 if bar_range > 0 else False
        bullish_candle = (c_now > o_now) and upper_half

        # Price touched or crossed below EMA within last 3 bars (actual pullback happened)
        touched_ema = any(
            float(low.iloc[i]) <= float(ema20.iloc[i]) * 1.002
            for i in range(-3, 0)
        )

        # ── Bear market emergency exit ────────────────────────
        if _is_death_cross(df):
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.95,
                reason="DEATH CROSS — SMA50 crossed below SMA200, exiting bull positions",
                indicators={"ema20": round(ema_now, 2), "rsi": round(rsi_now, 1)},
            )

        # ── Per-symbol calibration filters ───────────────────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters = get_filters_for_symbol(self.name, symbol)
        _f_ema = cfg["filter_ema_dist_min"] if cfg["filter_ema_dist_min"] > 0 else _sym_filters.get("ema_dist_min", 0.0)
        _f_vol = cfg["filter_vol_min"]      if cfg["filter_vol_min"]      > 0 else _sym_filters.get("vol_min", 0.0)
        _f_bb  = cfg["filter_bb_pos_min"]   if cfg["filter_bb_pos_min"]   > 0 else _sym_filters.get("bb_pos_min", 0.0)

        ema_dist_ok = ema_dist_pct >= _f_ema if _f_ema > 0 else True

        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= _f_vol if _f_vol > 0 else True

        bb_pos_ok = True
        if _f_bb > 0:
            bb_mid   = float(_sma(close, 20).iloc[-1])
            bb_std   = float(close.rolling(20).std().iloc[-1]) if len(close) >= 20 else 0.0
            bb_lower = bb_mid - 2.0 * bb_std
            bb_upper = bb_mid + 2.0 * bb_std
            bb_width = bb_upper - bb_lower
            bb_pos   = (c_now - bb_lower) / bb_width if bb_width > 0 else 0.5
            bb_pos_ok = bb_pos >= _f_bb

        # ── BUY ──────────────────────────────────────────────
        # RSI 45–68: in bull market pullbacks, RSI bottoms around 45-50, not 35-40.
        # RSI < 45 at the EMA = deeper correction underway, not a clean pullback.
        near_ema_now = ema_dist_pct <= cfg["ema_distance_pct"] * 2
        in_pullback  = (cfg["min_ema_dist_pct"] <= ema_dist_pct <= max_ema_distance) \
                       or (touched_ema and near_ema_now)

        if in_pullback and bullish_candle and c_now > ema_now \
                and rsi_floor < rsi_now < 68 and ema_dist_ok and vol_ok and bb_pos_ok:
            stop   = ema_now * (1 - cfg["stop_pct"] / 100)
            risk   = c_now - stop
            if risk < atr_v * 0.25:
                return self._hold(symbol, "stop too tight")
            target = c_now + cfg["r_multiple"] * risk
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.75,
                reason=f"Bull pullback to EMA({cfg['ema_period']}) "
                       f"(dist={ema_dist_pct:.1f}%, RSI={rsi_now:.0f})",
                indicators={
                    "ema20":        round(ema_now, 2),
                    "ema_dist_pct": round(ema_dist_pct, 2),
                    "rsi":          round(rsi_now, 1),
                    "atr":          round(atr_v, 2),
                    "stop":         round(stop, 2),
                    "target":       round(target, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']} — overbought",
                indicators={"rsi": round(rsi_now, 1), "ema20": round(ema_now, 2)},
            )

        n = cfg["exit_bars_below"]
        if len(close) >= n:
            consecutive_below = all(
                float(close.iloc[i]) < float(ema20.iloc[i])
                for i in range(-n, 0)
            )
            if consecutive_below:
                return PerplexitySignal(
                    symbol=symbol, strategy_name=self.name, direction="SELL",
                    entry_price=c_now, confidence=0.72,
                    reason=f"{n} consecutive closes below EMA({cfg['ema_period']})",
                    indicators={"ema20": round(ema_now, 2), "rsi": round(rsi_now, 1)},
                )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 2 — MA Crossover with RSI Confirmation
# ══════════════════════════════════════════════════════════════
class MaCrossoverRsi(PerplexityStrategy):
    """
    Catch early-trend momentum when EMA(fast) crosses above EMA(slow) in a bull market.

    BUY : Bull market intact + EMA(20) crosses above EMA(50) + RSI in momentum zone.
    SELL: EMA(20) crosses below EMA(50) OR RSI overbought OR death cross.
    Stop : below recent 10-bar swing low or slow EMA, whichever is lower.
    Target: entry + r_multiple × risk.
    """
    name = "MA_Crossover_RSI"

    config: dict = {
        "ema_fast":      20,
        "ema_slow":      50,
        "rsi_low":       45,    # RSI floor at crossover — below 45 = weak momentum, skip
        "rsi_high":      68,    # RSI ceiling at crossover — above 68 = already extended, skip
        "rsi_exit":      78,    # exit on overbought
        "r_multiple":    3.0,
        "min_data_bars": 220,
        # ── Calibration filters ──
        "filter_vol_min":        0.0,
        "filter_ema_spread_min": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL

        close = df["Close"]
        ema_f = _ema(close, cfg["ema_fast"])
        ema_s = _ema(close, cfg["ema_slow"])
        rsi   = _rsi(close, 14)
        atr_v = _current_atr(df, 14)

        ef_now  = float(ema_f.iloc[-1])
        ef_prev = float(ema_f.iloc[-2])
        es_now  = float(ema_s.iloc[-1])
        es_prev = float(ema_s.iloc[-2])
        rsi_now = float(rsi.iloc[-1])
        c_now   = float(close.iloc[-1])

        bullish_cross = ef_prev <= es_prev and ef_now > es_now
        bearish_cross = ef_prev >= es_prev and ef_now < es_now

        swing_low_10 = float(df["Low"].iloc[-11:-1].min())

        # ── Bear market emergency exit ────────────────────────
        if _is_death_cross(df):
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.95,
                reason="DEATH CROSS — SMA50 crossed below SMA200, exiting bull positions",
                indicators={"ema_fast": round(ef_now, 2), "ema_slow": round(es_now, 2)},
            )

        # ── Per-symbol calibration filters ───────────────────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters = get_filters_for_symbol(self.name, symbol)
        _f_vol    = cfg["filter_vol_min"]        if cfg["filter_vol_min"]        > 0 else _sym_filters.get("vol_min", 0.0)
        _f_spread = cfg["filter_ema_spread_min"] if cfg["filter_ema_spread_min"] > 0 else _sym_filters.get("ema_spread_min", 0.0)

        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= _f_vol if _f_vol > 0 else True

        ema_spread_pct = abs(ef_now - es_now) / es_now * 100 if es_now > 0 else 0.0
        spread_ok = ema_spread_pct >= _f_spread if _f_spread > 0 else True

        # ── BUY ──────────────────────────────────────────────
        if bullish_cross and cfg["rsi_low"] <= rsi_now <= cfg["rsi_high"] \
                and vol_ok and spread_ok:
            if regime != MarketRegime.BULL:
                return self._hold(symbol, "OFF in Bear Regime")
            stop   = min(swing_low_10, es_now * 0.995)
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight at crossover")
            if risk > atr_v * 3.0:
                return self._hold(symbol, "stop too wide — risk exceeds 3×ATR")
            target = c_now + cfg["r_multiple"] * risk
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.78,
                reason=f"EMA({cfg['ema_fast']}) crossed above EMA({cfg['ema_slow']}) "
                       f"RSI={rsi_now:.0f} [{cfg['rsi_low']}–{cfg['rsi_high']}]",
                indicators={
                    "ema_fast":  round(ef_now, 2),
                    "ema_slow":  round(es_now, 2),
                    "rsi":       round(rsi_now, 1),
                    "atr":       round(atr_v, 2),
                    "swing_low": round(swing_low_10, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if bearish_cross:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.82,
                reason=f"EMA({cfg['ema_fast']}) crossed BELOW EMA({cfg['ema_slow']}) — momentum gone",
                indicators={"ema_fast": round(ef_now, 2), "ema_slow": round(es_now, 2),
                            "rsi": round(rsi_now, 1)},
            )
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.72,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']} — overbought",
                indicators={"rsi": round(rsi_now, 1), "ema_fast": round(ef_now, 2)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 3 — Breakout from Consolidation with Volume
# ══════════════════════════════════════════════════════════════
class BreakoutConsolidation(PerplexityStrategy):
    """
    Buy confirmed breakouts from tight consolidation bases in a bull market.

    Requires price ABOVE SMA50 (already in trend) + golden cross + ADX > 20.
    BUY : tight base (< 2.5×ATR range) + close > range_high + strong volume (≥ 2×avg).
    SELL: failed breakout reversal OR RSI overbought OR death cross.
    Stop : just below range_high (the base of the breakout).
    Target: entry + r_multiple × risk.
    """
    name = "Breakout_Consolidation"

    config: dict = {
        "consolidation_bars":  15,    # bars defining the base — 15 catches 3-week bases
        "atr_range_multiple":  2.5,   # base range must be < 2.5×ATR — tighter bases break out harder
        "breakout_buffer_pct": 0.3,   # % above range_high for confirmed breakout
        "vol_multiple":        2.0,   # volume must be ≥ 2× avg — real breakouts need conviction
        "r_multiple":          3.0,
        "rsi_exit":            80,    # breakout momentum can carry RSI very high
        "stop_below_range":    True,
        "min_data_bars":       220,
        # ── Calibration filters ──
        "filter_vol_min":       0.0,
        "filter_range_atr_max": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL

        # Strict gate: must be above SMA50 with ADX > 20 (confirmed bull trend momentum)
        in_uptrend, regime_reason = _market_regime_trend(df)
        if not in_uptrend:
            return self._hold(symbol, regime_reason)

        close   = df["Close"]
        c_now   = float(close.iloc[-1])
        sma50   = float(_sma(close, 50).iloc[-1])
        sma200  = float(_sma(close, 200).iloc[-1])
        n       = cfg["consolidation_bars"]
        atr_v   = _current_atr(df, 14)
        rsi     = _rsi(close, 14)
        rsi_now = float(rsi.iloc[-1])

        window_h   = df["High"].iloc[-n - 1:-1]
        window_l   = df["Low"].iloc[-n - 1:-1]
        range_high = float(window_h.max())
        range_low  = float(window_l.min())
        range_size = range_high - range_low

        if range_size > cfg["atr_range_multiple"] * atr_v:
            return self._hold(symbol, f"base too wide ({range_size:.2f} > {cfg['atr_range_multiple']}×ATR)")

        buffer   = range_high * cfg["breakout_buffer_pct"] / 100
        breakout = c_now > range_high + buffer

        vol_ratio = 1.0
        vol_ok    = True
        if "Volume" in df.columns:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean())
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
            vol_ok    = cur_vol >= cfg["vol_multiple"] * avg_vol if avg_vol > 0 else True

        # ── Bear market emergency exit ────────────────────────
        if _is_death_cross(df):
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.95,
                reason="DEATH CROSS — SMA50 crossed below SMA200, exiting bull positions",
                indicators={"sma50": round(sma50, 2), "sma200": round(sma200, 2)},
            )

        # ── Per-symbol calibration filters ───────────────────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters    = get_filters_for_symbol(self.name, symbol)
        _f_vol          = cfg["filter_vol_min"]       if cfg["filter_vol_min"]       > 0 else _sym_filters.get("vol_min", 0.0)
        _f_range_atr    = cfg["filter_range_atr_max"] if cfg["filter_range_atr_max"] > 0 else _sym_filters.get("atr_pct_max", 0.0)
        extra_vol_ok    = vol_ratio >= _f_vol if _f_vol > 0 else True
        range_atr_ratio = range_size / atr_v if atr_v > 0 else 0.0
        range_tight_ok  = range_atr_ratio <= _f_range_atr if _f_range_atr > 0 else True

        # ── BUY ──────────────────────────────────────────────
        # RSI 50–75: breakout with RSI < 50 lacks momentum; > 75 = already extended
        if breakout and vol_ok and 50 < rsi_now < 75 and extra_vol_ok and range_tight_ok:
            if regime != MarketRegime.BULL:
                return self._hold(symbol, "OFF in Bear Regime")
            stop   = (range_high if cfg["stop_below_range"] else range_low) * 0.998
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight")
            target = c_now + cfg["r_multiple"] * risk
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.80,
                reason=f"Breakout above {n}-bar base (range={range_size:.2f}, "
                       f"vol={vol_ratio:.1f}×avg, RSI={rsi_now:.0f})",
                indicators={
                    "range_high": round(range_high, 2),
                    "range_low":  round(range_low, 2),
                    "range_size": round(range_size, 2),
                    "vol_ratio":  round(vol_ratio, 2),
                    "sma50":      round(sma50, 2),
                    "sma200":     round(sma200, 2),
                    "rsi":        round(rsi_now, 1),
                    "atr":        round(atr_v, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        recent_had_breakout = any(
            float(df["Close"].iloc[i]) > range_high * (1 + cfg["breakout_buffer_pct"] / 100)
            for i in range(-5, -1)
        )
        if recent_had_breakout and c_now < range_high:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason=f"Failed breakout — closed back below base high ({range_high:.2f})",
                indicators={"range_high": round(range_high, 2), "rsi": round(rsi_now, 1)},
            )
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.72,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']} — overbought",
                indicators={"rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 4 — Bollinger Band Mean Reversion in Uptrend
# ══════════════════════════════════════════════════════════════
class BollingerMeanReversionUptrend(PerplexityStrategy):
    """
    Buy short-term oversold extremes in a confirmed bull market.

    BUY : Golden cross intact + price pushed below BB lower band + re-enters bands
          + RSI recovering from oversold + not a falling-knife (RSI floor 38).
    SELL: Price reaches BB upper band OR RSI overbought OR death cross.
    Stop : 2×ATR below entry price.
    Target: BB upper band (full mean reversion).
    """
    name = "BB_Mean_Reversion"

    config: dict = {
        "bb_period":      20,
        "bb_std":         2.0,
        "reentry_bars":   5,      # bars to re-enter after touching lower band (5 = ~1 week of dip)
        "atr_multiple":   2.0,    # stop = entry - atr_multiple × ATR
        "rsi_re_entry":   38,     # RSI must be above this — below 38 = still in freefall, avoid
        "use_rsi_filter": True,
        "rsi_exit":       74,
        "min_data_bars":  220,
        # ── Calibration filters ──
        "filter_vol_min":      0.0,
        "filter_atr_pct_max":  0.0,
        "filter_bb_depth_min": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime == MarketRegime.DEEP_BEAR and symbol not in _INDEX_ETFS:
            return self._hold(symbol, "OFF in Deep Bear Regime")

        in_uptrend, regime_reason = _market_regime_reversion(df)
        if not in_uptrend:
            return self._hold(symbol, regime_reason)

        close = df["Close"]
        bb    = compute_bollinger(close, cfg["bb_period"], cfg["bb_std"])
        rsi   = _rsi(close, 14)
        atr_v = _current_atr(df, 14)

        bb_lower  = bb.lower.values
        bb_middle = bb.middle.values
        bb_upper  = bb.upper.values

        c_now     = float(close.iloc[-1])
        rsi_now   = float(rsi.iloc[-1])
        lower_now = float(bb_lower.iloc[-1])
        upper_now = float(bb_upper.iloc[-1])
        mid_now   = float(bb_middle.iloc[-1])

        # Must have meaningfully pierced the lower band (not just a 1-tick brush)
        m = cfg["reentry_bars"]
        was_below = any(
            float(close.iloc[i]) < float(bb_lower.iloc[i]) - 0.1 * atr_v
            for i in range(-m - 1, -1)
        )
        re_entered = was_below and c_now > lower_now

        rsi_prev       = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now
        rsi_crossed_up = (rsi_prev < cfg["rsi_re_entry"]) and (rsi_now >= cfg["rsi_re_entry"])
        rsi_ok         = (not cfg["use_rsi_filter"]) or rsi_crossed_up or rsi_now >= cfg["rsi_re_entry"]
        if regime == MarketRegime.BEAR:
            rsi_ok = rsi_ok and rsi_now <= 34

        # ── Bear market emergency exit ────────────────────────
        if _is_death_cross(df):
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.95,
                reason="DEATH CROSS — SMA50 crossed below SMA200, exiting bull positions",
                indicators={"bb_lower": round(lower_now, 2), "rsi": round(rsi_now, 1)},
            )

        # ── Per-symbol calibration filters ───────────────────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters = get_filters_for_symbol(self.name, symbol)
        _f_vol       = cfg["filter_vol_min"]      if cfg["filter_vol_min"]      > 0 else _sym_filters.get("vol_min", 0.0)
        _f_atr_max   = cfg["filter_atr_pct_max"]  if cfg["filter_atr_pct_max"]  > 0 else _sym_filters.get("atr_pct_max", 0.0)
        _f_depth_min = cfg["filter_bb_depth_min"] if cfg["filter_bb_depth_min"] > 0 else _sym_filters.get("bb_depth_min", 0.0)

        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= _f_vol if _f_vol > 0 else True

        atr_pct = atr_v / c_now * 100 if c_now > 0 else 0.0
        atr_ok  = atr_pct <= _f_atr_max if _f_atr_max > 0 else True

        bb_width    = float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
        depth_below = 0.0
        for k in range(-m - 1, -1):
            c_bar = float(close.iloc[k])
            bl    = float(bb_lower.iloc[k])
            if c_bar < bl and bb_width > 0:
                depth_below = max(depth_below, (bl - c_bar) / bb_width)
        depth_ok = depth_below >= _f_depth_min if _f_depth_min > 0 else True

        # ── BUY ──────────────────────────────────────────────
        # RSI 38–62: floor of 38 avoids capitulation, ceiling of 62 avoids buying after recovery
        if re_entered and rsi_ok and 38 <= rsi_now <= 62 and vol_ok and atr_ok and depth_ok:
            stop   = c_now - cfg["atr_multiple"] * atr_v
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight")
            target = upper_now
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.76,
                reason=f"Re-entered BB lower band in bull market — mean reversion, RSI={rsi_now:.0f}",
                indicators={
                    "bb_lower":  round(lower_now, 2),
                    "bb_middle": round(mid_now, 2),
                    "bb_upper":  round(upper_now, 2),
                    "rsi":       round(rsi_now, 1),
                    "atr":       round(atr_v, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if c_now >= upper_now or rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason="Price at BB upper — full reversion complete"
                       if c_now >= upper_now else f"RSI {rsi_now:.0f} > {cfg['rsi_exit']}",
                indicators={
                    "bb_upper":  round(upper_now, 2),
                    "bb_middle": round(mid_now, 2),
                    "rsi":       round(rsi_now, 1),
                },
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 5 — Support / Fibonacci Pullback in Trend
# ══════════════════════════════════════════════════════════════
class FibPullbackSupport(PerplexityStrategy):
    """
    Buy pullbacks to Fibonacci retracement levels (38.2%, 50%, 61.8%) of the
    most recent impulse leg in a confirmed bull market.

    BUY : Golden cross intact + above SMA200 + price at a Fib level
          + bullish rejection candle OR RSI turning up from oversold zone.
    SELL: Price reaches prior swing high OR RSI overbought OR death cross.
    Stop : Fib level − 1.5×ATR (gives room for normal volatility).
    Target: prior swing high.
    """
    name = "Fib_Pullback_Support"

    config: dict = {
        "swing_lookback":   60,    # bars to identify the impulse leg — 60 catches longer swings
        "fib_levels":       [0.382, 0.50, 0.618],
        "fib_zone_pct":     1.2,   # price within ±1.2% of fib level qualifies (slightly wider than 1%)
        "rsi_oversold_low": 35,    # RSI floor — below 35 is falling knife territory, skip
        "rsi_oversold_hi":  62,    # RSI ceiling — allows mid-term pullbacks that don't go deep
        "atr_stop_mult":    1.5,
        "min_data_bars":    220,
        # ── Calibration filters ──
        "filter_lower_wick_min": 0.0,
        "filter_vol_min":        0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime == MarketRegime.DEEP_BEAR and symbol not in _INDEX_ETFS:
            return self._hold(symbol, "OFF in Deep Bear Regime")

        in_uptrend, regime_reason = _market_regime_reversion(df)
        if not in_uptrend:
            return self._hold(symbol, regime_reason)

        close  = df["Close"]
        c_now  = float(close.iloc[-1])
        sma50  = float(_sma(close, 50).iloc[-1])
        sma200 = float(_sma(close, 200).iloc[-1])
        rsi    = _rsi(close, 14)
        atr_v  = _current_atr(df, 14)
        rsi_now = float(rsi.iloc[-1])

        if regime == MarketRegime.BEAR and symbol not in _INDEX_ETFS and c_now <= sma200:
            return self._hold(symbol, "bear regime: only large caps / ETFs above SMA200")

        # Identify the most recent upward impulse leg
        lb         = cfg["swing_lookback"]
        window_h   = df["High"].iloc[-lb:]
        window_l   = df["Low"].iloc[-lb:]
        swing_low_idx  = int(window_l.values.argmin())
        swing_high_idx = int(window_h.values.argmax())
        swing_low  = float(window_l.iloc[swing_low_idx])
        swing_high = float(window_h.iloc[swing_high_idx])

        # Upward impulse: low must come before high
        if swing_low_idx >= swing_high_idx:
            return self._hold(symbol, "no valid upward impulse — swing low after swing high")

        # Swing high must be recent (within 40 bars = ~2 months)
        bars_since_high = lb - 1 - swing_high_idx
        if bars_since_high > 40:
            return self._hold(symbol, f"impulse too old — swing high {bars_since_high} bars ago")

        impulse = swing_high - swing_low
        if impulse < atr_v * 3:
            return self._hold(symbol, "impulse too small — need ≥ 3×ATR swing")

        fib_zones = {
            level: swing_high - level * impulse
            for level in cfg["fib_levels"]
        }

        zone_pct  = cfg["fib_zone_pct"] / 100
        if regime != MarketRegime.BULL:
            zone_pct = min(zone_pct, 0.01)
        hit_level = None
        hit_price = None
        for level, fib_price in fib_zones.items():
            if abs(c_now - fib_price) / fib_price <= zone_pct:
                hit_level = level
                hit_price = fib_price
                break

        if hit_level is None:
            return self._hold(symbol, "price not at a Fibonacci level")

        o_now      = float(df["Open"].iloc[-1])
        h_now      = float(df["High"].iloc[-1])
        l_now      = float(df["Low"].iloc[-1])
        body       = abs(c_now - o_now)
        bar_range  = h_now - l_now
        lower_wick = o_now - l_now if c_now >= o_now else c_now - l_now
        lower_wick_pct   = lower_wick / bar_range * 100 if bar_range > 0 else 0.0
        rejection_candle = lower_wick >= 1.5 * body if body > 0 else False

        rsi_prev      = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now
        rsi_turning_up = (rsi_prev < rsi_now) and (cfg["rsi_oversold_low"] <= rsi_now <= cfg["rsi_oversold_hi"])

        entry_ok = rejection_candle or rsi_turning_up
        if regime != MarketRegime.BULL:
            entry_ok = rejection_candle and lower_wick_pct >= 20 or rsi_turning_up

        # ── Bear market emergency exit ────────────────────────
        if _is_death_cross(df):
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.95,
                reason="DEATH CROSS — SMA50 crossed below SMA200, exiting bull positions",
                indicators={"sma50": round(sma50, 2), "sma200": round(sma200, 2)},
            )

        # ── Per-symbol calibration filters ───────────────────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters = get_filters_for_symbol(self.name, symbol)
        _f_wick = cfg["filter_lower_wick_min"] if cfg["filter_lower_wick_min"] > 0 else _sym_filters.get("lower_wick_min", 0.0)
        _f_vol  = cfg["filter_vol_min"]        if cfg["filter_vol_min"]        > 0 else _sym_filters.get("vol_min", 0.0)

        wick_ok = lower_wick_pct >= _f_wick if _f_wick > 0 else True

        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= _f_vol if _f_vol > 0 else True

        # ── BUY ──────────────────────────────────────────────
        atr_stop_mult = cfg["atr_stop_mult"]
        if regime == MarketRegime.BEAR:
            atr_stop_mult = min(atr_stop_mult, 1.25)
        if entry_ok and cfg["rsi_oversold_low"] < rsi_now < cfg["rsi_oversold_hi"] \
                and wick_ok and vol_ok:
            stop   = hit_price - atr_stop_mult * atr_v
            stop   = min(stop, l_now - 0.1 * atr_v)
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight at Fib level")
            target       = swing_high
            reason_parts = []
            if rejection_candle:
                reason_parts.append("rejection candle")
            if rsi_turning_up:
                reason_parts.append(f"RSI turning up ({rsi_now:.0f})")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.76,
                reason=f"Bull pullback to {hit_level:.1%} Fib (${hit_price:.2f}) — "
                       + ", ".join(reason_parts),
                indicators={
                    "swing_low":  round(swing_low, 2),
                    "swing_high": round(swing_high, 2),
                    "fib_382":    round(fib_zones[0.382], 2),
                    "fib_500":    round(fib_zones[0.50], 2),
                    "fib_618":    round(fib_zones[0.618], 2),
                    "hit_level":  f"{hit_level:.1%}",
                    "hit_price":  round(hit_price, 2),
                    "rsi":        round(rsi_now, 1),
                    "sma50":      round(sma50, 2),
                    "sma200":     round(sma200, 2),
                    "atr":        round(atr_v, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        near_swing_high = c_now >= swing_high * 0.97
        if near_swing_high or rsi_now > 74:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.76,
                reason=f"Approaching swing high ({swing_high:.2f}) — take profit"
                       if near_swing_high else f"RSI {rsi_now:.0f} > 74 — overbought",
                indicators={"swing_high": round(swing_high, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)
