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
"""

import pandas as pd

from app.services.indicators.atr import compute_atr
from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma
from app.services.indicators.ema import compute_ema
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


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


def _above_sma200(df: pd.DataFrame) -> bool:
    if len(df) < 200:
        return False
    s = _sma(df["Close"], 200).iloc[-1]
    return float(df["Close"].iloc[-1]) > float(s)


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
    Mean reversion to the 20 EMA inside a strong uptrend.

    BUY : close > SMA(200)  AND  price pulls back within ema_distance_pct of EMA(20)
          AND bullish reversal candle (close > open, close in upper half of range).
    SELL: close < EMA(20) for N consecutive bars  OR  RSI > rsi_exit.
    Stop : min(candle_low, EMA20 * (1 - stop_pct))
    Target: entry + 2R  (configurable R multiple)
    """
    name = "EMA_Mean_Reversion"

    # Default configurable parameters
    config: dict = {
        "ema_period":        20,    # EMA period for the pullback anchor
        "ema_distance_pct":  4.0,   # max % distance from EMA to qualify as a pullback
        "stop_pct":          1.5,   # % below EMA(20) for the stop (fallback)
        "r_multiple":        2.5,   # reward-to-risk target multiple
        "exit_bars_below":   2,     # consecutive closes below EMA to trigger SELL
        "rsi_exit":          75,    # RSI level at which to exit (overbought)
        "min_data_bars":     220,   # minimum history needed
        # ── Data-driven filters (discovered via trade pattern analysis) ──
        "filter_ema_dist_min":  0.0,   # require EMA distance >= this % (0 = off, 1.5 = recommended)
        "filter_vol_min":       0.0,   # require volume ratio >= this (0 = off, 1.0 = recommended)
        "filter_bb_pos_min":    0.0,   # require BB position >= this (0 = off, 0.72 = recommended)
    }

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")
        if not _above_sma200(df):
            return self._hold(symbol, "below SMA(200) — no uptrend")

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        opens  = df["Open"]

        ema20  = _ema(close, cfg["ema_period"])
        rsi    = _rsi(close, 14)
        atr_v  = _current_atr(df, 14)

        c_now      = float(close.iloc[-1])
        o_now      = float(opens.iloc[-1])
        h_now      = float(high.iloc[-1])
        l_now      = float(low.iloc[-1])
        ema_now    = float(ema20.iloc[-1])
        rsi_now    = float(rsi.iloc[-1])

        # Distance from EMA as a %
        ema_dist_pct = abs(c_now - ema_now) / ema_now * 100

        # Bullish reversal candle: close > open AND close in upper 40% of bar range
        bar_range = h_now - l_now
        upper_half = (c_now - l_now) / bar_range > 0.40 if bar_range > 0 else False
        bullish_candle = (c_now > o_now) and upper_half

        # Pullback: price dipped to or below EMA within the last 3 bars, now recovering
        touched_ema = any(
            float(low.iloc[i]) <= ema_now * 1.005
            for i in range(-3, 0)
        )

        # ── Data-driven filters (per-symbol profile, falls back to config) ──────
        from app.services.backtest.symbol_profiles import get_filters_for_symbol
        _sym_filters = get_filters_for_symbol(self.name, symbol)
        # Per-symbol profile takes precedence; config values used as override
        # when explicitly set (non-zero in config means user manually tuned it)
        _f_ema  = cfg["filter_ema_dist_min"] if cfg["filter_ema_dist_min"] > 0 else _sym_filters["ema_dist_min"]
        _f_vol  = cfg["filter_vol_min"]      if cfg["filter_vol_min"]      > 0 else _sym_filters["vol_min"]
        _f_bb   = cfg["filter_bb_pos_min"]   if cfg["filter_bb_pos_min"]   > 0 else _sym_filters["bb_pos_min"]

        # EMA distance filter
        ema_dist_ok = ema_dist_pct >= _f_ema if _f_ema > 0 else True

        # Volume filter
        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol = float(df["Volume"].iloc[-21:-1].mean()) if len(df) > 21 else 1.0
            cur_vol = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= _f_vol if _f_vol > 0 else True

        # BB position filter
        bb_pos_ok = True
        if _f_bb > 0:
            bb_mid = float(_sma(close, 20).iloc[-1])
            bb_std = float(close.rolling(20).std().iloc[-1]) if len(close) >= 20 else 0.0
            bb_lower = bb_mid - 2.0 * bb_std
            bb_upper = bb_mid + 2.0 * bb_std
            bb_width = bb_upper - bb_lower
            bb_pos = (c_now - bb_lower) / bb_width if bb_width > 0 else 0.5
            bb_pos_ok = bb_pos >= _f_bb

        # ── BUY ──────────────────────────────────────────────
        in_pullback = ema_dist_pct <= cfg["ema_distance_pct"] or touched_ema
        if in_pullback and bullish_candle and c_now > ema_now and 40 < rsi_now < 70 \
                and ema_dist_ok and vol_ok and bb_pos_ok:
            stop   = min(l_now, ema_now * (1 - cfg["stop_pct"] / 100))
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
                reason=f"Bullish bounce off EMA({cfg['ema_period']}) in uptrend "
                       f"(dist={ema_dist_pct:.1f}%, RSI={rsi_now:.0f})",
                indicators={
                    "ema20": round(ema_now, 2),
                    "ema_dist_pct": round(ema_dist_pct, 2),
                    "rsi": round(rsi_now, 1),
                    "atr": round(atr_v, 2),
                    "stop": round(stop, 2),
                    "target": round(target, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']} — overbought exit",
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
    Catch new swings when the faster EMA crosses above the slower one,
    confirmed by RSI in the momentum zone, with optional SMA(200) trend filter.

    BUY : EMA(fast) crosses ABOVE EMA(slow)  AND  RSI in [rsi_low, rsi_high]
          AND (close > SMA(200) if use_sma200 is True).
    SELL: EMA(fast) crosses BELOW EMA(slow)  OR  RSI > rsi_exit.
    Stop : below EMA(slow) or recent 10-bar swing low.
    Target: entry + r_multiple * risk.
    """
    name = "MA_Crossover_RSI"

    config: dict = {
        "ema_fast":      20,
        "ema_slow":      50,
        "use_sma200":    False,  # require close > SMA(200)
        "rsi_low":       30,     # RSI must be above this at crossover
        "rsi_high":      75,     # RSI must be below this at crossover (not already overbought)
        "rsi_exit":      75,     # RSI level to trigger an exit
        "r_multiple":    2.5,
        "min_data_bars": 220,
    }

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")
        if cfg["use_sma200"] and not _above_sma200(df):
            return self._hold(symbol, "below SMA(200) — no uptrend")

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

        bullish_cross  = ef_prev <= es_prev and ef_now > es_now
        bearish_cross  = ef_prev >= es_prev and ef_now < es_now

        # Recent swing low as a stop anchor
        swing_low_10 = float(df["Low"].iloc[-11:-1].min())

        # ── BUY ──────────────────────────────────────────────
        if bullish_cross and cfg["rsi_low"] <= rsi_now <= cfg["rsi_high"]:
            stop   = min(swing_low_10, es_now * 0.995)
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight at crossover")
            target = c_now + cfg["r_multiple"] * risk
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.78,
                reason=f"EMA({cfg['ema_fast']}) crossed above EMA({cfg['ema_slow']}) "
                       f"with RSI={rsi_now:.0f} in momentum zone [{cfg['rsi_low']}–{cfg['rsi_high']}]",
                indicators={
                    "ema_fast": round(ef_now, 2),
                    "ema_slow": round(es_now, 2),
                    "rsi": round(rsi_now, 1),
                    "atr": round(atr_v, 2),
                    "swing_low": round(swing_low_10, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if bearish_cross:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason=f"EMA({cfg['ema_fast']}) crossed BELOW EMA({cfg['ema_slow']})",
                indicators={"ema_fast": round(ef_now, 2), "ema_slow": round(es_now, 2),
                            "rsi": round(rsi_now, 1)},
            )
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.72,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']}",
                indicators={"rsi": round(rsi_now, 1), "ema_fast": round(ef_now, 2)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 3 — Breakout from Consolidation with Volume
# ══════════════════════════════════════════════════════════════
class BreakoutConsolidation(PerplexityStrategy):
    """
    Breakout from a tight consolidation range in the direction of the trend.

    Trend filter: close > SMA(50) AND close > SMA(200).
    Consolidation: last N bars trade in a range narrower than atr_range_multiple * ATR.
    BUY : close > range_high + breakout_buffer%  AND  volume > vol_multiple * avg_vol(20).
    SELL: close < range_high (failed breakout) OR RSI > rsi_exit.
    Stop : just below range_high (or range_low for wider stop).
    Target: entry + r_multiple * risk.
    """
    name = "Breakout_Consolidation"

    config: dict = {
        "consolidation_bars":  8,     # N bars to define the consolidation range
        "atr_range_multiple":  7.0,   # range must be < this * ATR to qualify as consolidation
        "breakout_buffer_pct": 0.2,   # % above range_high required for confirmed breakout
        "vol_multiple":        1.4,   # volume must exceed this * 20-day avg volume
        "r_multiple":          2.5,
        "rsi_exit":            75,
        "stop_below_range":    True,  # True = stop below range_high; False = below range_low
        "min_data_bars":       220,
    }

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        # Trend filter
        sma50  = float(_sma(close, 50).iloc[-1])
        sma200 = float(_sma(close, 200).iloc[-1])
        if c_now <= sma50 or c_now <= sma200:
            return self._hold(symbol, "below SMA(50) or SMA(200) — no trend")

        n   = cfg["consolidation_bars"]
        atr_v = _current_atr(df, 14)
        rsi   = _rsi(close, 14)
        rsi_now = float(rsi.iloc[-1])

        # Consolidation window = bars[-n-1 : -1] (exclude current breakout bar)
        window_h = df["High"].iloc[-n - 1:-1]
        window_l = df["Low"].iloc[-n - 1:-1]
        range_high = float(window_h.max())
        range_low  = float(window_l.min())
        range_size = range_high - range_low

        # Range must be tight: < atr_range_multiple * ATR
        if range_size > cfg["atr_range_multiple"] * atr_v:
            return self._hold(symbol, f"range too wide ({range_size:.2f} > {cfg['atr_range_multiple']}×ATR)")

        # Breakout: current close above range_high + buffer
        buffer = range_high * cfg["breakout_buffer_pct"] / 100
        breakout = c_now > range_high + buffer

        # Volume confirmation
        vol_ok = True
        if "Volume" in df.columns:
            avg_vol = float(df["Volume"].iloc[-21:-1].mean())
            cur_vol = float(df["Volume"].iloc[-1])
            vol_ok  = cur_vol > cfg["vol_multiple"] * avg_vol if avg_vol > 0 else True

        # ── BUY ──────────────────────────────────────────────
        if breakout and vol_ok and 40 < rsi_now < cfg["rsi_exit"]:
            stop   = range_high if cfg["stop_below_range"] else range_low
            stop   = stop * 0.998  # tiny buffer below level
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight")
            target = c_now + cfg["r_multiple"] * risk
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.78,
                reason=f"Breakout above {n}-bar consolidation range "
                       f"(range={range_size:.2f}, RSI={rsi_now:.0f})",
                indicators={
                    "range_high": round(range_high, 2),
                    "range_low":  round(range_low, 2),
                    "range_size": round(range_size, 2),
                    "sma50":  round(sma50, 2),
                    "sma200": round(sma200, 2),
                    "rsi":    round(rsi_now, 1),
                    "atr":    round(atr_v, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.72,
                reason=f"RSI {rsi_now:.0f} > {cfg['rsi_exit']}",
                indicators={"rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 4 — Bollinger Band Mean Reversion in Uptrend
# ══════════════════════════════════════════════════════════════
class BollingerMeanReversionUptrend(PerplexityStrategy):
    """
    Fade short-term oversold extremes in a bullish regime using Bollinger Bands.

    Trend filter: close > SMA(200).
    BUY : close < BB lower  THEN  within M bars close re-enters bands (close > BB lower)
          AND optional RSI crosses back above rsi_re_entry.
    SELL: close >= BB upper  OR  RSI > rsi_exit.
    Stop : atr_multiple * ATR(14) below entry.
    Target: partial at BB middle, final at BB upper.
    """
    name = "BB_Mean_Reversion"

    config: dict = {
        "bb_period":      20,
        "bb_std":         2.0,
        "reentry_bars":   5,      # M: bars allowed to re-enter after touching lower band
        "atr_multiple":   2.0,    # stop = entry - atr_multiple * ATR
        "rsi_re_entry":   32,     # RSI must cross above this to confirm re-entry
        "use_rsi_filter": False,  # require RSI crossover above rsi_re_entry
        "rsi_exit":       72,
        "min_data_bars":  220,
    }

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")
        if not _above_sma200(df):
            return self._hold(symbol, "below SMA(200) — no uptrend")

        close   = df["Close"]
        bb      = compute_bollinger(close, cfg["bb_period"], cfg["bb_std"])
        rsi     = _rsi(close, 14)
        atr_v   = _current_atr(df, 14)

        bb_lower  = bb.lower.values
        bb_middle = bb.middle.values
        bb_upper  = bb.upper.values

        c_now     = float(close.iloc[-1])
        rsi_now   = float(rsi.iloc[-1])
        lower_now = float(bb_lower.iloc[-1])
        upper_now = float(bb_upper.iloc[-1])
        mid_now   = float(bb_middle.iloc[-1])

        # Check if price was below lower band in the last M bars then re-entered
        m = cfg["reentry_bars"]
        was_below = any(float(close.iloc[i]) < float(bb_lower.iloc[i])
                        for i in range(-m - 1, -1))
        re_entered = was_below and c_now > lower_now

        # Optional RSI re-entry filter: RSI crossed above threshold
        rsi_prev = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now
        rsi_crossed_up = (rsi_prev < cfg["rsi_re_entry"]) and (rsi_now >= cfg["rsi_re_entry"])
        rsi_ok = (not cfg["use_rsi_filter"]) or rsi_crossed_up or rsi_now >= cfg["rsi_re_entry"]

        # ── BUY ──────────────────────────────────────────────
        if re_entered and rsi_ok and rsi_now < 55:
            stop   = c_now - cfg["atr_multiple"] * atr_v
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight")
            # Target: BB upper (full mean-reversion)
            target = upper_now
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.76,
                reason=f"Re-entered BB lower band (mean reversion), RSI={rsi_now:.0f}",
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
                reason="Price at BB upper — full mean-reversion target reached"
                       if c_now >= upper_now else f"RSI {rsi_now:.0f} > {cfg['rsi_exit']}",
                indicators={
                    "bb_upper": round(upper_now, 2),
                    "bb_middle": round(mid_now, 2),
                    "rsi": round(rsi_now, 1),
                },
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 5 — Support / Fibonacci Pullback in Trend
# ══════════════════════════════════════════════════════════════
class FibPullbackSupport(PerplexityStrategy):
    """
    Buy pullbacks to Fibonacci retracement levels (38.2%, 50%, 61.8%) of the
    most recent impulse leg in a confirmed uptrend.

    Trend filter: close > SMA(50) AND close > SMA(200), higher highs/lows implied.
    Impulse leg: swing_low to swing_high over the last `swing_lookback` bars.
    Entry: price pulls into a Fib zone AND shows a bullish rejection candle OR
           RSI turns up from 30–45 oversold zone.
    Stop : just below the Fib zone or candle low.
    Target: prior swing high.
    """
    name = "Fib_Pullback_Support"

    config: dict = {
        "swing_lookback":   40,   # bars to identify the impulse leg
        "fib_levels":       [0.382, 0.50, 0.618],  # Fibonacci retracement levels
        "fib_zone_pct":     2.0,  # price is "at a fib level" if within ±2.0%
        "rsi_oversold_low": 30,   # RSI floor for entry (below = potential falling knife)
        "rsi_oversold_hi":  45,   # RSI must be below this at entry
        "atr_stop_mult":    1.5,  # stop = entry - atr_stop_mult * ATR
        "r_multiple":       2.5,
        "min_data_bars":    220,
    }

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        # Trend filter
        sma50  = float(_sma(close, 50).iloc[-1])
        sma200 = float(_sma(close, 200).iloc[-1])
        if c_now <= sma50 or c_now <= sma200:
            return self._hold(symbol, "below SMA(50) or SMA(200) — no uptrend")

        rsi    = _rsi(close, 14)
        atr_v  = _current_atr(df, 14)
        rsi_now = float(rsi.iloc[-1])

        # Identify impulse leg: swing low and swing high over last `swing_lookback` bars
        lb = cfg["swing_lookback"]
        window_h = df["High"].iloc[-lb:]
        window_l = df["Low"].iloc[-lb:]
        swing_low  = float(window_l.min())
        swing_high = float(window_h.max())
        impulse    = swing_high - swing_low

        if impulse < atr_v:
            return self._hold(symbol, "impulse leg too small — no clear swing to retrace")

        # Compute Fibonacci retracement levels from swing_high down
        fib_zones = {
            level: swing_high - level * impulse
            for level in cfg["fib_levels"]
        }

        # Check if current price is within ±fib_zone_pct% of any Fib level
        zone_pct = cfg["fib_zone_pct"] / 100
        hit_level = None
        hit_price = None
        for level, fib_price in fib_zones.items():
            if abs(c_now - fib_price) / fib_price <= zone_pct:
                hit_level = level
                hit_price = fib_price
                break

        if hit_level is None:
            return self._hold(symbol, "price not at a Fibonacci retracement level")

        # Bullish rejection candle: lower wick >= 1.5× the body, or bullish engulfing
        o_now  = float(df["Open"].iloc[-1])
        h_now  = float(df["High"].iloc[-1])
        l_now  = float(df["Low"].iloc[-1])
        body   = abs(c_now - o_now)
        lower_wick = o_now - l_now if c_now >= o_now else c_now - l_now
        rejection_candle = lower_wick >= 1.5 * body if body > 0 else False

        # RSI turning up from oversold zone
        rsi_prev = float(rsi.iloc[-2]) if len(rsi) >= 2 else rsi_now
        rsi_turning_up = (rsi_prev < rsi_now) and (cfg["rsi_oversold_low"] <= rsi_now <= cfg["rsi_oversold_hi"])

        entry_ok = rejection_candle or rsi_turning_up

        # ── BUY ──────────────────────────────────────────────
        if entry_ok and cfg["rsi_oversold_low"] < rsi_now < 65:
            stop   = min(l_now, hit_price * (1 - cfg["atr_stop_mult"] * atr_v / c_now))
            stop   = max(stop, c_now - cfg["atr_stop_mult"] * atr_v)  # fallback
            risk   = c_now - stop
            if risk < atr_v * 0.2:
                return self._hold(symbol, "stop too tight at Fib level")
            target = swing_high  # target = prior swing high
            reason_parts = []
            if rejection_candle:
                reason_parts.append("rejection candle")
            if rsi_turning_up:
                reason_parts.append(f"RSI turning up from oversold ({rsi_now:.0f})")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.74,
                reason=f"Pullback to {hit_level:.1%} Fib level (${hit_price:.2f}) — "
                       + ", ".join(reason_parts),
                indicators={
                    "swing_low":   round(swing_low, 2),
                    "swing_high":  round(swing_high, 2),
                    "fib_382":     round(fib_zones[0.382], 2),
                    "fib_500":     round(fib_zones[0.50], 2),
                    "fib_618":     round(fib_zones[0.618], 2),
                    "hit_level":   f"{hit_level:.1%}",
                    "hit_price":   round(hit_price, 2),
                    "rsi":         round(rsi_now, 1),
                    "sma50":       round(sma50, 2),
                    "sma200":      round(sma200, 2),
                    "atr":         round(atr_v, 2),
                },
            )

        # ── SELL ─────────────────────────────────────────────
        # Exit if price approaches prior swing high or RSI overbought
        near_swing_high = c_now >= swing_high * 0.98
        if near_swing_high or rsi_now > 72:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.75,
                reason="Near prior swing high — take profit"
                       if near_swing_high else f"RSI {rsi_now:.0f} > 72",
                indicators={"swing_high": round(swing_high, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)
