from __future__ import annotations

"""
5 Perplexity swing trading strategies — rewritten for higher frequency (8-20 trades/year).

Strategy 1 — EmaMeanReversionUptrend  → RSI-2 Mean Reversion (Connors)
Strategy 2 — MaCrossoverRsi           → EMA(9/21) + MACD Confirmation
Strategy 3 — BreakoutConsolidation    → BB Width Squeeze → Expansion Breakout
Strategy 4 — BollingerMeanReversionUptrend → Pullback to Rising EMA(50) + Wick
Strategy 5 — FibPullbackSupport       → ATR-Spike / Fear-Capitulation Reversal

All operate on daily OHLCV bars. Long-only. Configurable via the Config tab.
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


def _spy_is_bull(df: pd.DataFrame) -> bool:
    """SPY above its own SMA(200) = BULL."""
    close = df["Close"]
    if len(close) < 200:
        return True
    return float(close.iloc[-1]) > float(_sma(close, 200).iloc[-1])


def _macd_line(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd, signal_line) as floats for the last bar."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return float(macd.iloc[-1]), float(sig.iloc[-1]), macd, sig


def _bb_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (upper, middle, lower) as pd.Series."""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std(ddof=0)
    return mid + std * sigma, mid, mid - std * sigma


def _swing_high_low(highs: pd.Series, lows: pd.Series, lookback: int = 30):
    window_h = highs.iloc[-lookback:]
    window_l = lows.iloc[-lookback:]
    return float(window_l.min()), float(window_h.max())


# ══════════════════════════════════════════════════════════════
# STRATEGY 1 — RSI-2 Mean Reversion (Connors)
# ══════════════════════════════════════════════════════════════
class EmaMeanReversionUptrend(PerplexityStrategy):
    """
    Connors RSI(2) mean reversion: buy extreme short-term oversold dips
    in a structural bull market (price > SMA200).

    BUY : price > SMA(200) (BULL only) + RSI(2) < entry_threshold
          + not in high-volatility spike (ATR% guard).
    SELL: RSI(2) > exit_threshold OR close > SMA(exit_sma) OR max_hold reached.
    Stop : hard_stop_pct % below entry.
    Target: entry × (1 + take_profit_pct/100).
    """
    name = "EMA_Mean_Reversion"

    config: dict = {
        "ema_period":       20,        # kept for UI compatibility (unused in new logic)
        "ema_distance_pct": 5.0,       # kept for UI compatibility
        "min_ema_dist_pct": 0.5,       # kept for UI compatibility
        "stop_pct":         2.5,       # kept for UI compatibility
        "r_multiple":       3.0,       # kept for UI compatibility
        "exit_bars_below":  3,         # kept for UI compatibility
        "rsi_exit":         76,        # kept for UI compatibility
        "min_data_bars":    220,
        # ── New logic params ──
        "rsi_period":            2,
        "rsi_entry_threshold":   10,   # RSI(2) < 10 → deeply oversold
        "rsi_exit_threshold":    70,   # RSI(2) > 70 → mean reversion complete
        "sma_trend":             200,  # must be above this for BULL
        "exit_sma":              5,    # close > SMA(5) as alternate exit
        "hard_stop_pct":         5.0,
        "take_profit_pct":       8.0,
        "max_hold_bars":         10,
        "atr_skip_threshold":    5.0,  # skip if ATR% > this (volatile spike day)
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
        if regime != MarketRegime.BULL:
            return self._hold(symbol, "RSI-2 BULL only")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        # Must be above SMA(200)
        sma200 = float(_sma(close, cfg["sma_trend"]).iloc[-1])
        if c_now <= sma200:
            return self._hold(symbol, f"price below SMA({cfg['sma_trend']})")

        # Skip high-volatility spike days
        atr_v   = _current_atr(df, 14)
        atr_pct = atr_v / c_now * 100
        if atr_pct > cfg["atr_skip_threshold"]:
            return self._hold(symbol, f"ATR%={atr_pct:.1f} — volatility spike, skip")

        rsi2    = _rsi(close, cfg["rsi_period"])
        rsi_now = float(rsi2.iloc[-1])
        sma5    = float(_sma(close, cfg["exit_sma"]).iloc[-1])

        # ── BUY ──
        if rsi_now < cfg["rsi_entry_threshold"]:
            stop   = c_now * (1 - cfg["hard_stop_pct"] / 100)
            target = c_now * (1 + cfg["take_profit_pct"] / 100)
            confidence = max(0.5, min(0.95, 1.0 - rsi_now / cfg["rsi_entry_threshold"]))
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(confidence, 2),
                reason=f"RSI({cfg['rsi_period']})={rsi_now:.1f} < {cfg['rsi_entry_threshold']} — extreme oversold in bull trend",
                indicators={
                    "rsi2":   round(rsi_now, 1),
                    "sma200": round(sma200, 2),
                    "atr_pct": round(atr_pct, 2),
                },
            )

        # ── SELL ──
        if rsi_now > cfg["rsi_exit_threshold"] or c_now > sma5:
            reason = (f"RSI({cfg['rsi_period']})={rsi_now:.1f} > {cfg['rsi_exit_threshold']}"
                      if rsi_now > cfg["rsi_exit_threshold"]
                      else f"close > SMA({cfg['exit_sma']}) — mean reversion complete")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason=reason,
                indicators={"rsi2": round(rsi_now, 1), "sma5": round(sma5, 2)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 2 — EMA(9/21) Crossover + MACD Confirmation
# ══════════════════════════════════════════════════════════════
class MaCrossoverRsi(PerplexityStrategy):
    """
    Buy when EMA(9) crosses above EMA(21) with MACD above its signal line
    and RSI in momentum zone. BULL market only.

    BUY : BULL + EMA(9) crosses above EMA(21) + MACD > signal + RSI in [rsi_min, rsi_max]
          + volume ≥ vol_ratio_min × 20-bar avg.
    SELL: EMA(9) crosses below EMA(21) OR max_hold OR ATR stop hit.
    Stop : entry − atr_stop_multiplier × ATR.
    Target: entry + atr_tp_multiplier × ATR.
    """
    name = "MA_Crossover_RSI"

    config: dict = {
        "ema_fast":      20,       # kept for UI compatibility
        "ema_slow":      50,       # kept for UI compatibility
        "rsi_low":       45,       # kept for UI compatibility
        "rsi_high":      68,       # kept for UI compatibility
        "rsi_exit":      78,       # kept for UI compatibility
        "r_multiple":    3.0,      # kept for UI compatibility
        "min_data_bars": 220,
        # ── New logic params ──
        "ema_fast_new":         9,
        "ema_slow_new":         21,
        "macd_fast":            12,
        "macd_slow":            26,
        "macd_signal":          9,
        "rsi_period":           14,
        "rsi_min":              35,
        "rsi_max":              75,
        "vol_ratio_min":        0.8,
        "atr_stop_multiplier":  1.5,
        "atr_tp_multiplier":    2.5,
        "max_hold_bars":        20,
        "crossover_lookback":   5,    # allow crossover within N bars (not exact)
        # ── Calibration filters ──
        "filter_vol_min":        0.0,
        "filter_ema_spread_min": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime != MarketRegime.BULL:
            return self._hold(symbol, "EMA/MACD crossover BULL only")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        ema9  = _ema(close, cfg["ema_fast_new"])
        ema21 = _ema(close, cfg["ema_slow_new"])
        rsi14 = _rsi(close, cfg["rsi_period"])

        e9_now  = float(ema9.iloc[-1])
        e9_prev = float(ema9.iloc[-2])
        e21_now = float(ema21.iloc[-1])
        e21_prev = float(ema21.iloc[-2])
        rsi_now = float(rsi14.iloc[-1])

        macd_val, sig_val, macd_s, sig_s = _macd_line(
            close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
        )
        atr_v = _current_atr(df, 14)

        # Crossover within the last N bars (not just exact bar)
        lb = cfg.get("crossover_lookback", 3)
        lb = min(lb, len(ema9) - 1)
        bullish_cross = (e9_now > e21_now) and any(
            float(ema9.iloc[-(i+2)]) <= float(ema21.iloc[-(i+2)])
            for i in range(lb)
        )
        bearish_cross = (e9_now < e21_now) and any(
            float(ema9.iloc[-(i+2)]) >= float(ema21.iloc[-(i+2)])
            for i in range(lb)
        )

        vol_ratio = 1.0
        if "Volume" in df.columns and len(df) > 21:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean())
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        # ── BUY ──
        if (bullish_cross
                and macd_val > sig_val
                and cfg["rsi_min"] <= rsi_now <= cfg["rsi_max"]
                and vol_ratio >= cfg["vol_ratio_min"]):
            stop   = c_now - cfg["atr_stop_multiplier"] * atr_v
            target = c_now + cfg["atr_tp_multiplier"] * atr_v
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.75,
                reason=f"EMA(9) crossed EMA(21), MACD above signal, RSI={rsi_now:.0f}, vol={vol_ratio:.1f}×",
                indicators={
                    "ema9":    round(e9_now, 2),
                    "ema21":   round(e21_now, 2),
                    "macd":    round(macd_val, 4),
                    "signal":  round(sig_val, 4),
                    "rsi":     round(rsi_now, 1),
                    "atr":     round(atr_v, 2),
                    "vol_ratio": round(vol_ratio, 2),
                },
            )

        # ── SELL ──
        if bearish_cross:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.80,
                reason="EMA(9) crossed below EMA(21) — momentum lost",
                indicators={"ema9": round(e9_now, 2), "ema21": round(e21_now, 2), "rsi": round(rsi_now, 1)},
            )
        if rsi_now > cfg["rsi_exit"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.72,
                reason=f"RSI={rsi_now:.0f} > {cfg['rsi_exit']} — overbought",
                indicators={"rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 3 — BB Width Squeeze → Expansion Breakout
# ══════════════════════════════════════════════════════════════
class BreakoutConsolidation(PerplexityStrategy):
    """
    Detect Bollinger Band squeeze (bandwidth contracting squeeze_bars in a row)
    then buy the expansion (close above upper BB) with volume confirmation.
    BULL only.

    BUY : BULL + BB width contracting squeeze_bars consecutively + close > upper_BB
          + RSI > rsi_entry_min + volume ≥ vol_ratio_min × avg.
    SELL: RSI > rsi_overbought OR max_hold reached.
    Stop : lower BB at entry.
    Target: entry + (upper_BB − lower_BB) (one bandwidth extension).
    """
    name = "Breakout_Consolidation"

    config: dict = {
        "consolidation_bars":  15,    # kept for UI compatibility
        "atr_range_multiple":  2.5,   # kept for UI compatibility
        "breakout_buffer_pct": 0.3,   # kept for UI compatibility
        "vol_multiple":        2.0,   # kept for UI compatibility
        "r_multiple":          3.0,   # kept for UI compatibility
        "rsi_exit":            80,    # kept for UI compatibility
        "stop_below_range":    True,  # kept for UI compatibility
        "min_data_bars":       220,
        # ── New logic params ──
        "bb_period":       20,
        "bb_std":          2.0,
        "squeeze_bars":    3,          # consecutive bars of narrowing bandwidth
        "rsi_period":      14,
        "rsi_entry_min":   45,         # RSI must be above 45 at breakout
        "rsi_overbought":  80,
        "vol_ratio_min":   1.1,
        "max_hold_bars":   15,
        # ── Calibration filters ──
        "filter_vol_min":       0.0,
        "filter_range_atr_max": 0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime != MarketRegime.BULL:
            return self._hold(symbol, "BB squeeze breakout BULL only")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        bb_upper, bb_mid, bb_lower = _bb_bands(close, cfg["bb_period"], cfg["bb_std"])
        bw = bb_upper - bb_lower  # bandwidth series

        # Check squeeze: squeeze happened recently (within squeeze_bars * 2 window)
        n = cfg["squeeze_bars"]
        window = n * 2
        if len(bw) < window + 2:
            return self._hold(symbol, "not enough bars for squeeze detection")

        # Any consecutive run of n narrowing bars in the recent window
        bw_window = [float(bw.iloc[-(window - i)]) for i in range(window)]
        squeeze = False
        for start in range(len(bw_window) - n):
            if all(bw_window[start + j + 1] < bw_window[start + j] for j in range(n)):
                squeeze = True
                break

        rsi14   = _rsi(close, cfg["rsi_period"])
        rsi_now = float(rsi14.iloc[-1])
        upper_now = float(bb_upper.iloc[-1])
        lower_now = float(bb_lower.iloc[-1])
        bw_now    = upper_now - lower_now

        vol_ratio = 1.0
        if "Volume" in df.columns and len(df) > 21:
            avg_vol   = float(df["Volume"].iloc[-21:-1].mean())
            cur_vol   = float(df["Volume"].iloc[-1])
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        # ── BUY ──
        if (squeeze
                and c_now > upper_now
                and rsi_now >= cfg["rsi_entry_min"]
                and vol_ratio >= cfg["vol_ratio_min"]):
            stop   = lower_now
            target = c_now + bw_now   # project one bandwidth above entry
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(c_now, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.78,
                reason=f"BB squeeze ({n} bars) + close above upper BB, RSI={rsi_now:.0f}, vol={vol_ratio:.1f}×",
                indicators={
                    "bb_upper": round(upper_now, 2),
                    "bb_lower": round(lower_now, 2),
                    "bb_width": round(bw_now, 2),
                    "rsi":      round(rsi_now, 1),
                    "vol_ratio": round(vol_ratio, 2),
                },
            )

        # ── SELL ──
        if rsi_now > cfg["rsi_overbought"]:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=c_now, confidence=0.75,
                reason=f"RSI={rsi_now:.0f} > {cfg['rsi_overbought']} — overbought after breakout",
                indicators={"rsi": round(rsi_now, 1), "bb_upper": round(upper_now, 2)},
            )

        return self._hold(symbol)


# ══════════════════════════════════════════════════════════════
# STRATEGY 4 — Pullback to Rising EMA(50) + Rejection Wick
# ══════════════════════════════════════════════════════════════
class BollingerMeanReversionUptrend(PerplexityStrategy):
    """
    Highest-frequency strategy: buy dips to the rising EMA(50) in BULL or mild BEAR.
    Requires a bullish rejection wick and RSI in recovery zone.

    BUY : EMA(50) rising (slope > 0) + price within ema_proximity_pct of EMA(50)
          + RSI in [rsi_min, rsi_max] + lower wick ≥ wick_ratio_min × bar range
          + NOT in deep bear (SPY > 10% below SMA200 threshold).
    SELL: RSI > exit_rsi OR close > entry × (1 + exit_extension_pct/100).
    Stop : hard_stop_pct % below entry.
    """
    name = "BB_Mean_Reversion"

    config: dict = {
        "bb_period":      20,      # kept for UI compatibility
        "bb_std":         2.0,     # kept for UI compatibility
        "reentry_bars":   5,       # kept for UI compatibility
        "atr_multiple":   2.0,     # kept for UI compatibility
        "rsi_re_entry":   38,      # kept for UI compatibility
        "use_rsi_filter": True,    # kept for UI compatibility
        "rsi_exit":       74,      # kept for UI compatibility
        "min_data_bars":  220,
        # ── New logic params ──
        "ema_trend":                50,
        "ema_slope_bars":           5,    # bars to measure EMA slope (must be positive)
        "price_ema_proximity_pct":  3.0,  # price within ±3% of EMA50
        "rsi_period":               14,
        "rsi_min":                  30,
        "rsi_max":                  60,
        "wick_ratio_min":           0.3,  # lower wick ≥ 30% of bar range
        "exit_rsi":                 65,
        "exit_extension_pct":       3.0,  # sell if price > entry + 3%
        "hard_stop_pct":            2.0,
        "max_hold_bars":            20,
        "bear_skip_threshold_pct":  10.0, # skip if price > 10% below SMA200
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
        if regime == MarketRegime.DEEP_BEAR:
            return self._hold(symbol, "OFF in Deep Bear")

        close = df["Close"]
        c_now = float(close.iloc[-1])

        # Deep bear guard: skip if price far below SMA200
        sma200 = float(_sma(close, 200).iloc[-1])
        if c_now < sma200 * (1 - cfg["bear_skip_threshold_pct"] / 100):
            return self._hold(symbol, f"price >{cfg['bear_skip_threshold_pct']}% below SMA200")

        ema50 = _ema(close, cfg["ema_trend"])
        ema50_now  = float(ema50.iloc[-1])
        ema50_prev = float(ema50.iloc[-cfg["ema_slope_bars"]])

        if ema50_now <= ema50_prev:
            return self._hold(symbol, "EMA(50) not rising — no uptrend")

        prox_pct = abs(c_now - ema50_now) / ema50_now * 100
        if prox_pct > cfg["price_ema_proximity_pct"]:
            return self._hold(symbol, f"price {prox_pct:.1f}% from EMA(50) — not a clean pullback")

        rsi14   = _rsi(close, cfg["rsi_period"])
        rsi_now = float(rsi14.iloc[-1])

        if not (cfg["rsi_min"] <= rsi_now <= cfg["rsi_max"]):
            return self._hold(symbol, f"RSI={rsi_now:.0f} outside [{cfg['rsi_min']},{cfg['rsi_max']}]")

        o_now = float(df["Open"].iloc[-1])
        h_now = float(df["High"].iloc[-1])
        l_now = float(df["Low"].iloc[-1])
        bar_range  = h_now - l_now
        lower_wick = min(o_now, c_now) - l_now
        wick_ratio = lower_wick / bar_range if bar_range > 0 else 0.0

        if wick_ratio < cfg["wick_ratio_min"]:
            return self._hold(symbol, f"wick ratio {wick_ratio:.2f} < {cfg['wick_ratio_min']} — no rejection")

        # ── BUY ──
        stop   = c_now * (1 - cfg["hard_stop_pct"] / 100)
        target = c_now * (1 + cfg["exit_extension_pct"] / 100)
        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(c_now, 2),
            stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=0.72,
            reason=f"Pullback to rising EMA(50), RSI={rsi_now:.0f}, wick={wick_ratio:.0%}",
            indicators={
                "ema50":      round(ema50_now, 2),
                "prox_pct":   round(prox_pct, 2),
                "rsi":        round(rsi_now, 1),
                "wick_ratio": round(wick_ratio, 2),
                "sma200":     round(sma200, 2),
            },
        )


# ══════════════════════════════════════════════════════════════
# STRATEGY 5 — ATR-Spike / Fear-Capitulation Reversal
# ══════════════════════════════════════════════════════════════
class FibPullbackSupport(PerplexityStrategy):
    """
    Buy fear/capitulation spikes: ATR% proxy signals extreme intraday panic
    followed by reversal signals. Works in BULL and mild BEAR.

    BUY : ATR%_today > atr_spike_threshold + RSI(14) < rsi_entry_max
          + BB%B < bb_pos_max + lower wick ≥ wick_ratio_min × bar range
          + 3-bar prior decline ≥ prior_decline_pct.
    SELL: RSI > rsi_exit OR ATR% < atr_exit_threshold (volatility normalized).
    Stop : hard_stop_pct % below entry.
    Target: take_profit_pct % above entry.
    """
    name = "Fib_Pullback_Support"

    config: dict = {
        "swing_lookback":   60,       # kept for UI compatibility
        "fib_levels":       [0.382, 0.50, 0.618],  # kept for UI compatibility
        "fib_zone_pct":     1.2,      # kept for UI compatibility
        "rsi_oversold_low": 35,       # kept for UI compatibility
        "rsi_oversold_hi":  62,       # kept for UI compatibility
        "atr_stop_mult":    1.5,      # kept for UI compatibility
        "min_data_bars":    220,
        # ── New logic params ──
        "atr_period":           14,
        "atr_spike_threshold":  1.5,   # ATR% > 1.5% = fear/elevated volatility
        "atr_exit_threshold":   1.0,   # ATR% < 1% = volatility normalized → exit
        "rsi_period":           14,
        "rsi_entry_max":        45,    # RSI < 45 = oversold/weak momentum
        "rsi_exit":             60,    # RSI recovers → exit
        "bb_pos_max":           0.35,  # BB%B < 0.35 = below midline
        "wick_ratio_min":       0.30,  # rejection wick ≥ 30% of bar range
        "prior_decline_pct":    1.0,   # prior 3-bar decline ≥ 1%
        "prior_decline_bars":   3,
        "hard_stop_pct":        4.0,
        "take_profit_pct":      6.0,
        "max_hold_bars":        8,
        # ── Calibration filters ──
        "filter_lower_wick_min": 0.0,
        "filter_vol_min":        0.0,
    }

    def run(self, symbol: str, df: pd.DataFrame, regime: MarketRegime | None = None) -> PerplexitySignal:
        cfg = self.config
        if len(df) < cfg["min_data_bars"]:
            return self._hold(symbol, "not enough data")

        regime = regime or MarketRegime.BULL
        if regime == MarketRegime.DEEP_BEAR:
            return self._hold(symbol, "OFF in Deep Bear")

        close = df["Close"]
        c_now = float(close.iloc[-1])
        atr_v = _current_atr(df, cfg["atr_period"])
        atr_pct = atr_v / c_now * 100 if c_now > 0 else 0.0

        # ATR% spike: today must be a fear day
        if atr_pct <= cfg["atr_spike_threshold"]:
            return self._hold(symbol, f"ATR%={atr_pct:.1f} — no fear spike (need >{cfg['atr_spike_threshold']}%)")

        rsi14   = _rsi(close, cfg["rsi_period"])
        rsi_now = float(rsi14.iloc[-1])

        if rsi_now >= cfg["rsi_entry_max"]:
            return self._hold(symbol, f"RSI={rsi_now:.0f} ≥ {cfg['rsi_entry_max']} — not oversold enough")

        # BB%B position
        bb_upper, bb_mid, bb_lower = _bb_bands(close, 20, 2.0)
        bw = float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
        bb_pos = (c_now - float(bb_lower.iloc[-1])) / bw if bw > 0 else 0.5
        if bb_pos > cfg["bb_pos_max"]:
            return self._hold(symbol, f"BB%B={bb_pos:.2f} — not near lower band")

        # Rejection wick
        o_now = float(df["Open"].iloc[-1])
        h_now = float(df["High"].iloc[-1])
        l_now = float(df["Low"].iloc[-1])
        bar_range  = h_now - l_now
        lower_wick = min(o_now, c_now) - l_now
        wick_ratio = lower_wick / bar_range if bar_range > 0 else 0.0
        if wick_ratio < cfg["wick_ratio_min"]:
            return self._hold(symbol, f"wick ratio {wick_ratio:.2f} < {cfg['wick_ratio_min']} — no reversal candle")

        # Prior decline guard (must have sold off to create panic)
        n_dec = cfg["prior_decline_bars"]
        if len(close) > n_dec:
            prior_close = float(close.iloc[-(n_dec + 1)])
            decline_pct = (prior_close - c_now) / prior_close * 100
            if decline_pct < cfg["prior_decline_pct"]:
                return self._hold(symbol, f"prior {n_dec}-bar decline {decline_pct:.1f}% < {cfg['prior_decline_pct']}%")

        # ── BUY ──
        stop   = c_now * (1 - cfg["hard_stop_pct"] / 100)
        target = c_now * (1 + cfg["take_profit_pct"] / 100)
        return PerplexitySignal(
            symbol=symbol, strategy_name=self.name, direction="BUY",
            entry_price=round(c_now, 2),
            stop_price=round(stop, 2),
            target_price=round(target, 2),
            confidence=0.80,
            reason=f"Fear spike: ATR%={atr_pct:.1f}%, RSI={rsi_now:.0f}, BB%B={bb_pos:.2f}, wick={wick_ratio:.0%}",
            indicators={
                "atr_pct":   round(atr_pct, 2),
                "rsi":       round(rsi_now, 1),
                "bb_pos":    round(bb_pos, 3),
                "wick_ratio": round(wick_ratio, 2),
                "bb_lower":  round(float(bb_lower.iloc[-1]), 2),
            },
        )
