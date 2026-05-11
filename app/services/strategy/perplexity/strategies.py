from __future__ import annotations

"""
5 Perplexity swing trading strategies.
All require close > SMA(200) as trend filter (long-side only).
Designed for daily bars, 3-10 day holds.

Fixes applied vs prior version:
- BUY logic always evaluated before SELL on the same bar (sell only fires when in a position)
- Strategy 2: ATR-based stop replaces BB-middle stop (was too tight for position sizer)
- Strategy 2: bandwidth NaN guard added; min data raised to 75 bars
- Strategy 3: RSI sell threshold raised to 75 (was 70, conflicted with RSI>50 buy)
- Strategy 4: stop set to EMA20 - 1*ATR instead of exactly EMA20 (was too tight)
              risk/target calculated from ATR-based stop, not the tiny EMA gap
              RSI ceiling raised from 65 to 70
- Strategy 5: BB-middle partial exit removed (engine can't partial-exit; was cutting winners short)
              RSI > 30 filter added to avoid buying falling knives
- Bollinger std changed to ddof=0 (population) to match TradingView / standard charting
"""

import pandas as pd

from app.services.indicators.atr import compute_atr
from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma
from app.services.indicators.ema import compute_ema
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


def _sma200_filter(df: pd.DataFrame) -> bool:
    """True when close is above the 200-day SMA (uptrend filter)."""
    if len(df) < 200:
        return False
    sma200 = df["Close"].rolling(200).mean().iloc[-1]
    return float(df["Close"].iloc[-1]) > float(sma200)


def _atr(df: pd.DataFrame) -> float:
    """Return current ATR(14), fallback to 2% of price if insufficient data."""
    val = compute_atr(df["High"], df["Low"], df["Close"]).values
    v = float(val.iloc[-1])
    return v if not pd.isna(v) else float(df["Close"].iloc[-1]) * 0.02


# ─────────────────────────────────────────────
# STRATEGY 1 — High Volume Momentum Breakout
# ─────────────────────────────────────────────
class HighVolumeMomentumBreakout(PerplexityStrategy):
    """
    BUY : Price breaks above 10-day high with volume > 1.3x average AND RSI 45-78.
          Trend filter: close > SMA(200) AND EMA(20) > EMA(50).
    SELL: Price closes below EMA(20) OR RSI > 78.
    Stop: 2x ATR below entry. Target: entry + 3x ATR (3:1 R:R).
    """
    name = "High_Volume_Momentum_Breakout"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 55:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close = df["Close"]
        high  = df["High"]
        atr_val   = _atr(df)
        rsi_now   = float(compute_rsi(close, 14).values.iloc[-1])
        ema20_now = float(compute_ema(close, 20).values.iloc[-1])
        ema50_now = float(compute_ema(close, 50).values.iloc[-1])
        current_close = float(close.iloc[-1])

        if ema20_now < ema50_now:
            return self._hold(symbol, "EMA20 below EMA50")

        high_10 = float(high.iloc[-11:-1].max())

        vol_surge = True
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol = float(vol.iloc[-21:-1].mean())
            vol_surge = float(vol.iloc[-1]) > 1.3 * avg_vol if avg_vol > 0 else True

        # ── BUY evaluated first ──────────────────────────────────
        breakout     = current_close > high_10
        rsi_momentum = 45 < rsi_now < 78

        if breakout and vol_surge and rsi_momentum:
            stop   = current_close - 2 * atr_val
            target = current_close + 3 * atr_val
            if (current_close - stop) / current_close < 0.001:
                return self._hold(symbol, "stop too tight after breakout")
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.78,
                reason=f"10-day high breakout with volume surge (RSI {rsi_now:.0f})",
                indicators={"high_10": round(high_10, 2), "ema20": round(ema20_now, 2),
                            "rsi": round(rsi_now, 1), "atr": round(atr_val, 2)},
            )

        # ── SELL only when conditions clearly say exit ───────────
        if current_close < ema20_now or rsi_now > 78:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="below EMA20" if current_close < ema20_now else "RSI > 78",
                indicators={"ema20": round(ema20_now, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 2 — Bollinger Squeeze Breakout
# ─────────────────────────────────────────────
class BollingerSqueezeBreakout(PerplexityStrategy):
    """
    BUY : BB bandwidth below its 50-bar average (squeeze) AND price closes above
          upper band with RSI > 50 — volatility expanding after compression.
    SELL: close drops below BB middle OR RSI > 75.
    Stop: 1.5x ATR below entry (replaces BB-middle stop which was too tight).
    Target: entry + 2.5x ATR (roughly upper-band to next resistance).
    """
    name = "Bollinger_Squeeze_Breakout"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 75:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close = df["Close"]
        # ddof=0 = population std, matches TradingView standard
        bb    = compute_bollinger(close, 20, 2.0)
        bw    = bb.bandwidth.values
        rsi   = compute_rsi(close, 14).values
        atr_val       = _atr(df)
        current_close = float(close.iloc[-1])

        avg_bw_50 = float(bw.rolling(50).mean().iloc[-1])
        cur_bw    = float(bw.iloc[-1])
        prev_bw   = float(bw.iloc[-2])

        # Guard: if rolling 50-bar avg hasn't warmed up yet, skip
        if pd.isna(avg_bw_50):
            return self._hold(symbol, "bandwidth avg not ready")

        bb_upper  = float(bb.upper.values.iloc[-1])
        bb_middle = float(bb.middle.values.iloc[-1])
        rsi_now   = float(rsi.iloc[-1])

        vol_above_avg = True
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol = float(vol.rolling(20).mean().iloc[-1])
            vol_above_avg = float(vol.iloc[-1]) > avg_vol if avg_vol > 0 else True

        # ── BUY evaluated first ──────────────────────────────────
        squeezed = cur_bw < avg_bw_50 or prev_bw < avg_bw_50
        breakout = current_close > bb_upper
        rsi_ok   = rsi_now > 50

        if squeezed and breakout and rsi_ok and vol_above_avg:
            stop   = current_close - 1.5 * atr_val   # ATR-based, not BB middle
            target = current_close + 2.5 * atr_val
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.80,
                reason=f"BB squeeze breakout above upper band (RSI {rsi_now:.0f})",
                indicators={"bb_upper": round(bb_upper, 2),
                            "bb_middle": round(bb_middle, 2),
                            "bb_bandwidth": round(cur_bw, 4),
                            "avg_bw_50": round(avg_bw_50, 4),
                            "rsi": round(rsi_now, 1), "atr": round(atr_val, 2)},
            )

        # ── SELL ─────────────────────────────────────────────────
        if current_close <= bb_middle or rsi_now > 75:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.75,
                reason="price at BB middle" if current_close <= bb_middle else "RSI > 75",
                indicators={"bb_middle": round(bb_middle, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 3 — MACD RSI Momentum
# ─────────────────────────────────────────────
class MacdRsiMomentum(PerplexityStrategy):
    """
    BUY : MACD line > signal AND histogram rising AND RSI 50-75.
    SELL: MACD crosses below signal OR RSI > 75.
    Stop: 1.5x ATR. Target: 4.5x ATR (3:1 R:R).
    """
    name = "MACD_RSI_Momentum"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close   = df["Close"]
        macd    = compute_macd(close, 12, 26, 9)
        rsi     = compute_rsi(close, 14).values
        atr_val = _atr(df)

        macd_line = macd.macd.values
        sig_line  = macd.signal.values
        hist      = macd.histogram.values

        current_close = float(close.iloc[-1])
        rsi_now   = float(rsi.iloc[-1])
        macd_now  = float(macd_line.iloc[-1])
        sig_now   = float(sig_line.iloc[-1])
        hist_now  = float(hist.iloc[-1])
        macd_prev = float(macd_line.iloc[-2])
        sig_prev  = float(sig_line.iloc[-2])
        hist_prev = float(hist.iloc[-2])

        hist_rising = hist_now > hist_prev

        # ── BUY evaluated first ──────────────────────────────────
        # RSI ceiling raised to 75 to match SELL threshold — no conflict
        if macd_now > sig_now and hist_now > 0 and 50 < rsi_now < 75 and hist_rising:
            stop   = current_close - 1.5 * atr_val
            target = current_close + 4.5 * atr_val  # 3:1 R:R
            confidence = min(0.92, 0.60 + (rsi_now - 50) / 100)
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(confidence, 2),
                reason="MACD momentum + RSI confirmed + histogram rising",
                indicators={"macd": round(macd_now, 4), "signal": round(sig_now, 4),
                            "histogram": round(hist_now, 4), "rsi": round(rsi_now, 1),
                            "atr": round(atr_val, 2)},
            )

        # ── SELL ─────────────────────────────────────────────────
        macd_crossed_below = macd_prev >= sig_prev and macd_now < sig_now
        if macd_crossed_below or rsi_now > 75:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="MACD crossed below signal" if macd_crossed_below else "RSI > 75",
                indicators={"macd": round(macd_now, 4), "signal": round(sig_now, 4),
                            "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 4 — EMA Pullback Support
# ─────────────────────────────────────────────
class EmaPullbackSupport(PerplexityStrategy):
    """
    BUY : Price pulls back to EMA(20) then closes above it (bullish candle).
          RSI 40-70, uptrend confirmed (EMA20 > EMA50, close > SMA200).
    SELL: Close >= BB upper OR RSI > 72 OR two consecutive closes below EMA20.
    Stop: EMA20 - 1x ATR (buffer below EMA to avoid noise stop-outs).
    Target: entry + 2x risk, capped at BB upper.
    """
    name = "EMA_Pullback_Support"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close   = df["Close"]
        low     = df["Low"]
        atr_val = _atr(df)
        ema20   = compute_ema(close, 20).values
        ema50   = compute_ema(close, 50).values
        rsi     = compute_rsi(close, 14).values
        bb      = compute_bollinger(close, 20, 2.0)
        bb_upper = float(bb.upper.values.iloc[-1])

        current_close = float(close.iloc[-1])
        current_low   = float(low.iloc[-1])
        ema20_now     = float(ema20.iloc[-1])
        ema20_prev    = float(ema20.iloc[-2])
        ema50_now     = float(ema50.iloc[-1])
        close_prev    = float(close.iloc[-2])
        rsi_now       = float(rsi.iloc[-1])

        if ema20_now < ema50_now:
            return self._hold(symbol, "EMA20 below EMA50 — no uptrend")

        # ── BUY evaluated first ──────────────────────────────────
        touched_ema    = current_low <= ema20_now * 1.005
        closed_above   = current_close > ema20_now
        bullish_candle = current_close > float(df["Open"].iloc[-1])
        # Raised RSI ceiling from 65 → 70 to capture more valid pullbacks
        rsi_healthy    = 40 < rsi_now < 70

        if touched_ema and closed_above and bullish_candle and rsi_healthy:
            # Stop below EMA20 by 1 ATR — gives room for normal noise
            stop   = ema20_now - atr_val
            risk   = current_close - stop          # realistic risk from entry to stop
            target = min(current_close + 2 * risk, bb_upper)
            # Ensure meaningful R:R — skip if target is too close
            if risk < atr_val * 0.3:
                return self._hold(symbol, "risk too small for meaningful trade")
            confidence = 0.75 if rsi_now < 55 else 0.65
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=confidence,
                reason=f"bullish bounce off EMA20 in uptrend (RSI {rsi_now:.0f})",
                indicators={"ema20": round(ema20_now, 2), "ema50": round(ema50_now, 2),
                            "bb_upper": round(bb_upper, 2), "rsi": round(rsi_now, 1),
                            "atr": round(atr_val, 2), "stop": round(stop, 2)},
            )

        # ── SELL ─────────────────────────────────────────────────
        if current_close >= bb_upper or rsi_now > 72:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="BB upper reached" if current_close >= bb_upper else "RSI > 72",
                indicators={"ema20": round(ema20_now, 2), "bb_upper": round(bb_upper, 2),
                            "rsi": round(rsi_now, 1)},
            )

        if current_close < ema20_now and close_prev < ema20_prev:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.75,
                reason="two closes below EMA20 — stop",
                indicators={"ema20": round(ema20_now, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 5 — Bollinger Reversion Uptrend
# ─────────────────────────────────────────────
class BollingerReversionUptrend(PerplexityStrategy):
    """
    BUY : Price closes below BB lower, then next bar closes back above BB lower.
          RSI > 30 (not in freefall). Uptrend: close > SMA200.
    SELL: Close >= BB upper OR RSI > 72.
    Stop: 2x ATR below entry. Target: BB upper (full mean-reversion target).
    """
    name = "Bollinger_Reversion_Uptrend"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close   = df["Close"]
        bb      = compute_bollinger(close, 20, 2.0)
        rsi     = compute_rsi(close, 14).values
        atr_val = _atr(df)

        bb_lower  = bb.lower.values
        bb_middle = float(bb.middle.values.iloc[-1])
        bb_upper  = float(bb.upper.values.iloc[-1])

        current_close = float(close.iloc[-1])
        prev_close    = float(close.iloc[-2])
        lower_now     = float(bb_lower.iloc[-1])
        lower_prev    = float(bb_lower.iloc[-2])
        rsi_now       = float(rsi.iloc[-1])

        # ── BUY evaluated first ──────────────────────────────────
        reverted = (prev_close < lower_prev) and (current_close > lower_now)
        # RSI > 30 ensures stock is not in a free-fall (avoids catching falling knives)
        rsi_not_crashing = rsi_now > 30

        if reverted and rsi_not_crashing:
            stop   = current_close - 2 * atr_val
            target = bb_upper   # full mean-reversion target
            confidence = 0.75 if rsi_now < 45 else 0.60
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=confidence,
                reason=f"BB lower reversion — price closed back above lower band (RSI {rsi_now:.0f})",
                indicators={"bb_lower": round(lower_now, 2),
                            "bb_middle": round(bb_middle, 2),
                            "bb_upper": round(bb_upper, 2),
                            "rsi": round(rsi_now, 1),
                            "atr": round(atr_val, 2)},
            )

        # ── SELL ─────────────────────────────────────────────────
        # Removed BB-middle partial exit — backtest engine can't partial-exit,
        # so exiting at middle was cutting winners short before reaching BB upper.
        if current_close >= bb_upper or rsi_now > 72:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="BB upper reached" if current_close >= bb_upper else "RSI > 72",
                indicators={"bb_upper": round(bb_upper, 2),
                            "bb_middle": round(bb_middle, 2),
                            "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)
