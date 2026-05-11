from __future__ import annotations

"""
5 Perplexity swing trading strategies.
All require close > SMA(200) as trend filter (long-side only).
Designed for daily bars, 3-10 day holds.
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


# ─────────────────────────────────────────────
# STRATEGY 1 — High Volume Momentum Breakout
# ─────────────────────────────────────────────
class HighVolumeMomentumBreakout(PerplexityStrategy):
    """
    BUY : Price breaks above 10-day high with volume > 1.3x average AND RSI 45-78.
          Trend filter: close > SMA(200) AND EMA(20) > EMA(50).
    SELL: Price closes below EMA(20) OR RSI > 78.
    Stop: 2x ATR below entry. Target: entry + 3x ATR (3:1 R:R).

    Rationale: High-volume breakouts above recent highs signal institutional
    buying. Using 10-day high (vs 20-day) generates more actionable signals
    while the volume filter removes low-conviction moves.
    """
    name = "High_Volume_Momentum_Breakout"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 55:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close  = df["Close"]
        high   = df["High"]
        atr    = compute_atr(df["High"], df["Low"], df["Close"]).values
        rsi    = compute_rsi(close, 14).values
        ema20  = compute_ema(close, 20).values
        ema50  = compute_ema(close, 50).values

        current_close = float(close.iloc[-1])
        ema20_now     = float(ema20.iloc[-1])
        ema50_now     = float(ema50.iloc[-1])
        rsi_now       = float(rsi.iloc[-1])
        atr_val       = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_close * 0.02

        # Require EMA20 > EMA50 — only trade confirmed uptrends
        if ema20_now < ema50_now:
            return self._hold(symbol, "EMA20 below EMA50")

        # 10-day high (excluding today) — generates more signals than 20-day
        high_10 = float(high.iloc[-11:-1].max())

        # Volume check: 1.3x average (vs 1.5x) — more balanced threshold
        vol_surge = True
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol = float(vol.iloc[-21:-1].mean())
            vol_surge = float(vol.iloc[-1]) > 1.3 * avg_vol if avg_vol > 0 else True

        # Sell: close below EMA20 OR RSI overextended
        if current_close < ema20_now or rsi_now > 78:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="below EMA20" if current_close < ema20_now else "RSI > 78",
                indicators={"ema20": round(ema20_now, 2), "rsi": round(rsi_now, 1)},
            )

        # Buy: breakout above 10-day high with volume + RSI in momentum zone
        breakout = current_close > high_10
        rsi_momentum = 45 < rsi_now < 78

        if breakout and vol_surge and rsi_momentum:
            stop   = current_close - 2 * atr_val
            target = current_close + 3 * atr_val  # 3:1 R:R
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

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 2 — Bollinger Squeeze Breakout
# ─────────────────────────────────────────────
class BollingerSqueezeBreakout(PerplexityStrategy):
    """
    BUY : BB bandwidth is below its 50-bar average (squeeze) AND price closes above
          upper band with RSI > 50 — volatility expanding after compression.
    SELL: BB middle touched OR RSI > 75.
    Stop: BB middle. Target: entry + (entry - BB middle) for 1:1 R:R above upper.
    """
    name = "Bollinger_Squeeze_Breakout"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 60:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close   = df["Close"]
        bb      = compute_bollinger(close, 20, 2.0)
        bw      = bb.bandwidth.values
        rsi     = compute_rsi(close, 14).values
        current_close = float(close.iloc[-1])

        # Squeeze: current bandwidth below 50-bar moving average of bandwidth
        avg_bw_50 = float(bw.rolling(50).mean().iloc[-1])
        cur_bw    = float(bw.iloc[-1])
        # Also check prev bar was in squeeze (not the exact moment of expansion)
        prev_bw   = float(bw.iloc[-2])

        bb_upper  = float(bb.upper.values.iloc[-1])
        bb_middle = float(bb.middle.values.iloc[-1])
        rsi_now   = float(rsi.iloc[-1])

        # Volume surge (optional bonus — not required to avoid over-filtering)
        vol_above_avg = True
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol = float(vol.rolling(20).mean().iloc[-1])
            vol_above_avg = float(vol.iloc[-1]) > avg_vol if avg_vol > 0 else True

        # Sell: price pulled back to BB middle OR RSI overbought
        if current_close <= bb_middle:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.75,
                reason="price at BB middle — trail stop hit",
                indicators={"bb_middle": round(bb_middle, 2), "rsi": round(rsi_now, 1)},
            )
        if rsi_now > 75:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.8,
                reason="RSI overbought exit",
                indicators={"bb_bandwidth": round(cur_bw, 4), "rsi": round(rsi_now, 1)},
            )

        # Buy: squeeze (bandwidth below 50-bar avg) AND price above BB upper AND RSI > 50
        squeezed = cur_bw < avg_bw_50 or prev_bw < avg_bw_50
        breakout = current_close > bb_upper
        rsi_ok   = rsi_now > 50

        if squeezed and breakout and rsi_ok and vol_above_avg:
            atr = compute_atr(df["High"], df["Low"], df["Close"]).values
            atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_close * 0.02
            stop   = bb_middle
            risk   = current_close - stop
            target = current_close + risk  # 1:1 above breakout point
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=0.80,
                reason=f"BB squeeze breakout — bw below 50-bar avg then upper band break (RSI {rsi_now:.0f})",
                indicators={"bb_upper": round(bb_upper, 2),
                            "bb_middle": round(bb_middle, 2),
                            "bb_bandwidth": round(cur_bw, 4),
                            "avg_bw_50": round(avg_bw_50, 4),
                            "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 3 — MACD RSI Momentum
# ─────────────────────────────────────────────
class MacdRsiMomentum(PerplexityStrategy):
    """
    BUY : MACD line > signal AND histogram > 0 AND RSI > 45.
    SELL: MACD line crosses below signal OR RSI > 75.
    """
    name = "MACD_RSI_Momentum"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close = df["Close"]
        macd  = compute_macd(close, 12, 26, 9)
        rsi   = compute_rsi(close, 14).values
        atr   = compute_atr(df["High"], df["Low"], df["Close"]).values

        macd_line = macd.macd.values
        sig_line  = macd.signal.values
        hist      = macd.histogram.values

        current_close = float(close.iloc[-1])
        rsi_now       = float(rsi.iloc[-1])
        atr_val       = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_close * 0.02

        macd_now  = float(macd_line.iloc[-1])
        sig_now   = float(sig_line.iloc[-1])
        hist_now  = float(hist.iloc[-1])
        macd_prev = float(macd_line.iloc[-2])
        sig_prev  = float(sig_line.iloc[-2])

        # Sell: MACD crosses below signal OR RSI overbought (tightened from 75→70)
        if (macd_prev >= sig_prev and macd_now < sig_now) or rsi_now > 70:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="MACD crossed below signal" if macd_now < sig_now else "RSI > 70",
                indicators={"macd": round(macd_now, 4), "signal": round(sig_now, 4),
                            "rsi": round(rsi_now, 1)},
            )

        # Buy: MACD bullish + RSI confirmed momentum (tightened from 45→50)
        # Also require histogram rising (accelerating momentum)
        hist_prev = float(hist.iloc[-2])
        hist_rising = hist_now > hist_prev
        if macd_now > sig_now and hist_now > 0 and rsi_now > 50 and hist_rising:
            stop   = current_close - 1.5 * atr_val
            target = current_close + 3 * 1.5 * atr_val  # widened to 3:1 reward:risk
            confidence = min(0.92, 0.60 + (rsi_now - 50) / 100)
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=round(confidence, 2),
                reason="MACD momentum + RSI > 50 + histogram rising",
                indicators={"macd": round(macd_now, 4), "signal": round(sig_now, 4),
                            "histogram": round(hist_now, 4), "rsi": round(rsi_now, 1),
                            "atr": round(atr_val, 2)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 4 — EMA Pullback Support
# ─────────────────────────────────────────────
class EmaPullbackSupport(PerplexityStrategy):
    """
    BUY : Price pulls back to EMA(20) then closes above it (bullish candle).
    SELL: 2x risk reached OR price touches BB upper.
    """
    name = "EMA_Pullback_Support"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close  = df["Close"]
        low    = df["Low"]
        ema20  = compute_ema(close, 20).values
        ema50  = compute_ema(close, 50).values
        rsi    = compute_rsi(close, 14).values
        bb     = compute_bollinger(close, 20, 2.0)
        bb_upper = float(bb.upper.values.iloc[-1])

        current_close = float(close.iloc[-1])
        current_low   = float(low.iloc[-1])
        ema20_now     = float(ema20.iloc[-1])
        ema20_prev    = float(ema20.iloc[-2])
        ema50_now     = float(ema50.iloc[-1])
        close_prev    = float(close.iloc[-2])
        rsi_now       = float(rsi.iloc[-1])

        # Trend confirmation: EMA20 must be above EMA50 (confirmed uptrend)
        if ema20_now < ema50_now:
            return self._hold(symbol, "EMA20 below EMA50 — no uptrend")

        # Sell: price at BB upper OR RSI overbought
        if current_close >= bb_upper or rsi_now > 70:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="BB upper reached" if current_close >= bb_upper else "RSI > 70",
                indicators={"ema20": round(ema20_now, 2), "bb_upper": round(bb_upper, 2),
                            "rsi": round(rsi_now, 1)},
            )

        # Sell: two consecutive closes below EMA20 (confirmed break)
        if current_close < ema20_now and close_prev < ema20_prev:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.75,
                reason="two closes below EMA20 — stop",
                indicators={"ema20": round(ema20_now, 2), "rsi": round(rsi_now, 1)},
            )

        # Buy: low touched EMA20 zone (within 0.5%), closed above, bullish candle,
        #      RSI in healthy range (not overbought entry)
        touched_ema    = current_low <= ema20_now * 1.005
        closed_above   = current_close > ema20_now
        bullish_candle = current_close > float(df["Open"].iloc[-1])
        rsi_healthy    = 40 < rsi_now < 65

        if touched_ema and closed_above and bullish_candle and rsi_healthy:
            risk   = current_close - ema20_now
            stop   = ema20_now
            target = min(current_close + 2 * risk, bb_upper)
            confidence = 0.75 if rsi_now < 55 else 0.65
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=confidence,
                reason="bullish bounce off EMA20 in uptrend (RSI confirmed)",
                indicators={"ema20": round(ema20_now, 2), "ema50": round(ema50_now, 2),
                            "bb_upper": round(bb_upper, 2), "rsi": round(rsi_now, 1)},
            )

        return self._hold(symbol)


# ─────────────────────────────────────────────
# STRATEGY 5 — Bollinger Reversion Uptrend
# ─────────────────────────────────────────────
class BollingerReversionUptrend(PerplexityStrategy):
    """
    BUY : Price closes below BB lower, then next bar closes back above BB lower.
    SELL: 50% at BB middle, 100% at BB upper OR RSI > 70.
    """
    name = "Bollinger_Reversion_Uptrend"

    def run(self, symbol: str, df: pd.DataFrame) -> PerplexitySignal:
        if len(df) < 40:
            return self._hold(symbol, "not enough data")
        if not _sma200_filter(df):
            return self._hold(symbol, "below SMA200")

        close  = df["Close"]
        bb     = compute_bollinger(close, 20, 2.0)
        rsi    = compute_rsi(close, 14).values
        atr    = compute_atr(df["High"], df["Low"], df["Close"]).values

        bb_lower  = bb.lower.values
        bb_middle = float(bb.middle.values.iloc[-1])
        bb_upper  = float(bb.upper.values.iloc[-1])

        current_close = float(close.iloc[-1])
        prev_close    = float(close.iloc[-2])
        lower_now     = float(bb_lower.iloc[-1])
        lower_prev    = float(bb_lower.iloc[-2])
        rsi_now       = float(rsi.iloc[-1])
        atr_val       = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_close * 0.02

        # Sell at BB upper OR RSI overbought
        if current_close >= bb_upper or rsi_now > 70:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.80,
                reason="BB upper reached" if current_close >= bb_upper else "RSI > 70",
                indicators={"bb_upper": round(bb_upper, 2),
                            "bb_middle": round(bb_middle, 2),
                            "rsi": round(rsi_now, 1)},
            )

        # Partial exit signal at BB middle (shown as SELL with lower confidence)
        if current_close >= bb_middle and prev_close < bb_middle:
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=current_close, confidence=0.50,
                reason="BB middle reached — partial exit (50%)",
                indicators={"bb_middle": round(bb_middle, 2), "rsi": round(rsi_now, 1)},
            )

        # Buy: prev bar closed below lower, current bar closed back above lower
        reverted = (prev_close < lower_prev) and (current_close > lower_now)
        if reverted:
            stop   = current_close - 2 * atr_val
            target = bb_upper
            confidence = 0.75 if rsi_now < 45 else 0.60
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=round(current_close, 2),
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                confidence=confidence,
                reason="BB lower reversion — price closed back above lower band",
                indicators={"bb_lower": round(lower_now, 2),
                            "bb_middle": round(bb_middle, 2),
                            "bb_upper": round(bb_upper, 2),
                            "rsi": round(rsi_now, 1),
                            "atr": round(atr_val, 2)},
            )

        return self._hold(symbol)
