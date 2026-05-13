from __future__ import annotations

"""
Strategy 2 — EMA Crossover + MACD Confirmation (Trend Following)

Three-filter system: trend (EMA 9/21 crossover) + momentum (MACD above signal)
+ timing (RSI 45–65) + volume confirmation.

BULL behaviour : all entry conditions active.
BEAR behaviour : strategy is OFF.

Typical frequency (BULL market): 8–14 signals/year per symbol.
Max hold: 20 bars.
"""

from typing import Optional
import pandas as pd

from trading_bot.strategies.market_regime import (
    NewStrategy, Signal,
    _ema, _rsi, _atr_series, _macd,
)


class EMAMACDCrossover(NewStrategy):
    """
    EMA(9) crosses above EMA(21) with MACD and RSI confirmation.

    Entry : SPY BULL + EMA9 crosses above EMA21 + MACD above signal
            + RSI(14) 45–65 + volume > 1.1× 20-day avg
    Exit  : EMA9 crosses below EMA21 OR MACD crosses below signal
            OR ATR-based stop OR ATR-based TP OR max-hold
    """

    name = "EMA_MACD_Crossover"

    default_config: dict = {
        "ema_fast":            9,
        "ema_slow":            21,
        "macd_fast":           12,
        "macd_slow":           26,
        "macd_signal":         9,
        "rsi_period":          14,
        "rsi_min":             45,
        "rsi_max":             65,
        "vol_ratio_min":       1.1,
        "atr_stop_multiplier": 1.5,
        "atr_tp_multiplier":   2.0,
        "max_hold_bars":       20,
    }

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        config: Optional[dict] = None,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        cfg = {**self.default_config, **(config or {})}
        signals: list[Signal] = []

        if len(df) < 250:
            return signals

        close  = df["Close"].ffill().dropna()
        volume = df["Volume"].ffill() if "Volume" in df.columns else pd.Series(1.0, index=df.index)

        ema_f  = _ema(close, cfg["ema_fast"]).ffill()
        ema_s  = _ema(close, cfg["ema_slow"]).ffill()
        rsi14  = _rsi(close, cfg["rsi_period"]).ffill()
        macd_l, macd_sig, _ = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
        macd_l   = macd_l.ffill()
        macd_sig = macd_sig.ffill()
        atr      = _atr_series(df, 14).ffill()
        vol_avg20 = volume.rolling(20, min_periods=5).mean().ffill()

        regime_series = self._get_regime_series(df, spy_df)

        in_position = False
        entry_price = 0.0
        entry_atr   = 0.0
        bars_held   = 0

        for i in range(40, len(df)):
            date   = df.index[i]
            c      = float(close.iloc[i])
            ef_now = float(ema_f.iloc[i])
            ef_prv = float(ema_f.iloc[i - 1])
            es_now = float(ema_s.iloc[i])
            es_prv = float(ema_s.iloc[i - 1])
            ml_now = float(macd_l.iloc[i]) if not pd.isna(macd_l.iloc[i]) else 0.0
            ms_now = float(macd_sig.iloc[i]) if not pd.isna(macd_sig.iloc[i]) else 0.0
            ml_prv = float(macd_l.iloc[i - 1]) if not pd.isna(macd_l.iloc[i - 1]) else 0.0
            ms_prv = float(macd_sig.iloc[i - 1]) if not pd.isna(macd_sig.iloc[i - 1]) else 0.0
            rsi_v  = float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50.0
            atr_v  = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else c * 0.01
            cur_vol = float(volume.iloc[i])
            avg_vol = float(vol_avg20.iloc[i]) if not pd.isna(vol_avg20.iloc[i]) else cur_vol
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
            regime = str(regime_series.iloc[i])

            bullish_ema_cross = ef_prv <= es_prv and ef_now > es_now
            bearish_ema_cross = ef_prv >= es_prv and ef_now < es_now
            macd_bearish_cross = ml_prv >= ms_prv and ml_now < ms_now

            # ── Manage open position ──────────────────────────────────────
            if in_position:
                bars_held += 1
                stop_price = entry_price - cfg["atr_stop_multiplier"] * entry_atr
                tp_price   = entry_price + cfg["atr_tp_multiplier"] * entry_atr
                exit_hit = (
                    bearish_ema_cross
                    or macd_bearish_cross
                    or c <= stop_price
                    or c >= tp_price
                    or bars_held >= cfg["max_hold_bars"]
                )
                if exit_hit:
                    signals.append(Signal(
                        date=date, symbol=symbol,
                        strategy_name=self.name, side="SELL",
                        price=c, stop_loss=stop_price, take_profit=tp_price,
                        confidence=0.70,
                        reason=f"EMA/MACD exit bars={bars_held} c={c:.2f}",
                        regime=regime, hold_bars=bars_held,
                    ))
                    in_position = False
                    entry_price = entry_atr = 0.0
                    bars_held = 0
                continue

            # ── Entry ─────────────────────────────────────────────────────
            if regime != "BULL":
                continue
            if not bullish_ema_cross:
                continue
            if ml_now <= ms_now:          # MACD must be above signal
                continue
            if not (cfg["rsi_min"] <= rsi_v <= cfg["rsi_max"]):
                continue
            if vol_ratio < cfg["vol_ratio_min"]:
                continue

            stop = c - cfg["atr_stop_multiplier"] * atr_v
            tp   = c + cfg["atr_tp_multiplier"] * atr_v
            conf = round(min(0.65 + (rsi_v - cfg["rsi_min"]) / 100, 0.90), 2)

            signals.append(Signal(
                date=date, symbol=symbol,
                strategy_name=self.name, side="BUY",
                price=c, stop_loss=stop, take_profit=tp,
                confidence=conf,
                reason=f"EMA({cfg['ema_fast']}) crossed EMA({cfg['ema_slow']}) "
                       f"MACD above signal | RSI={rsi_v:.0f} | vol={vol_ratio:.1f}x",
                regime=regime, hold_bars=cfg["max_hold_bars"],
            ))
            in_position = True
            entry_price = c
            entry_atr   = atr_v
            bars_held   = 0

        return signals
