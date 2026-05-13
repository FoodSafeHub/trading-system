from __future__ import annotations

"""
Strategy 4 — Pullback to Rising 50-EMA

The classic "buy the dip in an uptrend."  Enters when price pulls back to
a rising EMA(50) with RSI in a moderate pullback zone and a bullish wick
confirming rejection.

BULL behaviour : fully active.
BEAR behaviour : allowed UNLESS SPY is more than 10% below its SMA(200).
                 This makes it one of the two strategies that can still trade
                 in mild bear conditions (SPY close < SMA200 but not deeply).

Typical frequency: highest of all 5 strategies — 12–20 signals/year.
Max hold: 20 bars.
"""

from typing import Optional
import pandas as pd

from trading_bot.strategies.market_regime import (
    NewStrategy, Signal,
    _ema, _rsi,
)


class PullbackEMA50(NewStrategy):
    """
    Buy pullbacks to rising EMA(50).

    Entry : EMA(50) sloping up (today > 5 bars ago) + price within ±1% of EMA50
            + RSI(14) 35–55 + bullish wick (close-low)/(high-low) > 0.4
            + NOT in extreme bear (SPY > 10% below SMA200)
    Exit  : RSI(14) > 65 OR price > 3% above EMA50 OR hard stop 2% below EMA50
            OR max-hold
    """

    name = "Pullback_EMA50"

    default_config: dict = {
        "ema_trend":               50,
        "ema_slope_bars":          5,
        "price_ema_proximity_pct": 1.0,
        "rsi_period":              14,
        "rsi_min":                 35,
        "rsi_max":                 55,
        "wick_ratio_min":          0.4,
        "exit_rsi":                65,
        "exit_extension_pct":      3.0,
        "hard_stop_pct":           2.0,
        "max_hold_bars":           20,
        "bear_skip_threshold_pct": 10.0,  # skip if SPY > 10% below SMA200
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
        high   = df["High"].ffill()
        low    = df["Low"].ffill()

        ema50  = _ema(close, cfg["ema_trend"]).ffill()
        rsi14  = _rsi(close, cfg["rsi_period"]).ffill()

        regime_series = self._get_regime_series(df, spy_df)

        in_position = False
        entry_price = 0.0
        entry_stop  = 0.0
        entry_ema50 = 0.0
        bars_held   = 0

        slope_bars = cfg["ema_slope_bars"]

        for i in range(max(cfg["ema_trend"] + slope_bars, 60), len(df)):
            date    = df.index[i]
            c       = float(close.iloc[i])
            h       = float(high.iloc[i])
            lo      = float(low.iloc[i])
            e50     = float(ema50.iloc[i]) if not pd.isna(ema50.iloc[i]) else c
            e50_old = float(ema50.iloc[i - slope_bars]) if not pd.isna(ema50.iloc[i - slope_bars]) else e50
            rsi_v   = float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50.0
            regime  = str(regime_series.iloc[i])

            # ── Manage open position ──────────────────────────────────────
            if in_position:
                bars_held += 1
                extension_pct = (c / entry_ema50 - 1) * 100 if entry_ema50 > 0 else 0.0
                exit_hit = (
                    rsi_v > cfg["exit_rsi"]
                    or extension_pct > cfg["exit_extension_pct"]
                    or c <= entry_stop
                    or bars_held >= cfg["max_hold_bars"]
                )
                if exit_hit:
                    tp = entry_price * (1 + cfg["exit_extension_pct"] / 100)
                    signals.append(Signal(
                        date=date, symbol=symbol,
                        strategy_name=self.name, side="SELL",
                        price=c, stop_loss=entry_stop, take_profit=tp,
                        confidence=0.70,
                        reason=f"EMA50 pullback exit bars={bars_held} RSI={rsi_v:.0f} ext={extension_pct:.1f}%",
                        regime=regime, hold_bars=bars_held,
                    ))
                    in_position = False
                    entry_price = entry_stop = entry_ema50 = 0.0
                    bars_held = 0
                continue

            # ── Extreme bear skip (SPY >10% below SMA200) ────────────────
            spy_pct_below = self._spy_pct_below_sma200(spy_df, date)
            if spy_pct_below > cfg["bear_skip_threshold_pct"]:
                continue

            # ── Entry ─────────────────────────────────────────────────────
            # EMA50 must be rising
            if e50 <= e50_old:
                continue

            # Price must be within proximity band of EMA50
            prox_pct = abs(c - e50) / e50 * 100 if e50 > 0 else 999
            if prox_pct > cfg["price_ema_proximity_pct"]:
                continue

            # RSI in pullback zone
            if not (cfg["rsi_min"] <= rsi_v <= cfg["rsi_max"]):
                continue

            # Bullish wick confirmation
            bar_range = h - lo
            wick_ratio = (c - lo) / bar_range if bar_range > 0 else 0.0
            if wick_ratio < cfg["wick_ratio_min"]:
                continue

            stop = e50 * (1 - cfg["hard_stop_pct"] / 100)
            tp   = e50 * (1 + cfg["exit_extension_pct"] / 100)
            conf = round(min(0.60 + wick_ratio * 0.3, 0.90), 2)

            signals.append(Signal(
                date=date, symbol=symbol,
                strategy_name=self.name, side="BUY",
                price=c, stop_loss=stop, take_profit=tp,
                confidence=conf,
                reason=f"Pullback to EMA50={e50:.2f} (dist={prox_pct:.2f}%) "
                       f"RSI={rsi_v:.0f} wick={wick_ratio:.2f}",
                regime=regime, hold_bars=cfg["max_hold_bars"],
            ))
            in_position = True
            entry_price = c
            entry_stop  = stop
            entry_ema50 = e50
            bars_held   = 0

        return signals
