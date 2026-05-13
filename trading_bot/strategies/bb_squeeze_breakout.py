from __future__ import annotations

"""
Strategy 3 — Bollinger Band Squeeze + Breakout

Captures post-consolidation expansion moves.  Detects a bandwidth squeeze
(contracting volatility) and then buys the breakout when price closes above
the upper band with momentum and volume confirmation.

BULL behaviour : fully active.
BEAR behaviour : strategy is OFF (naturally — breakouts fail in downtrends).

Typical frequency (BULL market): 6–12 signals/year per symbol.
Max hold: 15 bars.
"""

from typing import Optional
import pandas as pd

from trading_bot.strategies.market_regime import (
    NewStrategy, Signal,
    _rsi, _atr_series, _bb,
)


class BBSqueezeBreakout(NewStrategy):
    """
    Bollinger Band squeeze detected, then breakout above upper band.

    Entry : SPY BULL + BB width contracting ≥5 bars + close > upper BB
            + RSI(14) > 50 + volume > 1.3× 20-day avg
    Exit  : close < middle BB OR RSI(14) > 80 OR stop (lower BB at entry)
            OR TP (mid + 2×width) OR max-hold
    """

    name = "BB_Squeeze_Breakout"

    default_config: dict = {
        "bb_period":      20,
        "bb_std":         2.0,
        "squeeze_bars":   5,     # consecutive bars of contracting bandwidth
        "rsi_period":     14,
        "rsi_entry_min":  50,
        "rsi_overbought": 80,
        "vol_ratio_min":  1.3,
        "max_hold_bars":  15,
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

        bb_upper, bb_mid, bb_lower, bb_bw, _ = _bb(close, cfg["bb_period"], cfg["bb_std"])
        bb_upper = bb_upper.ffill()
        bb_mid   = bb_mid.ffill()
        bb_lower = bb_lower.ffill()
        bb_bw    = bb_bw.ffill()

        rsi14    = _rsi(close, cfg["rsi_period"]).ffill()
        vol_avg  = volume.rolling(20, min_periods=5).mean().ffill()

        regime_series = self._get_regime_series(df, spy_df)

        in_position = False
        entry_price = 0.0
        entry_stop  = 0.0
        entry_tp    = 0.0
        bars_held   = 0

        sq = cfg["squeeze_bars"]

        for i in range(cfg["bb_period"] + sq + 5, len(df)):
            date   = df.index[i]
            c      = float(close.iloc[i])
            u_now  = float(bb_upper.iloc[i]) if not pd.isna(bb_upper.iloc[i]) else c
            m_now  = float(bb_mid.iloc[i]) if not pd.isna(bb_mid.iloc[i]) else c
            l_now  = float(bb_lower.iloc[i]) if not pd.isna(bb_lower.iloc[i]) else c * 0.95
            rsi_v  = float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50.0
            bw_now = float(bb_bw.iloc[i]) if not pd.isna(bb_bw.iloc[i]) else 0.0
            cur_vol = float(volume.iloc[i])
            avg_vol = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else cur_vol
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
            regime = str(regime_series.iloc[i])

            # ── Manage open position ──────────────────────────────────────
            if in_position:
                bars_held += 1
                exit_hit = (
                    c < m_now                        # closes below middle BB
                    or rsi_v > cfg["rsi_overbought"] # overbought exit
                    or c <= entry_stop               # hard stop (lower BB at entry)
                    or c >= entry_tp                 # take profit
                    or bars_held >= cfg["max_hold_bars"]
                )
                if exit_hit:
                    signals.append(Signal(
                        date=date, symbol=symbol,
                        strategy_name=self.name, side="SELL",
                        price=c, stop_loss=entry_stop, take_profit=entry_tp,
                        confidence=0.70,
                        reason=f"BB exit bars={bars_held} RSI={rsi_v:.0f} c={c:.2f} mid={m_now:.2f}",
                        regime=regime, hold_bars=bars_held,
                    ))
                    in_position = False
                    entry_price = entry_stop = entry_tp = 0.0
                    bars_held = 0
                continue

            # ── Entry ─────────────────────────────────────────────────────
            if regime != "BULL":
                continue
            if c <= u_now:           # must close above upper band
                continue
            if rsi_v < cfg["rsi_entry_min"]:
                continue
            if vol_ratio < cfg["vol_ratio_min"]:
                continue

            # Bandwidth squeeze: last sq bars all decreasing
            bw_window = [float(bb_bw.iloc[j]) for j in range(i - sq, i) if not pd.isna(bb_bw.iloc[j])]
            if len(bw_window) < sq:
                continue
            squeeze = all(bw_window[k] > bw_window[k + 1] for k in range(len(bw_window) - 1))
            if not squeeze:
                continue

            # Stop at lower band; TP at mid + 2×bandwidth
            band_width = u_now - l_now
            stop = l_now
            tp   = m_now + 2.0 * band_width
            conf = round(min(0.65 + vol_ratio / 10, 0.90), 2)

            signals.append(Signal(
                date=date, symbol=symbol,
                strategy_name=self.name, side="BUY",
                price=c, stop_loss=stop, take_profit=tp,
                confidence=conf,
                reason=f"BB squeeze {sq}+ bars | breakout above upper={u_now:.2f} "
                       f"| RSI={rsi_v:.0f} | vol={vol_ratio:.1f}x",
                regime=regime, hold_bars=cfg["max_hold_bars"],
            ))
            in_position = True
            entry_price = c
            entry_stop  = stop
            entry_tp    = tp
            bars_held   = 0

        return signals
