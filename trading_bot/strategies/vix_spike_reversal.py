from __future__ import annotations

"""
Strategy 5 — VIX Spike Reversal / Fear Capitulation

Catches sharp reversals after panic sell-offs.  Uses ATR% as a VIX proxy
(available from daily OHLCV without needing actual VIX data).

Works in BOTH BULL and BEAR regimes — one of two strategies allowed when SPY
is below its SMA(200).  Position size should be halved in BEAR.

Typical frequency: 4–10 signals/year (relatively rare, high-quality setups).
Max hold: 8 bars (fast spike-reversal trades).
"""

from typing import Optional
import pandas as pd

from trading_bot.strategies.market_regime import (
    NewStrategy, Signal,
    _rsi, _atr_series, _bb,
)


class VIXSpikeReversal(NewStrategy):
    """
    ATR-spike + oversold + long lower wick reversal.

    Entry : ATR% > 3.0 (fear spike) + RSI(14) < 30 + BB position < 0.15
            + long lower wick > 0.5 of bar range + prior 3-bar decline > 2%
            (works in BULL AND BEAR — no regime gate on entry)
    Exit  : ATR% < 2.0 (calm returns) OR RSI(14) > 55 OR hard stop -4%
            OR TP +6% OR max-hold 8 bars
    """

    name = "VIX_Spike_Reversal"

    default_config: dict = {
        "atr_period":         14,
        "atr_spike_threshold": 3.0,   # ATR% must spike above this
        "atr_exit_threshold":  2.0,   # exit when ATR% falls back below this
        "rsi_period":          14,
        "rsi_entry_max":       30,    # must be deeply oversold
        "rsi_exit":            55,    # recovery exit
        "bb_pos_max":          0.15,  # must be near lower Bollinger Band
        "wick_ratio_min":      0.5,   # lower wick > 50% of bar range
        "prior_decline_pct":   2.0,   # price 3 bars ago > close today + this%
        "prior_decline_bars":  3,
        "hard_stop_pct":       4.0,
        "take_profit_pct":     6.0,
        "max_hold_bars":       8,
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

        atr    = _atr_series(df, cfg["atr_period"]).ffill()
        rsi14  = _rsi(close, cfg["rsi_period"]).ffill()
        _, _, _, _, bb_pos = _bb(close, 20, 2.0)
        bb_pos = bb_pos.ffill()

        regime_series = self._get_regime_series(df, spy_df)

        in_position = False
        entry_price = 0.0
        bars_held   = 0
        entry_atr_pct = 0.0

        pb = cfg["prior_decline_bars"]

        for i in range(max(50, pb + 5), len(df)):
            date   = df.index[i]
            c      = float(close.iloc[i])
            h      = float(high.iloc[i])
            lo     = float(low.iloc[i])
            atr_v  = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else c * 0.01
            atr_pct = atr_v / c * 100 if c > 0 else 0.0
            rsi_v  = float(rsi14.iloc[i]) if not pd.isna(rsi14.iloc[i]) else 50.0
            bb_p   = float(bb_pos.iloc[i]) if not pd.isna(bb_pos.iloc[i]) else 0.5
            regime = str(regime_series.iloc[i])

            # ── Manage open position ──────────────────────────────────────
            if in_position:
                bars_held += 1
                stop_price = entry_price * (1 - cfg["hard_stop_pct"] / 100)
                tp_price   = entry_price * (1 + cfg["take_profit_pct"] / 100)
                exit_hit = (
                    atr_pct < cfg["atr_exit_threshold"]   # volatility fades
                    or rsi_v > cfg["rsi_exit"]             # recovery
                    or c <= stop_price                     # hard stop
                    or c >= tp_price                       # take profit
                    or bars_held >= cfg["max_hold_bars"]
                )
                if exit_hit:
                    signals.append(Signal(
                        date=date, symbol=symbol,
                        strategy_name=self.name, side="SELL",
                        price=c, stop_loss=stop_price, take_profit=tp_price,
                        confidence=0.70,
                        reason=f"VIX reversal exit bars={bars_held} ATR%={atr_pct:.1f} RSI={rsi_v:.0f}",
                        regime=regime, hold_bars=bars_held,
                    ))
                    in_position = False
                    entry_price = 0.0
                    bars_held   = 0
                    entry_atr_pct = 0.0
                continue

            # ── Entry (BOTH regimes allowed) ──────────────────────────────
            # ATR% spike — fear/panic
            if atr_pct < cfg["atr_spike_threshold"]:
                continue
            # Deeply oversold RSI
            if rsi_v >= cfg["rsi_entry_max"]:
                continue
            # Near lower Bollinger Band
            if bb_p >= cfg["bb_pos_max"]:
                continue

            # Long lower wick (reversal candle)
            bar_range  = h - lo
            wick_ratio = (c - lo) / bar_range if bar_range > 0 else 0.0
            if wick_ratio < cfg["wick_ratio_min"]:
                continue

            # Prior decline confirmation
            c_ago = float(close.iloc[i - pb]) if not pd.isna(close.iloc[i - pb]) else c
            decline_pct = (c_ago - c) / c_ago * 100 if c_ago > 0 else 0.0
            if decline_pct < cfg["prior_decline_pct"]:
                continue

            stop = c * (1 - cfg["hard_stop_pct"] / 100)
            tp   = c * (1 + cfg["take_profit_pct"] / 100)
            conf = round(min(0.65 + (cfg["rsi_entry_max"] - rsi_v) / 40, 0.90), 2)

            signals.append(Signal(
                date=date, symbol=symbol,
                strategy_name=self.name, side="BUY",
                price=c, stop_loss=stop, take_profit=tp,
                confidence=conf,
                reason=f"VIX spike ATR%={atr_pct:.1f} RSI={rsi_v:.0f} "
                       f"BB_pos={bb_p:.2f} wick={wick_ratio:.2f} "
                       f"decline({pb}d)={decline_pct:.1f}% | regime={regime}",
                regime=regime, hold_bars=cfg["max_hold_bars"],
            ))
            in_position = True
            entry_price = c
            entry_atr_pct = atr_pct
            bars_held   = 0

        return signals
