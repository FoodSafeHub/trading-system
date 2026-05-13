from __future__ import annotations

"""
Strategy 1 — RSI-2 Mean Reversion (Connors-style)

Based on Larry Connors' RSI(2) system.  Buys short-term oversold dips inside
a long-term uptrend.  Historical win rate on SPY > 85% over 10+ years.

BULL behaviour: fires whenever RSI(2) dips below the entry threshold while
                the symbol is above its own SMA(200) and SPY is in BULL regime.
BEAR behaviour: strategy is OFF — no entries in a bear market.

Typical frequency (BULL market): 8–18 signals/year per large-cap symbol.
Max hold: 10 bars.  Very short-duration mean-reversion trades.
"""

from typing import Optional
import pandas as pd

from trading_bot.strategies.market_regime import (
    NewStrategy, Signal,
    _sma, _ema, _rsi, _atr_series,
)


class RSI2MeanReversion(NewStrategy):
    """
    Connors RSI(2) mean-reversion swing trade.

    Entry : SPY BULL + symbol > SMA(200) + RSI(2) < 10 + ATR% <= skip threshold
    Exit  : close > SMA(5) OR RSI(2) > 70 OR hard stop/TP/max-hold hit
    """

    name = "RSI2_Mean_Reversion"

    default_config: dict = {
        "rsi_period":           2,
        "rsi_entry_threshold":  10,    # RSI(2) must close below this to enter
        "rsi_exit_threshold":   70,    # RSI(2) crosses above → exit
        "sma_trend":            200,   # symbol must be above this SMA
        "exit_sma":             5,     # close above this SMA → exit
        "hard_stop_pct":        5.0,   # % below entry
        "take_profit_pct":      8.0,   # % above entry
        "max_hold_bars":        10,
        "atr_skip_threshold":   5.0,   # skip entry if ATR% > this (extreme vol)
    }

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        config: Optional[dict] = None,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """
        Scan all bars in df and return a list of BUY signals.
        Exit signals (SELL) are NOT returned — the backtest engine handles
        stop/TP/max-hold logic internally; this method only flags entries.
        """
        cfg = {**self.default_config, **(config or {})}
        signals: list[Signal] = []

        if len(df) < 250:
            return signals  # not enough warmup bars

        close  = df["Close"].ffill().dropna()
        high   = df["High"].ffill()
        low    = df["Low"].ffill()

        rsi2   = _rsi(close, cfg["rsi_period"]).ffill()
        sma200 = _sma(close, cfg["sma_trend"]).ffill()
        sma5   = _sma(close, cfg["exit_sma"]).ffill()
        atr    = _atr_series(df, 14).ffill()

        regime_series = self._get_regime_series(df, spy_df)

        # Walk bar by bar; track position state to avoid re-entering
        in_position    = False
        entry_price    = 0.0
        bars_held      = 0

        for i in range(cfg["sma_trend"], len(df)):
            date      = df.index[i]
            c         = float(close.iloc[i])
            r2        = float(rsi2.iloc[i]) if not pd.isna(rsi2.iloc[i]) else 50.0
            s200      = float(sma200.iloc[i]) if not pd.isna(sma200.iloc[i]) else 0.0
            s5        = float(sma5.iloc[i]) if not pd.isna(sma5.iloc[i]) else c
            atr_v     = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else c * 0.01
            atr_pct   = atr_v / c * 100 if c > 0 else 0.0
            regime    = str(regime_series.iloc[i])

            # ── Manage open position ──────────────────────────────────────
            if in_position:
                bars_held += 1
                stop_price = entry_price * (1 - cfg["hard_stop_pct"] / 100)
                tp_price   = entry_price * (1 + cfg["take_profit_pct"] / 100)
                exit_hit = (
                    c > s5                           # above exit SMA
                    or r2 > cfg["rsi_exit_threshold"] # RSI crosses up
                    or c <= stop_price                # hard stop
                    or c >= tp_price                  # hard take profit
                    or bars_held >= cfg["max_hold_bars"]
                )
                if exit_hit:
                    signals.append(Signal(
                        date=date, symbol=symbol,
                        strategy_name=self.name, side="SELL",
                        price=c,
                        stop_loss=stop_price, take_profit=tp_price,
                        confidence=0.70,
                        reason=f"Exit RSI2={r2:.1f} close={c:.2f} sma5={s5:.2f} bars={bars_held}",
                        regime=regime, hold_bars=bars_held,
                    ))
                    in_position = False
                    entry_price = 0.0
                    bars_held   = 0
                continue

            # ── Entry conditions ──────────────────────────────────────────
            if regime != "BULL":
                continue                           # strategy OFF in bear
            if s200 <= 0 or c <= s200:
                continue                           # symbol below own SMA200
            if atr_pct > cfg["atr_skip_threshold"]:
                continue                           # extreme volatility skip
            if r2 >= cfg["rsi_entry_threshold"]:
                continue                           # RSI not deeply oversold

            stop  = c * (1 - cfg["hard_stop_pct"] / 100)
            tp    = c * (1 + cfg["take_profit_pct"] / 100)
            conf  = round(min(0.65 + (cfg["rsi_entry_threshold"] - r2) / 20, 0.95), 2)

            signals.append(Signal(
                date=date, symbol=symbol,
                strategy_name=self.name, side="BUY",
                price=c, stop_loss=stop, take_profit=tp,
                confidence=conf,
                reason=f"RSI(2)={r2:.1f} < {cfg['rsi_entry_threshold']} | "
                       f"above SMA200={s200:.2f} | ATR%={atr_pct:.1f}",
                regime=regime, hold_bars=cfg["max_hold_bars"],
            ))
            in_position = True
            entry_price = c
            bars_held   = 0

        return signals
