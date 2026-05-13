#!/usr/bin/env python
"""
Validation script for the 5 rewritten Perplexity strategies.

Downloads 10 years of daily OHLCV for AAPL, MSFT, GOOGL, SPY, NVDA (2015-2025)
via yfinance, runs each strategy's run() method bar-by-bar, simulates round-trip
trades (fill at next-bar open), and produces a summary table.

Usage:
    python scripts/validate_perplexity_strategies.py

Output:
    - Printed summary table (console)
    - data/perplexity_validation_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from app.services.market_regime import MarketRegime
from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend,
    MaCrossoverRsi,
    BreakoutConsolidation,
    BollingerMeanReversionUptrend,
    FibPullbackSupport,
)

START   = "2015-01-01"
END     = "2025-12-31"
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "SPY", "NVDA"]

STRATEGIES = [
    EmaMeanReversionUptrend(),
    MaCrossoverRsi(),
    BreakoutConsolidation(),
    BollingerMeanReversionUptrend(),
    FibPullbackSupport(),
]

MIN_SIGNALS_PER_YEAR = 5.0
MIN_WIN_RATE_PCT     = 45.0
MIN_AVG_PNL_PCT      = 0.0


def _download(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.ffill().dropna()
    return df


def _get_regime(df: pd.DataFrame, i: int) -> MarketRegime:
    """Simple SPY SMA200 regime based on df slice up to bar i."""
    close = df["Close"].iloc[:i]
    if len(close) < 200:
        return MarketRegime.BULL
    sma200 = close.rolling(200).mean().iloc[-1]
    c = float(close.iloc[-1])
    if c < sma200 * 0.9:
        return MarketRegime.DEEP_BEAR
    if c < sma200:
        return MarketRegime.BEAR
    return MarketRegime.BULL


def _simulate(strategy, df: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Bar-by-bar simulation. Fill BUY at next-bar open, SELL at next-bar open.
    Enforce stop and take_profit from signal.
    """
    min_bars = strategy.config.get("min_data_bars", 220)
    trades = []
    in_trade = False
    entry_price = 0.0
    stop_price  = 0.0
    target_price = 0.0
    entry_bar   = 0

    for i in range(min_bars, len(df) - 1):
        df_slice = df.iloc[:i]
        regime   = _get_regime(df, i)
        next_open = float(df["Open"].iloc[i + 1])

        if not in_trade:
            try:
                sig = strategy.run(symbol, df_slice, regime=regime)
            except Exception:
                continue
            if sig.direction == "BUY" and sig.stop_price and sig.target_price:
                in_trade     = True
                entry_price  = next_open
                stop_price   = sig.stop_price
                target_price = sig.target_price
                entry_bar    = i
        else:
            # Check stop/target on current bar
            lo = float(df["Low"].iloc[i])
            hi = float(df["High"].iloc[i])
            max_hold = strategy.config.get("max_hold_bars", 20)

            hit_stop   = lo <= stop_price
            hit_target = hi >= target_price
            hit_hold   = (i - entry_bar) >= max_hold

            if hit_stop:
                pnl_pct = (stop_price - entry_price) / entry_price * 100
                trades.append({"pnl_pct": pnl_pct, "hold_bars": i - entry_bar, "exit": "stop"})
                in_trade = False
            elif hit_target:
                pnl_pct = (target_price - entry_price) / entry_price * 100
                trades.append({"pnl_pct": pnl_pct, "hold_bars": i - entry_bar, "exit": "target"})
                in_trade = False
            elif hit_hold:
                # Exit at next open
                pnl_pct = (next_open - entry_price) / entry_price * 100
                trades.append({"pnl_pct": pnl_pct, "hold_bars": i - entry_bar, "exit": "max_hold"})
                in_trade = False
            else:
                # Check strategy SELL signal
                try:
                    sig = strategy.run(symbol, df_slice, regime=regime)
                except Exception:
                    continue
                if sig.direction == "SELL":
                    pnl_pct = (next_open - entry_price) / entry_price * 100
                    trades.append({"pnl_pct": pnl_pct, "hold_bars": i - entry_bar, "exit": "signal"})
                    in_trade = False

    # Close any open trade at last bar
    if in_trade:
        last_close = float(df["Close"].iloc[-1])
        pnl_pct = (last_close - entry_price) / entry_price * 100
        trades.append({"pnl_pct": pnl_pct, "hold_bars": len(df) - 1 - entry_bar, "exit": "eod"})

    return trades


def _status(sig_per_yr: float, win_rate: float, avg_pnl: float) -> str:
    issues = []
    if sig_per_yr < MIN_SIGNALS_PER_YEAR:
        issues.append("TOO SPARSE")
    if win_rate < MIN_WIN_RATE_PCT:
        issues.append("UNDERPERFORMING")
    if avg_pnl < MIN_AVG_PNL_PCT:
        issues.append("NEGATIVE EXPECTANCY")
    return " | ".join(issues) if issues else "OK"


def main():
    print(f"\n{'='*90}")
    print("  Perplexity Strategy Validation — 10 Years (2015-2025)")
    print(f"{'='*90}\n")

    data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            data[sym] = _download(sym)
            print(f"  {sym}: {len(data[sym])} bars")
        except Exception as exc:
            print(f"  {sym}: ERROR — {exc}")

    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    rows  = []

    for sym, df in data.items():
        for strategy in STRATEGIES:
            try:
                trades = _simulate(strategy, df, sym)
            except Exception as exc:
                print(f"  ERROR {sym}/{strategy.name}: {exc}")
                rows.append({
                    "Symbol": sym, "Strategy": strategy.name,
                    "Trades": 0, "Wins": 0, "Losses": 0,
                    "Win%": 0.0, "Avg P&L%": 0.0, "Avg Hold": 0.0,
                    "Trades/yr": 0.0, "Status": "ERROR",
                })
                continue

            wins     = sum(1 for t in trades if t["pnl_pct"] > 0)
            losses   = len(trades) - wins
            win_rate = wins / len(trades) * 100 if trades else 0.0
            avg_pnl  = sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0.0
            avg_hold = sum(t["hold_bars"] for t in trades) / len(trades) if trades else 0.0
            tpy      = len(trades) / years

            rows.append({
                "Symbol":    sym,
                "Strategy":  strategy.name,
                "Trades":    len(trades),
                "Wins":      wins,
                "Losses":    losses,
                "Win%":      round(win_rate, 1),
                "Avg P&L%":  round(avg_pnl, 2),
                "Avg Hold":  round(avg_hold, 1),
                "Trades/yr": round(tpy, 1),
                "Status":    _status(tpy, win_rate, avg_pnl),
            })

    results_df = pd.DataFrame(rows)

    header = (
        f"{'Symbol':<8} | {'Strategy':<28} | {'Trades/yr':>9} | "
        f"{'Win%':>6} | {'Avg P&L%':>9} | {'Avg Hold':>8} | Status"
    )
    print(f"\n{'-'*95}")
    print(header)
    print(f"{'-'*95}")
    for _, row in results_df.iterrows():
        print(
            f"{row['Symbol']:<8} | {row['Strategy']:<28} | "
            f"{row['Trades/yr']:>9.1f} | "
            f"{row['Win%']:>5.1f}% | "
            f"{row['Avg P&L%']:>+8.2f}% | "
            f"{row['Avg Hold']:>8.1f} | "
            f"{row['Status']}"
        )
    print(f"{'-'*95}\n")

    os.makedirs("data", exist_ok=True)
    csv_path = "data/perplexity_validation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")

    ok = (results_df["Status"] == "OK").sum()
    print(f"Summary: {ok}/{len(results_df)} strategy×symbol combinations passed all checks.\n")


if __name__ == "__main__":
    main()
