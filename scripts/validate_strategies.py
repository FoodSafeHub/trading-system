#!/usr/bin/env python
"""
Validation script for the 5 new swing trading strategies.

Downloads 10 years of daily OHLCV for AAPL, MSFT, GOOGL, SPY (2015–2025)
using yfinance, runs each strategy's generate_signals(), and produces a
summary table with signal frequency, win rate, avg P&L, and a status flag.

Usage:
    python scripts/validate_strategies.py

Output:
    - Printed summary table (console)
    - data/strategy_validation_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from trading_bot.strategies.rsi2_mean_reversion import RSI2MeanReversion
from trading_bot.strategies.ema_macd_crossover import EMAMACDCrossover
from trading_bot.strategies.bb_squeeze_breakout import BBSqueezeBreakout
from trading_bot.strategies.pullback_ema50 import PullbackEMA50
from trading_bot.strategies.vix_spike_reversal import VIXSpikeReversal
from trading_bot.strategies.market_regime import get_market_regime

# ── Configuration ─────────────────────────────────────────────────────────────

START = "2015-01-01"
END   = "2025-12-31"
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "SPY"]

STRATEGIES = [
    RSI2MeanReversion(),
    EMAMACDCrossover(),
    BBSqueezeBreakout(),
    PullbackEMA50(),
    VIXSpikeReversal(),
]

# Thresholds for flagging
MIN_SIGNALS_PER_YEAR  = 5.0    # fewer than this → TOO SPARSE
MIN_WIN_RATE_PCT      = 45.0   # below this → UNDERPERFORMING
MIN_AVG_PNL_PCT       = 0.0    # below this → NEGATIVE EXPECTANCY


# ── Data download ─────────────────────────────────────────────────────────────

def _download(symbol: str) -> pd.DataFrame:
    """Download daily OHLCV via yfinance; normalize column names."""
    df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    # yfinance may return MultiIndex columns when downloading single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.ffill().dropna()
    return df


# ── Round-trip P&L simulation ─────────────────────────────────────────────────

def _simulate_trades(signals, df: pd.DataFrame):
    """
    Simulate round-trip trades from the signal list.

    BUY signals set the entry; the next SELL closes it.
    If no SELL follows, the position is closed at the last bar close.
    Returns list of dicts with pnl_pct and hold_bars.
    """
    trades = []
    pending_buy = None

    for sig in signals:
        if sig.side == "BUY" and pending_buy is None:
            pending_buy = sig
        elif sig.side == "SELL" and pending_buy is not None:
            pnl_pct = (sig.price - pending_buy.price) / pending_buy.price * 100
            try:
                entry_idx = df.index.get_loc(pending_buy.date)
                exit_idx  = df.index.get_loc(sig.date)
                hold      = exit_idx - entry_idx
            except Exception:
                hold = sig.hold_bars
            trades.append({"pnl_pct": pnl_pct, "hold_bars": hold})
            pending_buy = None

    # Close any still-open position at last bar
    if pending_buy is not None:
        last_close = float(df["Close"].iloc[-1])
        pnl_pct = (last_close - pending_buy.price) / pending_buy.price * 100
        try:
            entry_idx = df.index.get_loc(pending_buy.date)
            hold      = len(df) - entry_idx - 1
        except Exception:
            hold = 0
        trades.append({"pnl_pct": pnl_pct, "hold_bars": hold})

    return trades


# ── Status evaluation ─────────────────────────────────────────────────────────

def _status(signals_per_year: float, win_rate: float, avg_pnl: float) -> str:
    issues = []
    if signals_per_year < MIN_SIGNALS_PER_YEAR:
        issues.append("TOO SPARSE")
    if win_rate < MIN_WIN_RATE_PCT:
        issues.append("UNDERPERFORMING")
    if avg_pnl < MIN_AVG_PNL_PCT:
        issues.append("NEGATIVE EXPECTANCY")
    return " | ".join(issues) if issues else "✅ OK"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*80}")
    print("  Strategy Validation — 10 Years (2015–2025)")
    print(f"{'='*80}\n")
    print(f"Downloading data for: {', '.join(SYMBOLS)} ...\n")

    data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            data[sym] = _download(sym)
            print(f"  {sym}: {len(data[sym])} bars")
        except Exception as exc:
            print(f"  {sym}: ERROR — {exc}")

    spy_df = data.get("SPY")
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25

    rows = []

    for sym, df in data.items():
        if df.empty:
            continue
        for strategy in STRATEGIES:
            try:
                signals = strategy.generate_signals(df, sym, spy_df=spy_df)
            except Exception as exc:
                print(f"  ERROR {sym} / {strategy.name}: {exc}")
                rows.append({
                    "Symbol":       sym,
                    "Strategy":     strategy.name,
                    "Total Signals": 0,
                    "Wins":         0,
                    "Losses":       0,
                    "Win%":         0.0,
                    "Avg P&L%":     0.0,
                    "Avg Hold Bars": 0.0,
                    "Signals/yr":   0.0,
                    "Status":       "ERROR",
                })
                continue

            buys  = [s for s in signals if s.side == "BUY"]
            sells = [s for s in signals if s.side == "SELL"]
            trades = _simulate_trades(signals, df)

            wins   = sum(1 for t in trades if t["pnl_pct"] > 0)
            losses = len(trades) - wins
            win_rate  = wins / len(trades) * 100 if trades else 0.0
            avg_pnl   = sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0.0
            avg_hold  = sum(t["hold_bars"] for t in trades) / len(trades) if trades else 0.0
            sig_per_yr = len(buys) / years

            status = _status(sig_per_yr, win_rate, avg_pnl)

            rows.append({
                "Symbol":         sym,
                "Strategy":       strategy.name,
                "Total Signals":  len(buys),
                "Wins":           wins,
                "Losses":         losses,
                "Win%":           round(win_rate, 1),
                "Avg P&L%":       round(avg_pnl, 2),
                "Avg Hold Bars":  round(avg_hold, 1),
                "Signals/yr":     round(sig_per_yr, 1),
                "Status":         status,
            })

    results_df = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────────────────
    header = (
        f"{'Symbol':<8} | {'Strategy':<25} | {'Signals/yr':>10} | "
        f"{'Win%':>6} | {'Avg P&L':>8} | Status"
    )
    print(f"\n{'-'*90}")
    print(header)
    print(f"{'-'*90}")
    for _, row in results_df.iterrows():
        print(
            f"{row['Symbol']:<8} | {row['Strategy']:<25} | "
            f"{row['Signals/yr']:>10.1f} | "
            f"{row['Win%']:>5.1f}% | "
            f"{row['Avg P&L%']:>+7.2f}% | "
            f"{row['Status']}"
        )
    print(f"{'-'*90}\n")

    # ── Regime breakdown ──────────────────────────────────────────────────────
    if spy_df is not None:
        regime_series = get_market_regime(spy_df)
        bull_pct = (regime_series == "BULL").mean() * 100
        bear_pct = 100 - bull_pct
        print(f"SPY Market Regime (2015–2025):")
        print(f"  BULL: {bull_pct:.1f}%  |  BEAR: {bear_pct:.1f}%\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    csv_path = "data/strategy_validation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count    = (results_df["Status"] == "✅ OK").sum()
    total_count = len(results_df)
    print(f"Summary: {ok_count}/{total_count} strategy×symbol combinations passed all checks.\n")


if __name__ == "__main__":
    main()
