"""
Validate all 5 day trading strategies with 1-year of intraday backtest data.

Usage:
    python scripts/validate_daytrading_strategies.py

Output:
    Prints summary table to stdout.
    Saves results to: data/daytrading_validation.csv
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yfinance as yf

from app.services.strategy.daytrading.market_open import ET, compute_vwap, regime_allows_strategy
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP

SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

# yfinance limits: 5m → 60d, 15m → 730d
PERIOD_5M = "60d"
PERIOD_15M = "730d"

MIN_TRADES_PER_DAY = 0.5
MIN_WIN_RATE = 45.0


def download_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  Downloading {symbol} 5m ({PERIOD_5M})…", end=" ", flush=True)
    df5 = yf.download(symbol, period=PERIOD_5M, interval="5m", progress=False)
    print("done.")
    time.sleep(0.5)

    print(f"  Downloading {symbol} 15m ({PERIOD_15M})…", end=" ", flush=True)
    df15 = yf.download(symbol, period=PERIOD_15M, interval="15m", progress=False)
    print("done.")
    time.sleep(0.5)

    for df in (df5, df15):
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            idx = pd.to_datetime(df.index)
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC").tz_convert(ET)
            else:
                idx = idx.tz_convert(ET)
            df.index = idx

    return df5, df15


def simulate_strategy(strategy, df5: pd.DataFrame, df15: pd.DataFrame, symbol: str) -> dict:
    trades = []
    dates = sorted(set(df5.index.date))
    trading_days = len(dates)

    for date in dates:
        day5 = df5[df5.index.date == date]
        day15 = df15[df15.index.date == date] if not df15.empty else pd.DataFrame()

        if len(day5) < 4:
            continue

        # simple regime
        prev = df5[df5.index.date < date]
        if prev.empty:
            regime = "CHOPPY"
        else:
            prior_close = float(prev["Close"].iloc[-1])
            open_p = float(day5["Open"].iloc[0])
            current = float(day5["Close"].iloc[-1])
            vwap_s = compute_vwap(day5)
            vwap_v = float(vwap_s.iloc[-1])
            if open_p > prior_close and current > vwap_v:
                regime = "BULL_OPEN"
            elif open_p < prior_close and current < vwap_v:
                regime = "BEAR_OPEN"
            else:
                regime = "CHOPPY"

        if not regime_allows_strategy(regime, strategy.name):
            continue

        try:
            signals = strategy.generate_signals(day5, day15, symbol, None, regime)
        except Exception as e:
            continue

        for sig in signals:
            if sig.direction == "HOLD":
                continue

            entry = sig.entry_price
            stop = sig.stop_price
            target = sig.target_price
            sig_time = pd.Timestamp(sig.signal_time)
            future = day5[day5.index > sig_time]

            outcome = "EOD_EXIT"
            exit_price = entry
            hold_bars = 0
            max_hold = strategy.default_config.get("max_hold_bars", 60)

            for _, bar in future.iterrows():
                hold_bars += 1
                bh, bl = float(bar["High"]), float(bar["Low"])

                if sig.direction == "BUY":
                    if bl <= stop:
                        exit_price, outcome = stop, "STOPPED"
                        break
                    if bh >= target:
                        exit_price, outcome = target, "TARGET"
                        break
                else:
                    if bh >= stop:
                        exit_price, outcome = stop, "STOPPED"
                        break
                    if bl <= target:
                        exit_price, outcome = target, "TARGET"
                        break

                if hold_bars >= max_hold:
                    exit_price = float(bar["Close"])
                    outcome = "TIME_EXIT"
                    break

            if outcome == "EOD_EXIT":
                exit_price = float(future["Close"].iloc[-1]) if not future.empty else entry

            if sig.direction == "BUY":
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100

            trades.append({
                "date": date,
                "direction": sig.direction,
                "pnl_pct": pnl_pct,
                "outcome": outcome,
                "hold_bars": hold_bars,
            })

    if not trades:
        return {
            "trades_per_day": 0.0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_trades": 0,
            "max_consecutive_losses": 0,
            "trading_days": trading_days,
        }

    df = pd.DataFrame(trades)
    wins = (df["pnl_pct"] > 0).sum()
    total = len(df)
    win_rate = wins / total * 100

    # max consecutive losses
    results = (df["pnl_pct"] > 0).tolist()
    max_cl = cur_cl = 0
    for r in results:
        if not r:
            cur_cl += 1
            max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0

    return {
        "trades_per_day": round(total / max(trading_days, 1), 2),
        "win_rate": round(win_rate, 1),
        "avg_pnl_pct": round(df["pnl_pct"].mean(), 4),
        "total_trades": total,
        "max_consecutive_losses": max_cl,
        "trading_days": trading_days,
    }


def flag_status(stats: dict) -> str:
    if stats["total_trades"] == 0:
        return "⚠️ NO DATA"
    if stats["trades_per_day"] < MIN_TRADES_PER_DAY:
        return "⚠️ TOO SPARSE"
    if stats["win_rate"] < MIN_WIN_RATE:
        return "❌ UNDERPERFORMING"
    if stats["avg_pnl_pct"] < 0:
        return "❌ NEGATIVE EXPECTANCY"
    return "✅ OK"


def main():
    print("=" * 72)
    print("Day Trading Strategy Validation")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"5m period: {PERIOD_5M}  |  15m period: {PERIOD_15M}")
    print("=" * 72)

    rows = []

    for symbol in SYMBOLS:
        print(f"\n▶ {symbol}")
        df5, df15 = download_data(symbol)

        for strategy in ALL_STRATEGIES:
            print(f"  Running {strategy.name}…", end=" ", flush=True)
            stats = simulate_strategy(strategy, df5, df15, symbol)
            status = flag_status(stats)
            print(status)

            rows.append({
                "Symbol": symbol,
                "Strategy": strategy.name,
                "T/day": stats["trades_per_day"],
                "Win%": stats["win_rate"],
                "AvgP&L%": stats["avg_pnl_pct"],
                "Trades": stats["total_trades"],
                "MaxConsecLoss": stats["max_consecutive_losses"],
                "TradingDays": stats["trading_days"],
                "Status": status,
            })

    df_results = pd.DataFrame(rows)

    print("\n")
    print("=" * 85)
    print(f"{'Symbol':<8} {'Strategy':<26} {'T/day':>6} {'Win%':>6} {'AvgP&L':>8}  Status")
    print("-" * 85)
    for _, row in df_results.iterrows():
        print(
            f"{row['Symbol']:<8} {row['Strategy']:<26} "
            f"{row['T/day']:>6.2f} {row['Win%']:>5.1f}% "
            f"{row['AvgP&L%']:>+7.3f}%  {row['Status']}"
        )
    print("=" * 85)

    os.makedirs("data", exist_ok=True)
    out_path = "data/daytrading_validation.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
