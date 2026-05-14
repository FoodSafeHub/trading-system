#!/usr/bin/env python
"""
Trade analysis: winner vs loser feature breakdown for any strategy+symbol.

For each trade recorded in a backtest run, captures the entry-bar indicator
snapshot and compares distributions between winning and losing trades.

Output:
  - Console report with mean/median/separation per feature
  - data/trade_analysis_{strategy}_{symbol}.csv
  - data/trade_analysis_{strategy}_{symbol}.json  (machine-readable, for optimizer)

Usage:
    python scripts/analyze_trades.py --strategy BB_Mean_Reversion --symbol NVDA
    python scripts/analyze_trades.py --strategy EMA_Mean_Reversion --symbol SPY --period 10y
    python scripts/analyze_trades.py --all --period 5y   (runs all 5x6 combinations)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
import pandas as pd

from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend, MaCrossoverRsi, BreakoutConsolidation,
    BollingerMeanReversionUptrend, FibPullbackSupport,
)
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.market_data.provider import get_ohlcv
from app.services.backtest.perplexity_engine import _atr_series

STRATEGIES = {
    "EMA_Mean_Reversion":    EmaMeanReversionUptrend(),
    "MA_Crossover_RSI":      MaCrossoverRsi(),
    "Breakout_Consolidation": BreakoutConsolidation(),
    "BB_Mean_Reversion":     BollingerMeanReversionUptrend(),
    "Fib_Pullback_Support":  FibPullbackSupport(),
}

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL"]


# ── Feature extraction from trade pairs ──────────────────────────────────────

def _extract_features(trade: dict, df_full: pd.DataFrame) -> dict:
    """
    Re-derive entry-bar indicator snapshot from the trade record.
    The backtest stores entry date and 'indicators' dict in the BUY record.
    We supplement with a few derived features.
    """
    features = {}

    # Raw indicators stored at signal time
    ind = trade.get("indicators") or {}
    for k, v in ind.items():
        features[k] = v

    # regime and volatility bucket
    features["regime"]            = trade.get("regime", "unknown")
    features["volatility_bucket"] = trade.get("volatility_bucket", "unknown")
    features["atr_pct"]           = trade.get("atr_pct", None)

    # Derive volume ratio from df if not in indicators
    if "volume_ratio" not in features and "Volume" in df_full.columns:
        entry_date = trade.get("date")
        try:
            idx = df_full.index.get_loc(entry_date) if entry_date in df_full.index else None
            if idx and idx > 20:
                avg_vol = float(df_full["Volume"].iloc[idx - 20:idx].mean())
                cur_vol = float(df_full["Volume"].iloc[idx])
                features["volume_ratio"] = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        except Exception:
            pass

    return features


def extract_trade_snapshots(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 10_000,
    position_pct: float = 0.20,
) -> list[dict]:
    """
    Run backtest and return one dict per completed trade with:
    - all entry-bar indicators
    - outcome: 'win' | 'loss'
    - pnl_pct, hold_bars, exit_type
    """
    strat = STRATEGIES[strategy_name]
    df_full = get_ohlcv(symbol, period=period)

    r = run_perplexity_backtest(
        strat, symbol, period=period,
        initial_capital=initial_capital, position_pct=position_pct,
    )

    buy_trades  = [t for t in r.trades if t["side"] == "BUY"]
    sell_trades = [t for t in r.trades if "SELL" in t["side"]]

    snapshots = []
    for buy, sell in zip(buy_trades, sell_trades):
        pnl      = sell.get("pnl") or 0.0
        buy_val  = buy.get("value") or 1.0
        pnl_pct  = round(pnl / buy_val * 100, 3) if buy_val else 0.0
        outcome  = "win" if pnl > 0 else "loss"

        features = _extract_features(buy, df_full)
        features.update({
            "symbol":        symbol,
            "strategy":      strategy_name,
            "entry_date":    buy.get("date"),
            "exit_date":     sell.get("date"),
            "entry_price":   buy.get("price"),
            "exit_price":    sell.get("price"),
            "exit_type":     sell.get("side"),
            "pnl_pct":       pnl_pct,
            "pnl_usd":       round(pnl, 2),
            "outcome":       outcome,
            "confidence":    buy.get("confidence"),
            "reason":        buy.get("reason"),
        })
        snapshots.append(features)

    return snapshots


# ── Statistical analysis ──────────────────────────────────────────────────────

def _mean(vals):
    return round(statistics.mean(vals), 4) if vals else None

def _median(vals):
    return round(statistics.median(vals), 4) if vals else None

def _std(vals):
    return round(statistics.stdev(vals), 4) if len(vals) >= 2 else None

def _cohens_d(w_vals, l_vals):
    """Effect size: how well does this feature separate winners from losers."""
    if len(w_vals) < 2 or len(l_vals) < 2:
        return None
    wm, lm = statistics.mean(w_vals), statistics.mean(l_vals)
    var_w = statistics.variance(w_vals)
    var_l = statistics.variance(l_vals)
    pooled = ((var_w + var_l) / 2) ** 0.5 or 1e-9
    return round((wm - lm) / pooled, 3)


def analyze_snapshots(snapshots: list[dict]) -> dict:
    """
    Returns a dict keyed by feature name, each with:
    wins_mean, losses_mean, wins_median, losses_median, cohens_d, signal
    """
    wins   = [s for s in snapshots if s["outcome"] == "win"]
    losses = [s for s in snapshots if s["outcome"] == "loss"]

    if not wins or not losses:
        return {}

    # Collect all numeric feature keys
    numeric_keys = set()
    for s in snapshots:
        for k, v in s.items():
            if isinstance(v, (int, float)) and k not in (
                "entry_price", "exit_price", "pnl_usd", "pnl_pct", "confidence"
            ):
                numeric_keys.add(k)

    results = {}
    for key in sorted(numeric_keys):
        w_vals = [s[key] for s in wins   if isinstance(s.get(key), (int, float))]
        l_vals = [s[key] for s in losses if isinstance(s.get(key), (int, float))]
        if len(w_vals) < 3 or len(l_vals) < 3:
            continue
        d = _cohens_d(w_vals, l_vals)
        results[key] = {
            "wins_mean":   _mean(w_vals),
            "losses_mean": _mean(l_vals),
            "wins_median": _median(w_vals),
            "losses_median": _median(l_vals),
            "wins_std":    _std(w_vals),
            "losses_std":  _std(l_vals),
            "cohens_d":    d,
            "signal":      "higher=win" if (d or 0) > 0 else "lower=win",
            "strength":    "strong" if abs(d or 0) >= 0.5 else ("moderate" if abs(d or 0) >= 0.2 else "weak"),
        }

    return results


def _print_report(
    strategy: str, symbol: str, period: str,
    snapshots: list[dict], analysis: dict,
):
    wins   = [s for s in snapshots if s["outcome"] == "win"]
    losses = [s for s in snapshots if s["outcome"] == "loss"]
    n      = len(snapshots)
    wr     = len(wins) / n * 100 if n else 0
    avg_win_pct  = _mean([s["pnl_pct"] for s in wins])   or 0
    avg_loss_pct = _mean([s["pnl_pct"] for s in losses]) or 0
    expectancy   = round(wr / 100 * avg_win_pct + (1 - wr / 100) * avg_loss_pct, 3)

    SEP = "=" * 88
    sep = "-" * 88
    print(f"\n{SEP}")
    print(f"  Trade Analysis: {strategy} / {symbol} / {period}")
    print(f"  Trades: {n}  |  Win rate: {wr:.1f}%  |  Avg win: +{avg_win_pct:.2f}%  "
          f"|  Avg loss: {avg_loss_pct:.2f}%  |  Expectancy: {expectancy:+.3f}%")
    print(SEP)

    # Sort by |cohen's d| descending
    ranked = sorted(
        [(k, v) for k, v in analysis.items() if v.get("cohens_d") is not None],
        key=lambda x: abs(x[1]["cohens_d"]),
        reverse=True,
    )

    if not ranked:
        print("  Not enough data to compute feature separation.")
        return

    print(f"\n{'Feature':<22} {'W-mean':>8} {'L-mean':>8} {'W-med':>7} {'L-med':>7} "
          f"{'Cohen d':>8} {'Strength':<10} {'Signal'}")
    print(sep)
    for key, v in ranked:
        print(
            f"  {key:<20} {str(v['wins_mean']):>8} {str(v['losses_mean']):>8} "
            f"{str(v['wins_median']):>7} {str(v['losses_median']):>7} "
            f"{str(v['cohens_d']):>8}  {v['strength']:<10} {v['signal']}"
        )

    # Recommended filters
    print(f"\n  Recommended entry filters (moderate+ separation only):")
    useful = [(k, v) for k, v in ranked if v["strength"] in ("strong", "moderate")]
    if not useful:
        print("  None — no feature shows meaningful separation between wins and losses.")
    else:
        for key, v in useful:
            if v["signal"] == "higher→win":
                # Use 25th percentile of win distribution as minimum threshold
                w_vals = [s[key] for s in wins if isinstance(s.get(key), (int, float))]
                threshold = round(sorted(w_vals)[len(w_vals) // 4], 3)
                print(f"  + require {key} >= {threshold}  "
                      f"(wins avg {v['wins_mean']} vs losses avg {v['losses_mean']})")
            else:
                w_vals = [s[key] for s in wins if isinstance(s.get(key), (int, float))]
                threshold = round(sorted(w_vals, reverse=True)[len(w_vals) // 4], 3)
                print(f"  + require {key} <= {threshold}  "
                      f"(wins avg {v['wins_mean']} vs losses avg {v['losses_mean']})")

    # Exit type breakdown
    exit_counts = {}
    for s in snapshots:
        et = s.get("exit_type", "unknown")
        exit_counts[et] = exit_counts.get(et, {"win": 0, "loss": 0})
        exit_counts[et][s["outcome"]] += 1
    print(f"\n  Exit type breakdown:")
    for et, counts in sorted(exit_counts.items()):
        total = counts["win"] + counts["loss"]
        wr2 = counts["win"] / total * 100 if total else 0
        print(f"    {et:<22} {total:>3} trades  WR={wr2:.0f}%")

    print(f"\n{SEP}\n")


def run_analysis(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    save: bool = True,
) -> dict:
    print(f"Running backtest: {strategy_name}/{symbol}/{period} ...")
    snapshots = extract_trade_snapshots(strategy_name, symbol, period)
    if len(snapshots) < 5:
        print(f"  Skipping — only {len(snapshots)} trades (need ≥5)")
        return {}

    analysis = analyze_snapshots(snapshots)
    _print_report(strategy_name, symbol, period, snapshots, analysis)

    if save:
        os.makedirs("data", exist_ok=True)
        tag = f"{strategy_name}_{symbol}"

        # CSV: one row per trade
        df = pd.DataFrame(snapshots)
        csv_path = f"data/trade_analysis_{tag}.csv"
        df.to_csv(csv_path, index=False)

        # JSON: analysis summary (for optimizer)
        json_path = f"data/trade_analysis_{tag}.json"
        out = {
            "strategy": strategy_name, "symbol": symbol, "period": period,
            "n_trades": len(snapshots),
            "n_wins":   sum(1 for s in snapshots if s["outcome"] == "win"),
            "win_rate": round(sum(1 for s in snapshots if s["outcome"] == "win") / len(snapshots) * 100, 1),
            "feature_analysis": analysis,
            "snapshots": snapshots,
        }
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2, default=str)

        print(f"  Saved: {csv_path}")
        print(f"  Saved: {json_path}")

    return {"snapshots": snapshots, "analysis": analysis}


def main():
    parser = argparse.ArgumentParser(description="Perplexity trade analyzer")
    parser.add_argument("--strategy", default=None, help="Strategy name")
    parser.add_argument("--symbol",   default=None, help="Symbol e.g. NVDA")
    parser.add_argument("--period",   default="5y", help="Backtest period (default 5y)")
    parser.add_argument("--all",      action="store_true", help="Run all strategy x symbol combos")
    args = parser.parse_args()

    if args.all:
        for sname in STRATEGIES:
            for sym in DEFAULT_SYMBOLS:
                try:
                    run_analysis(sname, sym, args.period)
                except Exception as e:
                    print(f"  ERROR {sname}/{sym}: {e}")
    else:
        if not args.strategy or not args.symbol:
            parser.error("Provide --strategy and --symbol, or use --all")
        if args.strategy not in STRATEGIES:
            parser.error(f"Unknown strategy. Choose from: {list(STRATEGIES)}")
        run_analysis(args.strategy, args.symbol.upper(), args.period)


if __name__ == "__main__":
    main()
