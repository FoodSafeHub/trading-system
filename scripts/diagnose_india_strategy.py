#!/usr/bin/env python
"""Diagnose WHY an India strategy loses: exit-reason breakdown, regime mix, and
cost drag across a sample of liquid Nifty names.

The aggregate verdict table tells us a strategy is unprofitable; this tells us
the mechanism — are trades dying on stops, giving back gains on the trail,
timing out, or never reaching target? Run before tuning.

Usage:
    python scripts/diagnose_india_strategy.py India_Leader_Pullback [--period 5y] [--limit 30]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.markets import NIFTY_50, NIFTY_NEXT_50
from app.services.market_data.provider import get_ohlcv
from app.services.backtest.costs import INDIA_DEFAULT
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES

_BY_NAME = {s.name: s for s in PERPLEXITY_STRATEGIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    strat = _BY_NAME.get(args.strategy)
    if strat is None:
        print(f"Unknown strategy. Options:\n  " + "\n  ".join(sorted(_BY_NAME)))
        return

    universe = sorted(set(NIFTY_50 + NIFTY_NEXT_50))[: args.limit]
    try:
        nsei = get_ohlcv("^NSEI", period="10y")["Close"]
    except Exception:
        nsei = None
    try:
        vix = get_ohlcv("^INDIAVIX", period="10y")["Close"]
    except Exception:
        vix = None

    exit_reasons = Counter()      # 'stop' / 'target' / 'time' / 'signal' / 'close'
    pnl_by_exit = defaultdict(float)
    regime_at_entry = Counter()
    total_net = total_comm = 0.0
    n_trades = wins = 0

    for sym in universe:
        try:
            df = get_ohlcv(sym, period=args.period)
        except Exception:
            continue
        if df is None or df.empty or len(df) < 280:
            continue
        try:
            r = run_perplexity_backtest(strat, sym, period=args.period, df_full=df,
                                        mom_index_close=nsei, mom_vix_close=vix,
                                        cost_model=INDIA_DEFAULT)
        except Exception:
            continue
        total_net += r.total_pnl
        for t in r.trades:
            total_comm += t.get("commission") or 0.0
            if t["side"] in ("BUY", "SHORT"):
                regime_at_entry[t.get("regime", "?")] += 1
            # exit legs carry pnl + a side like "SELL (stop)" / "SELL (time)"
            if ("SELL" in t["side"] or "COVER" in t["side"]) and t.get("pnl") is not None:
                n_trades += 1
                pnl = t["pnl"]
                if pnl > 0:
                    wins += 1
                side = t["side"].lower()
                if "stop" in side:
                    key = "stop"
                elif "target" in side:
                    key = "target"
                elif "time" in side:
                    key = "time"
                elif "close" in side:
                    key = "close(eod)"
                else:
                    key = "signal"
                exit_reasons[key] += 1
                pnl_by_exit[key] += pnl

    print(f"\n=== {strat.name} — {len(universe)} names, {args.period} ===")
    print(f"Net P&L: {total_net:,.0f} | trades: {n_trades} | "
          f"WR: {wins / n_trades * 100 if n_trades else 0:.0f}% | "
          f"total commission: {total_comm:,.0f}")
    print(f"\nExit-reason breakdown:")
    print(f"  {'reason':<12} | {'count':>6} | {'%':>5} | {'net P&L':>12} | {'avg/trade':>10}")
    for k in ("target", "signal", "trail", "time", "stop", "close(eod)"):
        if exit_reasons.get(k):
            c = exit_reasons[k]
            print(f"  {k:<12} | {c:>6} | {c / n_trades * 100:>4.0f}% | "
                  f"{pnl_by_exit[k]:>12,.0f} | {pnl_by_exit[k] / c:>10,.0f}")
    print(f"\nRegime at entry: {dict(regime_at_entry)}")
    print(f"Cost drag: {total_comm:,.0f} on {total_net:,.0f} net "
          f"({abs(total_comm / total_net) * 100 if total_net else 0:.0f}% of |net|)")


if __name__ == "__main__":
    main()
