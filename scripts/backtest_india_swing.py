#!/usr/bin/env python
"""Backtest the new India swing strategies + the 2 candle keepers across the
Nifty 100 + midcap universe.

Uses the production perplexity_engine (India-routes regime via is_india_symbol)
with the after-costs India cost model, so the verdict matches the live harness.

Usage:
    python scripts/backtest_india_swing.py [--period 5y] [--limit N]

Outpus an aggregated per-strategy table and applies the KEEP gates:
    net > 0 (costed), >= 20 trades, >= 50% win rate.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.markets import NIFTY_50, NIFTY_NEXT_50, NIFTY_MIDSMALL_EXTRA
from app.services.market_data.provider import get_ohlcv
from app.services.backtest.costs import INDIA_DEFAULT
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.strategy.perplexity.india_swing_strategies import (
    NiftyLeaderPullback, FiftyTwoWeekHighBreakout, VcpContractionBreakout,
)
from app.services.strategy.perplexity.momentum_strategies import (
    PerpThreeBarPush, PerpHammerShootingStar,
)
from app.services.strategy.perplexity.india_advanced_strategies import (
    MomentumBreakout, TrendPullbackEma, TrendFollowingHHHL,
    SupportResistanceBounce, WyckoffSpringTest,
)

STRATEGIES = [
    # earlier India swing set
    NiftyLeaderPullback(),
    FiftyTwoWeekHighBreakout(),
    VcpContractionBreakout(),
    PerpThreeBarPush(),
    PerpHammerShootingStar(),
    # five spec-driven advanced strategies
    MomentumBreakout(),
    TrendPullbackEma(),
    TrendFollowingHHHL(),
    SupportResistanceBounce(),
    WyckoffSpringTest(),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--limit", type=int, default=0, help="cap universe size for a quick run")
    ap.add_argument("--approach-c", action="store_true", help="arm tight trail on SELL signals")
    ap.add_argument("--trail", type=float, default=3.0, help="Approach C trail %% (with --approach-c)")
    args = ap.parse_args()

    universe = NIFTY_50 + NIFTY_NEXT_50 + NIFTY_MIDSMALL_EXTRA
    universe = sorted(set(universe))
    if args.limit:
        universe = universe[: args.limit]

    cost_model = INDIA_DEFAULT
    _cmode = f"Approach C {args.trail:.1f}% trail" if args.approach_c else "default (market exit)"
    print(f"Universe: {len(universe)} names | period {args.period} | costs: INDIA_DEFAULT | exit: {_cmode}")

    # Pre-fetch ^NSEI / ^INDIAVIX once and share across all runs.
    try:
        nsei = get_ohlcv("^NSEI", period="10y")["Close"]
    except Exception:
        nsei = None
    try:
        vix = get_ohlcv("^INDIAVIX", period="10y")["Close"]
    except Exception:
        vix = None

    agg = defaultdict(lambda: {"net": 0.0, "trades": 0, "wins": 0,
                               "gross_win": 0.0, "gross_loss": 0.0, "symbols": 0})

    for n, sym in enumerate(universe, 1):
        try:
            df = get_ohlcv(sym, period=args.period)
        except Exception as exc:
            print(f"  [{n}/{len(universe)}] {sym}: data error — {exc}")
            continue
        if df is None or df.empty or len(df) < 280:
            print(f"  [{n}/{len(universe)}] {sym}: insufficient bars")
            continue

        for strat in STRATEGIES:
            try:
                r = run_perplexity_backtest(
                    strat, sym, period=args.period,
                    df_full=df, mom_index_close=nsei, mom_vix_close=vix,
                    cost_model=cost_model,
                    approach_c=args.approach_c, tight_trail_pct=args.trail,
                )
            except Exception:
                continue
            a = agg[strat.name]
            a["net"] += r.total_pnl
            a["trades"] += r.total_trades
            a["wins"] += r.winning_trades
            if r.total_trades:
                a["symbols"] += 1
            for tp in r.trade_pairs:
                pnl = tp.get("pnl") or 0.0
                if pnl > 0:
                    a["gross_win"] += pnl
                else:
                    a["gross_loss"] += abs(pnl)
        print(f"  [{n}/{len(universe)}] {sym} done", flush=True)

    print("\n" + "=" * 92)
    print(f"{'Strategy':<26} | {'Net P&L':>12} | {'Trades':>6} | {'WR%':>5} | "
          f"{'PF':>5} | {'Syms':>4} | Verdict")
    print("-" * 92)
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["net"]):
        wr = a["wins"] / a["trades"] * 100 if a["trades"] else 0.0
        pf = a["gross_win"] / a["gross_loss"] if a["gross_loss"] > 0 else float("inf")
        keep = a["net"] > 0 and a["trades"] >= 20 and wr >= 50.0
        verdict = "KEEP" if keep else "RETIRE/RESEARCH"
        print(f"{name:<26} | {a['net']:>12,.0f} | {a['trades']:>6} | {wr:>4.0f}% | "
              f"{pf:>5.2f} | {a['symbols']:>4} | {verdict}")
    print("=" * 92)


if __name__ == "__main__":
    main()
