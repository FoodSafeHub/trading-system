#!/usr/bin/env python3
"""
Pre-market scanner CLI — prints a ranked watchlist for today's session.

Usage:
    python scripts/run_scanner.py
    python scripts/run_scanner.py --max-symbols 10
    python scripts/run_scanner.py --universe "SPY,QQQ,IWM,TSLA,NVDA,AAPL,MSFT,AMZN,META"
    python scripts/run_scanner.py --universe "TSLA,NVDA,AAPL" --market-state TREND_UP
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure the repo root is on the path when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.services.strategy.daytrading.scanners import (
    DayTradingScanner,
    DayTradingScannerConfig,
)


def _fmt_vol(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(int(v))


def _fmt_tags(tags: list[str], max_tags: int = 3) -> str:
    shown = tags[:max_tags]
    rest = len(tags) - max_tags
    s = ", ".join(shown)
    if rest > 0:
        s += f" +{rest}"
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily pre-market scanner for the day-trading bot."
    )
    parser.add_argument("--max-symbols", type=int, default=20, help="Max watchlist size")
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        help='Comma-separated symbol override, e.g. "TSLA,NVDA,AAPL"',
    )
    parser.add_argument(
        "--market-state",
        type=str,
        default="",
        help="Force market state: TREND_UP | TREND_DOWN | CHOPPY | HIGH_VOL | NEWS_RISK",
    )
    parser.add_argument("--min-atr", type=float, default=1.0)
    parser.add_argument("--max-atr", type=float, default=8.0)
    parser.add_argument("--min-volume", type=float, default=1_000_000)
    args = parser.parse_args()

    # Build config from CLI args
    cfg = DayTradingScannerConfig(
        min_atr_pct=args.min_atr,
        max_atr_pct=args.max_atr,
        min_avg_volume=args.min_volume,
    )

    universe = (
        [s.strip().upper() for s in args.universe.split(",") if s.strip()]
        if args.universe
        else None
    )

    print(f"\nRunning pre-market scanner ({time.strftime('%Y-%m-%d %H:%M:%S ET')})")
    print(f"Universe : {', '.join(universe) if universe else 'default'}")
    print(f"Max syms : {args.max_symbols}")
    if args.market_state:
        print(f"State    : {args.market_state} (forced)")
    print()

    scanner = DayTradingScanner(config=cfg, universe=universe)
    market_state = args.market_state.strip() or None

    t0 = time.time()
    watchlist = scanner.get_intraday_watchlist(
        max_symbols=args.max_symbols,
        market_state=market_state,
    )
    elapsed = time.time() - t0

    if not watchlist:
        print("No symbols passed filters.")
        return

    # ── Table header ──────────────────────────────────────────────────────────
    HDR = f"{'Symbol':<7}  {'Score':>5}  {'Adj':>5}  {'Bucket':<10}  {'Tags':<30}  {'Gap%':>6}  {'RelVol':>7}  {'ATR%':>5}  {'AvgVol':>8}"
    sep = "-" * len(HDR)
    print(HDR)
    print(sep)

    for r in watchlist:
        m = r.metrics
        gap_str = f"{m.premarket_gap_pct:+.1f}%"
        rv_str = f"{m.premarket_rel_vol:.1%}"
        row = (
            f"{r.symbol:<7}  "
            f"{r.score:>5.3f}  "
            f"{r.adjusted_score:>5.3f}  "
            f"{r.recommended_strategy_bucket:<10}  "
            f"{_fmt_tags(r.tags):<30}  "
            f"{gap_str:>6}  "
            f"{rv_str:>7}  "
            f"{m.atr_pct:>4.1f}%  "
            f"{_fmt_vol(m.avg_daily_volume_30d):>8}"
        )
        suffix = "  *" if m.has_catalyst else ""
        print(row + suffix)

    print(sep)
    print(f"\n{len(watchlist)} symbols | scan took {elapsed:.1f}s")
    if any(r.metrics.has_catalyst for r in watchlist):
        print("  * = catalyst detected (elevated risk)")

    # Show skipped symbols at -v (stub — always show top rejects for now)
    print()


if __name__ == "__main__":
    main()
