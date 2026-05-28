#!/usr/bin/env python
"""
Exit-timing analysis for the v2 (regime-aware) strategies.

Question this answers: "Are we selling too early? How long does a stock run
before it actually reverses?"

For each assigned symbol x its live strategy config (resolved exactly as the
scheduler/scanner does, via _make_generic_configs_full so calibrated param
overrides are included), we run the backtest engine and then, for every
completed round-trip, measure:

  - hold_bars                : how many bars the position was held
  - realized_pct             : the P&L we actually captured
  - run_after_5/10/20        : the MAX favorable move in the N bars AFTER our SELL
                               (how much more the stock ran once we were out)
  - peak_to_exit_pct         : highest close during the hold vs our exit price
                               (did we exit near the top of the move, or give it back?)

Output is written to a UTF-8 file (console here is cp1252 and chokes on arrows).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
from app.services.market_data.provider import get_ohlcv
from app.services.backtest.engine import run_backtest
from app.services.scanner.scanner_service import _make_generic_configs_full

PERIOD = "5y"
OUT = "data/exit_timing_report.txt"

# (symbol, live strategy_name) pairs to study. Drawn from current enabled
# assignments; we resolve each via _make_generic_configs_full so the params
# match live. US names with long history -> meaningful sample sizes.
TARGETS = [
    ("NVDA", "NVDA_RSI2_Mean_Reversion"),
    ("TSLA", "Legacy_TSLA_Fib_Pullback"),
    ("FANG", "FANG_Pullback_EMA50"),
    ("XOM",  "XOM_Pullback_EMA50"),
    ("COP",  "COP_Pullback_EMA50"),
    ("SO",   "SO_Pullback_EMA50"),
    ("BWA",  "BWA_Pullback_EMA50"),
    ("ARKG", "ARKG_EMA_MACD_Crossover"),
    ("TOST", "TOST_RSI2_Mean_Reversion"),
    ("AVPT", "AVPT_RSI2_Mean_Reversion"),
    ("ARLO", "ARLO_VIX_Spike_Reversal"),
    ("COLL", "COLL_VIX_Spike_Reversal"),
]


def _mean(v):
    return round(statistics.mean(v), 2) if v else None


def _median(v):
    return round(statistics.median(v), 2) if v else None


def analyze_one(symbol: str, strategy_name: str, lines: list[str]) -> dict | None:
    cfgs = _make_generic_configs_full(symbol)
    cfg = next((c for c in cfgs if c.name == strategy_name), None)
    if cfg is None:
        lines.append(f"  {symbol}/{strategy_name}: no config resolved, skipping")
        return None

    df = get_ohlcv(symbol, period=PERIOD)
    if df is None or df.empty or len(df) < 60:
        lines.append(f"  {symbol}/{strategy_name}: insufficient data ({0 if df is None else len(df)} bars)")
        return None

    res = run_backtest(
        strategy_name=strategy_name,
        symbol=symbol,
        strategy_type=cfg.type,
        params=cfg.params,
        period=PERIOD,
        quantity=0,          # use all capital -> clean per-trade pct
        df=df,
    )

    closes = df["Close"].dropna().reset_index(drop=True)
    dates = [str(d)[:10] for d in df.index]
    date_to_iloc = {d: i for i, d in enumerate(dates)}

    buys = [t for t in res.trades if t.side == "BUY"]
    sells = [t for t in res.trades if "SELL" in t.side]

    rows = []
    for b, s in zip(buys, sells):
        bi = date_to_iloc.get(b.date)
        si = date_to_iloc.get(s.date)
        if bi is None or si is None or si <= bi:
            continue
        hold = si - bi
        realized = (s.price - b.price) / b.price * 100.0

        # Peak during the hold (did we exit near the local top?)
        hold_slice = closes.iloc[bi:si + 1]
        peak_in_hold = float(hold_slice.max())
        peak_to_exit = (peak_in_hold - s.price) / s.price * 100.0

        # How much further did it run AFTER we sold?
        def run_after(n):
            end = min(si + n, len(closes) - 1)
            if end <= si:
                return 0.0
            fut = closes.iloc[si + 1:end + 1]
            if fut.empty:
                return 0.0
            return (float(fut.max()) - s.price) / s.price * 100.0

        rows.append({
            "entry": b.date, "exit": s.date,
            "hold": hold, "realized": round(realized, 2),
            "peak_to_exit": round(peak_to_exit, 2),
            "after5": round(run_after(5), 2),
            "after10": round(run_after(10), 2),
            "after20": round(run_after(20), 2),
            "win": realized > 0,
        })

    if not rows:
        lines.append(f"  {symbol}/{strategy_name}: 0 completed round-trips")
        return None

    n = len(rows)
    wins = [r for r in rows if r["win"]]
    wr = len(wins) / n * 100

    summary = {
        "symbol": symbol, "strategy": strategy_name, "type": cfg.type,
        "n": n, "wr": round(wr, 1), "total_return": round(res.total_return_pct, 1),
        "avg_hold": _mean([r["hold"] for r in rows]),
        "med_hold": _median([r["hold"] for r in rows]),
        "avg_realized": _mean([r["realized"] for r in rows]),
        "avg_win": _mean([r["realized"] for r in wins]),
        "avg_loss": _mean([r["realized"] for r in rows if not r["win"]]),
        # The "sold too early" signal: average extra run after we exited
        "avg_after5": _mean([r["after5"] for r in rows]),
        "avg_after10": _mean([r["after10"] for r in rows]),
        "avg_after20": _mean([r["after20"] for r in rows]),
        # On winners only — the part we care about (cutting winners short)
        "win_after10": _mean([r["after10"] for r in wins]),
        "win_after20": _mean([r["after20"] for r in wins]),
        "avg_peak_to_exit": _mean([r["peak_to_exit"] for r in rows]),
    }

    lines.append("")
    lines.append("=" * 92)
    lines.append(f"  {symbol}  /  {strategy_name}  ({cfg.type})  [{PERIOD}]")
    lines.append("-" * 92)
    lines.append(
        f"  Trades: {n}  WinRate: {summary['wr']}%  TotalRet: {summary['total_return']}%  "
        f"AvgHold: {summary['avg_hold']} bars (median {summary['med_hold']})"
    )
    lines.append(
        f"  AvgRealized: {summary['avg_realized']}%   AvgWin: {summary['avg_win']}%   "
        f"AvgLoss: {summary['avg_loss']}%"
    )
    lines.append(
        f"  Gave-back during hold (peak vs exit): {summary['avg_peak_to_exit']}%  "
        f"(how far below the in-hold peak we exited)"
    )
    lines.append(
        f"  Ran AFTER our SELL (all trades):  +5b {summary['avg_after5']}%   "
        f"+10b {summary['avg_after10']}%   +20b {summary['avg_after20']}%"
    )
    lines.append(
        f"  Ran AFTER our SELL (WINNERS only): +10b {summary['win_after10']}%   "
        f"+20b {summary['win_after20']}%   <-- 'sold too early' cost on winners"
    )

    # Per-trade detail (most recent 8)
    lines.append(f"  {'entry':<11}{'exit':<11}{'hold':>5}{'realiz%':>9}{'after10%':>10}{'after20%':>10}")
    for r in rows[-8:]:
        lines.append(
            f"  {r['entry']:<11}{r['exit']:<11}{r['hold']:>5}{r['realized']:>9}"
            f"{r['after10']:>10}{r['after20']:>10}"
        )

    return summary


def main():
    lines: list[str] = []
    lines.append("EXIT-TIMING ANALYSIS  (v2 regime-aware strategies, live param configs)")
    lines.append(f"Period: {PERIOD}   Question: are we selling too early?")
    summaries = []
    for sym, sname in TARGETS:
        try:
            s = analyze_one(sym, sname, lines)
            if s:
                summaries.append(s)
        except Exception as e:
            lines.append(f"  {sym}/{sname}: ERROR {type(e).__name__}: {e}")

    # Aggregate
    if summaries:
        lines.append("")
        lines.append("#" * 92)
        lines.append("  AGGREGATE")
        lines.append("#" * 92)
        all_after10 = [s["avg_after10"] for s in summaries if s["avg_after10"] is not None]
        all_win_after20 = [s["win_after20"] for s in summaries if s["win_after20"] is not None]
        all_hold = [s["avg_hold"] for s in summaries if s["avg_hold"] is not None]
        all_wr = [s["wr"] for s in summaries]
        all_peak = [s["avg_peak_to_exit"] for s in summaries if s["avg_peak_to_exit"] is not None]
        lines.append(f"  Strategies analyzed: {len(summaries)}")
        lines.append(f"  Mean avg-hold across strategies: {_mean(all_hold)} bars")
        lines.append(f"  Mean win-rate: {_mean(all_wr)}%")
        lines.append(f"  Mean give-back (peak-to-exit): {_mean(all_peak)}%")
        lines.append(f"  Mean run-after-SELL (+10b, all trades): {_mean(all_after10)}%")
        lines.append(f"  Mean run-after-SELL (+20b, WINNERS): {_mean(all_win_after20)}%")

        # Rank by 'most left on the table' on winners
        ranked = sorted(
            [s for s in summaries if s["win_after20"] is not None],
            key=lambda x: x["win_after20"], reverse=True,
        )
        lines.append("")
        lines.append("  Most money left on the table after selling winners (+20 bars):")
        for s in ranked[:8]:
            lines.append(
                f"    {s['symbol']:<6} {s['type']:<22} left +{s['win_after20']}%  "
                f"(avg hold {s['avg_hold']}b, WR {s['wr']}%)"
            )

    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"WROTE {OUT}  ({len(lines)} lines, {len(summaries)} strategies)")


if __name__ == "__main__":
    main()
