"""
5-year benchmark targeted at the 6 RESEARCH-ONLY / NEEDS-FOLLOW-UP strategies.

Goal: with short-side support now wired and a longer window, accumulate enough
trades to finalize KEEP / RETIRE / still-RESEARCH-ONLY verdicts for each.

Locked spec:
  symbols      : same 10 (5 US + 5 NSE) as the prior A/B harness
  period       : 5y (was 2y)
  strategies   : the 6 below
  data         : pre-fetched once per symbol, injected into each strategy run
  cost models  : US_DEFAULT for US tickers, INDIA_DEFAULT for NSE tickers
  short path   : now active (sees direction="SELL" entries from candle patterns)

Output:
  reports/perplexity_research_5y_<stamp>.{md,json}

Run via:
  python scripts/perplexity_research_5y_benchmark.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.backtest.costs import INDIA_DEFAULT, US_DEFAULT  # noqa: E402
from app.services.backtest.perplexity_engine import run_perplexity_backtest  # noqa: E402
from app.services.market_data.provider import get_ohlcv  # noqa: E402
from app.services.strategy.perplexity import PERPLEXITY_STRATEGIES  # noqa: E402

SYMBOLS_US = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
SYMBOLS_IN = ["RELIANCE", "BHARTIARTL", "INFY", "TCS", "HDFCBANK"]
SYMBOLS = SYMBOLS_US + SYMBOLS_IN
PERIOD = "5y"
INITIAL_CAPITAL = 100_000.0

# The 6 strategies whose verdicts were deferred in the decision pass.
TARGET_STRATEGIES = {
    "Breakout_Consolidation",
    "RSI_Swing_Reversal",
    "Daily_NR_Breakout",
    "Daily_Hammer_Star",
    "Daily_Engulfing_Volume",
    "Daily_Three_Bar_Push",
}


def _cost_model_for(sym: str):
    return INDIA_DEFAULT if sym in SYMBOLS_IN else US_DEFAULT


def run() -> dict:
    print(f"[fetch] period={PERIOD} symbols={len(SYMBOLS)}", file=sys.stderr)
    try:
        spy_close = get_ohlcv("SPY", period="10y")["Close"]
    except Exception:
        spy_close = None

    dfs = {}
    for sym in SYMBOLS:
        try:
            df = get_ohlcv(sym, period=PERIOD)
            if df is None or df.empty or len(df) < 220:
                print(f"[skip] {sym}: {0 if df is None else len(df)} bars", file=sys.stderr)
                continue
            dfs[sym] = df
        except Exception as exc:
            print(f"[skip] {sym}: {exc}", file=sys.stderr)

    target_strats = [s for s in PERPLEXITY_STRATEGIES if s.name in TARGET_STRATEGIES]

    agg: dict[str, dict] = {}
    # Also collect side breakdown to see short-side contribution.
    side_breakdown: dict[str, dict[str, int]] = {}
    for sym, df in dfs.items():
        cost_model = _cost_model_for(sym)
        for strat in target_strats:
            try:
                r = run_perplexity_backtest(
                    strategy=strat, symbol=sym, period=PERIOD,
                    initial_capital=INITIAL_CAPITAL,
                    df_full=df, spy_close=spy_close,
                    cost_model=cost_model,
                )
            except Exception as exc:
                print(f"[err] {strat.name} {sym}: {exc}", file=sys.stderr)
                continue

            a = agg.setdefault(strat.name, dict(
                syms=0, fire=0, trd=0, wins=0, losses=0, pnl=0.0,
                ret_sum=0.0, hold_days_sum=0.0, hold_days_n=0,
                pf_sum=0.0, pf_n=0, wr_weighted=0.0, wr_n=0,
            ))
            sb = side_breakdown.setdefault(strat.name, dict(long_entries=0, short_entries=0))
            a["syms"] += 1
            # Count entry sides regardless of trade outcome
            for t in r.trades:
                if t["side"] == "BUY":
                    sb["long_entries"] += 1
                elif t["side"] == "SHORT":
                    sb["short_entries"] += 1
            if r.total_trades > 0:
                a["fire"] += 1
                a["trd"] += r.total_trades
                a["wins"] += r.winning_trades
                a["losses"] += r.losing_trades
                a["pnl"] += r.total_pnl
                a["ret_sum"] += r.total_return_pct
                a["wr_weighted"] += r.win_rate_pct * r.total_trades
                a["wr_n"] += r.total_trades
                if r.profit_factor and r.profit_factor not in (float("inf"),):
                    a["pf_sum"] += r.profit_factor
                    a["pf_n"] += 1
                if r.average_holding_days:
                    a["hold_days_sum"] += r.average_holding_days * r.total_trades
                    a["hold_days_n"] += r.total_trades

    rows = []
    for name, a in agg.items():
        wr = a["wr_weighted"] / a["wr_n"] if a["wr_n"] > 0 else 0.0
        pf = a["pf_sum"] / a["pf_n"] if a["pf_n"] > 0 else 0.0
        avg_ret = a["ret_sum"] / a["fire"] if a["fire"] > 0 else 0.0
        avg_hold = a["hold_days_sum"] / a["hold_days_n"] if a["hold_days_n"] > 0 else 0.0
        sb = side_breakdown.get(name, {})
        rows.append(dict(
            strategy=name, fire=a["fire"], syms=a["syms"], trades=a["trd"],
            wins=a["wins"], losses=a["losses"],
            win_rate_pct=round(wr, 2), profit_factor=round(pf, 2),
            avg_holding_days=round(avg_hold, 2),
            avg_return_pct=round(avg_ret, 2),
            total_pnl=round(a["pnl"], 2),
            long_entries=sb.get("long_entries", 0),
            short_entries=sb.get("short_entries", 0),
        ))
    rows.sort(key=lambda r: -r["total_pnl"])

    return {
        "mode": "5y-research-only",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": list(dfs.keys()),
        "period": PERIOD,
        "initial_capital": INITIAL_CAPITAL,
        "strategy_rows": rows,
        "aggregate": {
            "total_pnl": round(sum(r["total_pnl"] for r in rows), 2),
            "total_trades": sum(r["trades"] for r in rows),
            "total_long_entries": sum(r["long_entries"] for r in rows),
            "total_short_entries": sum(r["short_entries"] for r in rows),
        },
    }


def write_artifact(result: dict) -> Path:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"perplexity_research_5y_{stamp}"

    base.with_suffix(".json").write_text(json.dumps(result, indent=2))

    rows = result["strategy_rows"]
    md = []
    md.append(f"# Perplexity 5y benchmark -- RESEARCH-ONLY strategies\n")
    md.append(f"- Captured: `{result['captured_at_utc']}`")
    md.append(f"- Period: `{result['period']}`")
    md.append(f"- Symbols ({len(result['symbols'])}): `{', '.join(result['symbols'])}`")
    md.append(f"- Initial capital: `${result['initial_capital']:,.0f}`")
    md.append(f"- Short-side support: ACTIVE\n")
    md.append("| strategy | fire | trd | wr% | pf | hold_d | long_e | short_e | total_pnl |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['strategy']} | {r['fire']}/{r['syms']} | {r['trades']} | "
            f"{r['win_rate_pct']:.1f} | {r['profit_factor']:.2f} | "
            f"{r['avg_holding_days']:.1f} | {r['long_entries']} | "
            f"{r['short_entries']} | ${r['total_pnl']:+,.0f} |"
        )
    md.append("")
    agg = result["aggregate"]
    md.append(
        f"**Aggregate**: total_pnl=`${agg['total_pnl']:+,.0f}` "
        f"trades=`{agg['total_trades']}` long_entries=`{agg['total_long_entries']}` "
        f"short_entries=`{agg['total_short_entries']}`"
    )
    base.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return base


def main() -> int:
    result = run()
    artifact = write_artifact(result)
    print(f"\nArtifact: {artifact}.md")
    agg = result["aggregate"]
    print(
        f"Aggregate: pnl=${agg['total_pnl']:+,.0f} "
        f"trades={agg['total_trades']} "
        f"long_entries={agg['total_long_entries']} "
        f"short_entries={agg['total_short_entries']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
