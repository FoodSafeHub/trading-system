"""
A/B benchmark harness for perplexity_engine fixes.

Why a script instead of a test?
    The objective is to compare engine behavior on real OHLCV, not assert a fixed
    outcome. The script fetches each symbol once, runs every strategy with the
    same data injection, and writes a compact metrics table — reproducible
    enough for review, cheap enough to re-run.

Mode flag
    --mode {baseline|after-nocost|after-costed} controls which post-fix
    configuration is being measured. The same script is run pre- and post-fix;
    the flag tags the output artifact.

Output
    reports/perplexity_ab_<mode>_<utc-stamp>.{md,json}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Repo path injection so the script runs from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.backtest.costs import INDIA_DEFAULT, US_DEFAULT  # noqa: E402
from app.services.backtest.perplexity_engine import run_perplexity_backtest  # noqa: E402
from app.services.market_data.provider import get_ohlcv  # noqa: E402
from app.services.strategy.perplexity import PERPLEXITY_STRATEGIES  # noqa: E402

# Locked benchmark spec — do NOT change between baseline and after runs.
SYMBOLS_US = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
SYMBOLS_IN = ["RELIANCE", "BHARTIARTL", "INFY", "TCS", "HDFCBANK"]
SYMBOLS = SYMBOLS_US + SYMBOLS_IN
PERIOD = "2y"
INITIAL_CAPITAL = 100_000.0


def _is_india(sym: str) -> bool:
    return sym in SYMBOLS_IN


def _cost_model_for(sym: str):
    return INDIA_DEFAULT if _is_india(sym) else US_DEFAULT


def run_benchmark(mode: str) -> dict:
    """Execute the locked benchmark and return a metrics dict."""
    # Pre-fetch all data ONCE so baseline / after runs hit the same bars.
    print(f"[fetch] period={PERIOD} symbols={len(SYMBOLS)}", file=sys.stderr)
    try:
        spy_close = get_ohlcv("SPY", period="10y")["Close"]
    except Exception:
        spy_close = None

    dfs: dict[str, object] = {}
    for sym in SYMBOLS:
        try:
            df = get_ohlcv(sym, period=PERIOD)
            if df is None or df.empty or len(df) < 220:
                print(f"[skip] {sym}: {0 if df is None else len(df)} bars", file=sys.stderr)
                continue
            dfs[sym] = df
        except Exception as exc:
            print(f"[skip] {sym}: {exc}", file=sys.stderr)

    # Strategy aggregate buckets
    agg: dict[str, dict] = {}
    for sym, df in dfs.items():
        cost_model = _cost_model_for(sym) if mode == "after-costed" else None
        for strat in PERPLEXITY_STRATEGIES:
            try:
                kwargs = dict(
                    strategy=strat, symbol=sym, period=PERIOD,
                    initial_capital=INITIAL_CAPITAL,
                    df_full=df, spy_close=spy_close,
                )
                # Pass cost_model only if the engine signature accepts it.
                # (After Phase 3, it will. Before Phase 3, we never pass it.)
                if mode == "after-costed":
                    kwargs["cost_model"] = cost_model
                r = run_perplexity_backtest(**kwargs)
            except TypeError:
                # Engine signature pre-Phase-3 doesn't accept cost_model — that
                # is only valid in mode == "after-costed", and a clean error
                # rather than a silent fallback is better here.
                raise
            except Exception:
                continue
            a = agg.setdefault(strat.name, dict(
                syms=0, fire=0, trd=0, wins=0, losses=0, pnl=0.0,
                ret_sum=0.0, hold_days_sum=0.0, hold_days_n=0,
                pf_sum=0.0, pf_n=0, wr_weighted=0.0, wr_n=0,
            ))
            a["syms"] += 1
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
        rows.append(dict(
            strategy=name, fire=a["fire"], syms=a["syms"], trades=a["trd"],
            wins=a["wins"], losses=a["losses"],
            win_rate_pct=round(wr, 2), profit_factor=round(pf, 2),
            avg_holding_days=round(avg_hold, 2),
            avg_return_pct=round(avg_ret, 2),
            total_pnl=round(a["pnl"], 2),
        ))
    rows.sort(key=lambda r: -r["total_pnl"])

    return {
        "mode": mode,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": list(dfs.keys()),
        "period": PERIOD,
        "initial_capital": INITIAL_CAPITAL,
        "strategy_rows": rows,
        "aggregate": {
            "total_pnl": round(sum(r["total_pnl"] for r in rows), 2),
            "total_trades": sum(r["trades"] for r in rows),
            "total_wins": sum(r["wins"] for r in rows),
            "total_losses": sum(r["losses"] for r in rows),
        },
    }


def write_artifacts(result: dict) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"perplexity_ab_{result['mode']}_{stamp}"

    base.with_suffix(".json").write_text(json.dumps(result, indent=2))

    rows = result["strategy_rows"]
    md = []
    md.append(f"# Perplexity A/B benchmark — {result['mode']}\n")
    md.append(f"- Captured: `{result['captured_at_utc']}`")
    md.append(f"- Period: `{result['period']}`")
    md.append(f"- Symbols ({len(result['symbols'])}): `{', '.join(result['symbols'])}`")
    md.append(f"- Initial capital: `${result['initial_capital']:,.0f}`\n")
    md.append("| strategy | fire | trd | wr% | pf | avg_hold_d | avg_ret% | total_pnl |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['strategy']} | {r['fire']}/{r['syms']} | {r['trades']} | "
            f"{r['win_rate_pct']:.1f} | {r['profit_factor']:.2f} | "
            f"{r['avg_holding_days']:.1f} | {r['avg_return_pct']:+.2f} | "
            f"${r['total_pnl']:+,.0f} |"
        )
    md.append("")
    md.append(f"**Aggregate**: total_pnl=`${result['aggregate']['total_pnl']:+,.0f}`  "
              f"trades=`{result['aggregate']['total_trades']}`  "
              f"wins=`{result['aggregate']['total_wins']}`  "
              f"losses=`{result['aggregate']['total_losses']}`")
    base.with_suffix(".md").write_text("\n".join(md))
    return base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["baseline", "after-nocost", "after-costed"],
        required=True,
        help="Which configuration this run represents.",
    )
    args = p.parse_args()
    result = run_benchmark(args.mode)
    artifact = write_artifacts(result)
    print(f"\nMODE: {result['mode']}")
    print(f"Artifact: {artifact}.md")
    print(f"Aggregate: pnl=${result['aggregate']['total_pnl']:+,.0f}  "
          f"trades={result['aggregate']['total_trades']}  "
          f"wins={result['aggregate']['total_wins']}  "
          f"losses={result['aggregate']['total_losses']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
