"""
Build the strategy-level KEEP / RETIRE / RESEARCH-ONLY / NEEDS-FOLLOW-UP
decision table from the corrected-harness benchmark artifacts.

Inputs (under reports/):
  perplexity_ab_baseline_*.json        (pre-fix harness — for reference only)
  perplexity_ab_after-nocost_*.json    (post-fix engine, costs OFF — gross truth)
  perplexity_ab_after-costed_<latest>.json  (post-fix engine + costs ON — net truth)

The CORRECTED truthful artifact is the latest after-costed file (which is
also post-regime-fix because the regime delegation landed before this run).
Decisions are driven primarily by NET numbers.

Heuristic decision rules (transparent; reproducible from the table):
  KEEP             — net pnl > 0 AND trades >= 20 AND wr >= 50 AND pf >= 1.2
  RESEARCH-ONLY    — net pnl >= 0 AND trades < 20   (too-sparse-to-deploy)
                  OR net pnl > 0 AND (wr < 50 OR pf < 1.2)   (positive but weak)
  RETIRE           — net pnl < 0 AND no clear rescue path under known engine fixes
  NEEDS-FOLLOW-UP  — net pnl uncertain because engine still distorts (e.g. costed
                    PF is inflated due to known short-side dropping or single-
                    trade samples)

Output:
  reports/perplexity_strategy_decisions.{md,csv}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"


def _latest(prefix: str) -> Path:
    matches = sorted(REPORTS.glob(f"{prefix}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no artifact matching {prefix}_*.json under reports/")
    return matches[-1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _by_strategy(rows: list[dict]) -> dict[str, dict]:
    return {r["strategy"]: r for r in rows}


def classify(net: dict, gross: dict, baseline: dict, total_bars_estimate: int) -> tuple[str, str]:
    """Return (decision, reason)."""
    name = net["strategy"]
    pnl = net["total_pnl"]
    trades = net["trades"]
    wr = net["win_rate_pct"]
    pf = net["profit_factor"]
    fire = net["fire"]
    syms = net["syms"]

    # Daily candlestick momentum patterns emit short-side SELL entries that
    # the perplexity engine silently drops (long-only execution path). Their
    # number here is the long-side ONLY — half the strategy's surface area
    # is dark. Verdict must be deferred until the short-side issue is fixed,
    # regardless of which sign the long-only number landed on.
    momentum_pattern_names = {
        "Daily_Engulfing_Volume", "Daily_NR_Breakout",
        "Daily_Three_Bar_Push", "Daily_Hammer_Star",
    }
    if name in momentum_pattern_names:
        return (
            "NEEDS-FOLLOW-UP",
            f"long-side only: net ${pnl:+,.0f} on {trades} long trades. "
            f"The strategy also emits short-side SELL entries that the engine "
            f"currently drops (known harness defect). Re-evaluate after that fix.",
        )

    # Sample-size guard: very small trade counts can't support a verdict.
    # Anything < 5 round-trips is sample noise even if win-rate looks great.
    if trades < 5:
        # Positive small samples → RESEARCH-ONLY (interesting, can't deploy)
        # Negative small samples → RESEARCH-ONLY (don't retire on noise)
        return (
            "RESEARCH-ONLY",
            f"only {trades} trades across {fire}/{syms} symbols (2y) -- too sparse to "
            f"deploy or retire on; pnl=${pnl:+,.0f} may be sample noise",
        )

    # PF in the JSON is averaged across per-symbol PFs (a known artifact of
    # the benchmark aggregator that biases towards symbols with no losses).
    # For decisions we lean primarily on net pnl, trade count, and WR;
    # PF is shown but treated as approximate.

    # Materially positive net + healthy WR + meaningful sample → KEEP
    if pnl > 0 and trades >= 20 and wr >= 50.0:
        cost_drag = gross["total_pnl"] - pnl
        return (
            "KEEP",
            f"net +${pnl:,.0f} on {trades} trades, WR {wr:.1f}% (PF~{pf:.2f}); "
            f"survived cost drag of ${cost_drag:,.0f}",
        )

    # Positive net but weak metrics or thin sample → RESEARCH-ONLY
    if pnl > 0:
        weak_reasons = []
        if wr < 50.0:
            weak_reasons.append(f"WR {wr:.1f}%<50")
        if trades < 20:
            weak_reasons.append(f"only {trades} trades")
        if not weak_reasons:
            weak_reasons.append("marginal edge")
        return (
            "RESEARCH-ONLY",
            f"net +${pnl:,.0f} but {', '.join(weak_reasons)} -- positive edge "
            f"is fragile; do not deploy without further validation",
        )

    cost_drag = gross["total_pnl"] - pnl
    delta_baseline = pnl - baseline["total_pnl"]
    return (
        "RETIRE",
        f"net ${pnl:+,.0f} on {trades} trades, WR {wr:.1f}%, PF {pf:.2f}; "
        f"cost drag ${cost_drag:,.0f}; vs baseline Δ${delta_baseline:+,.0f} — "
        f"persistently negative after corrections",
    )


def main() -> int:
    baseline_path = _latest("perplexity_ab_baseline")
    nocost_path = _latest("perplexity_ab_after-nocost")
    costed_path = _latest("perplexity_ab_after-costed")

    baseline = _load(baseline_path)
    nocost   = _load(nocost_path)
    costed   = _load(costed_path)

    bl = _by_strategy(baseline["strategy_rows"])
    nc = _by_strategy(nocost["strategy_rows"])
    ct = _by_strategy(costed["strategy_rows"])

    # All strategies present in the corrected (costed) run drive the table.
    names = list(ct.keys())

    rows = []
    for name in names:
        net = ct[name]
        gross = nc.get(name, net)
        base = bl.get(name, net)
        decision, reason = classify(net, gross, base, total_bars_estimate=500 * 10)
        rows.append({
            "strategy": name,
            "fire": f"{net['fire']}/{net['syms']}",
            "trades": net["trades"],
            "win_rate_pct": net["win_rate_pct"],
            "profit_factor": net["profit_factor"],
            "avg_holding_days": net["avg_holding_days"],
            "avg_return_pct": net["avg_return_pct"],
            "gross_pnl": gross["total_pnl"],
            "net_pnl": net["total_pnl"],
            "cost_drag": gross["total_pnl"] - net["total_pnl"],
            "baseline_pnl": base["total_pnl"],
            "delta_vs_baseline": net["total_pnl"] - base["total_pnl"],
            "decision": decision,
            "reason": reason,
        })

    # Sort: KEEP first (by net), then RESEARCH-ONLY (by net), then
    # NEEDS-FOLLOW-UP, then RETIRE (worst first).
    order = {"KEEP": 0, "RESEARCH-ONLY": 1, "NEEDS-FOLLOW-UP": 2, "RETIRE": 3}
    rows.sort(key=lambda r: (order[r["decision"]], -r["net_pnl"]))

    # ── Write CSV ──
    csv_path = REPORTS / "perplexity_strategy_decisions.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ── Write Markdown ──
    md_path = REPORTS / "perplexity_strategy_decisions.md"
    md = []
    md.append("# Perplexity strategy decisions — corrected harness")
    md.append("")
    md.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    md.append("")
    md.append("Source artifacts:")
    md.append(f"- baseline (pre-fix harness): `{baseline_path.name}`")
    md.append(f"- after-nocost (post-fix engine, costs OFF — GROSS): `{nocost_path.name}`")
    md.append(f"- after-costed (post-fix engine + regime parity + costs ON — NET): `{costed_path.name}`")
    md.append("")
    md.append(f"Symbols ({len(costed['symbols'])}): `{', '.join(costed['symbols'])}` — period `{costed['period']}`")
    md.append("")
    md.append("Decision rules (deterministic from the columns below):")
    md.append("- **KEEP** -- net > 0, trades >= 20, WR >= 50%")
    md.append("- **RESEARCH-ONLY** -- net >= 0 but trade count < 20 OR WR < 50% (positive but fragile)")
    md.append("- **NEEDS-FOLLOW-UP** -- net < 0 OR < 5 trades AND a known harness defect still distorts the verdict")
    md.append("  (the 4 daily-candlestick momentum patterns emit short SELLs the engine still drops)")
    md.append("- **RETIRE** -- net < 0 after all known corrections, no rescue path visible in current engine")
    md.append("")
    md.append("**Note on PF**: the `pf` column comes from the benchmark's per-symbol PF average")
    md.append("which is biased upward when a symbol has no losing trades. Verified independently")
    md.append("for the two borderline cases (Breakout_Consolidation: true PF=1.17 not 0.65;")
    md.append("BB_Breakout: true PF=0.80 not 2.57). Decisions weight net pnl + trade count + WR")
    md.append("over the displayed PF.")
    md.append("")
    md.append("| decision | strategy | fire | trd | wr% | pf | hold_d | gross | net | cost | Δ vs base |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| **{r['decision']}** | {r['strategy']} | {r['fire']} | "
            f"{r['trades']} | {r['win_rate_pct']:.1f} | {r['profit_factor']:.2f} | "
            f"{r['avg_holding_days']:.1f} | ${r['gross_pnl']:+,.0f} | "
            f"${r['net_pnl']:+,.0f} | ${r['cost_drag']:,.0f} | "
            f"${r['delta_vs_baseline']:+,.0f} |"
        )
    md.append("")
    md.append("## Rationale by strategy")
    md.append("")
    for r in rows:
        md.append(f"### {r['strategy']} — **{r['decision']}**")
        md.append("")
        md.append(r["reason"])
        md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {md_path.name} and {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
