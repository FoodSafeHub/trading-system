#!/usr/bin/env python
"""GATE 1 — strictly offline unified-cutover validation (READ-ONLY).

Zero impact: no assignments, profiles, strategies.json, or flags are touched.
For each candidate symbol it runs, on IDENTICAL data:
  * head-to-head: unified type vs current-legacy type, GROSS (costs off) and NET
    (US/India cost preset),
  * walk-forward OOS (simple split) for both the unified and legacy type,
and prints a per-symbol verdict: ADVANCE or HOLD-ON-LEGACY.

The "legacy" config is exactly what trades live today: _make_generic_configs(sym)
(so NVDA's legacy config carries its Chandelier trail). The "unified" config is
_make_unified_configs(sym) (calibrated entry params + exit_policy).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.backtest.engine import run_backtest
from app.services.backtest.costs import US_DEFAULT, INDIA_DEFAULT
from app.services.backtest.walkforward_v2 import run_simple_split_v2
from app.services.market_data.provider import get_ohlcv
from app.services.scanner.scanner_service import _make_generic_configs, _make_unified_configs
from app.api.routes.backtest import _trade_metrics
from app.services.markets import is_india_symbol

PERIOD = "5y"
INIT = 10_000.0
MIN_SAMPLE = 10        # unified must have >= this many round-trips
WFE_FLOOR = 0.30       # OOS not a collapse: WFE >= floor OR OOS CAGR > 0

# (symbol, legacy_type, unified_type, highest_risk, paper_only_note)
PAIRS = [
    ("NVDA",       "rsi2_mean_reversion", "rsi2_reversion", True,  ""),
    ("AAPL",       "pullback_ema50",      "trend_pullback", False, ""),
    ("FANG",       "pullback_ema50",      "trend_pullback", False, ""),
    ("BHARTIARTL", "rsi2_mean_reversion", "rsi2_reversion", False,
     "PAPER-ONLY until India regime gate (^NSEI/^INDIAVIX) is validated"),
]


def _cfg(sym, gen_type, uni_type):
    leg = next((c for c in _make_generic_configs(sym) if c.type == gen_type), None)
    uni = next((c for c in _make_unified_configs(sym) if c.type == uni_type), None)
    return leg, uni


def _bt(sym, stype, params, df, cm):
    r = run_backtest(strategy_name=f"{stype}:{sym}", symbol=sym, strategy_type=stype,
                     params=params, period=PERIOD, initial_capital=INIT, quantity=0,
                     df=df.copy(), cost_model=cm)
    m = _trade_metrics(r)
    m["ret"] = r.total_return_pct
    m["sharpe"] = r.sharpe_ratio
    m["maxdd"] = r.max_drawdown_pct
    return m


def _wfe(sym, stype, params):
    try:
        res = run_simple_split_v2(stype, sym, params, period=PERIOD, train_pct=0.70)
        return res.wfe, res.oos_segment.cagr, res.oos_segment.total_return_pct
    except Exception as e:
        return None, None, f"err: {str(e)[:40]}"


def _wins(uni, leg):
    """Unified beats legacy on return AND per-trade expectancy."""
    return uni["ret"] > leg["ret"] and uni["expectancy_pct"] >= leg["expectancy_pct"]


def _fmt(m):
    return (f"ret%={m['ret']:8.1f}  exp%={m['expectancy_pct']:6.2f}  wr%={m['win_rate_pct']:5.1f}  "
            f"PF={str(m['profit_factor']):>5}  maxDD%={m['maxdd']:5.1f}  rt={m['round_trips']:>3}")


def main():
    print("=" * 96)
    print("GATE 1 — Offline unified-cutover validation (READ-ONLY; no files/assignments/flags changed)")
    print(f"Period {PERIOD} | head-to-head on identical data | gross + net | walk-forward OOS")
    print("=" * 96)

    summary = []
    for sym, gen_type, uni_type, high_risk, paper_note in PAIRS:
        tag = "  [HIGHEST-RISK SWITCH]" if high_risk else ""
        print(f"\n### {sym}: {uni_type} (unified) vs {gen_type} (legacy){tag}")
        leg_cfg, uni_cfg = _cfg(sym, gen_type, uni_type)
        if leg_cfg is None or uni_cfg is None:
            print(f"  CONFIG MISSING (legacy={leg_cfg is not None}, unified={uni_cfg is not None}) -> HOLD-ON-LEGACY")
            summary.append((sym, "HOLD-ON-LEGACY", "config missing")); continue
        try:
            df = get_ohlcv(sym, period=PERIOD)
        except Exception as e:
            print(f"  DATA FETCH FAILED: {e} -> HOLD-ON-LEGACY")
            summary.append((sym, "HOLD-ON-LEGACY", "no data")); continue
        if df is None or df.empty or len(df) < 120:
            print(f"  INSUFFICIENT DATA (len={0 if df is None else len(df)}) -> HOLD-ON-LEGACY")
            summary.append((sym, "HOLD-ON-LEGACY", "insufficient data")); continue

        net_cm = INDIA_DEFAULT if is_india_symbol(sym) else US_DEFAULT
        cm_label = "INDIA" if is_india_symbol(sym) else "US"

        leg_g = _bt(sym, gen_type, leg_cfg.params, df, None)
        uni_g = _bt(sym, uni_type, uni_cfg.params, df, None)
        leg_n = _bt(sym, gen_type, leg_cfg.params, df, net_cm)
        uni_n = _bt(sym, uni_type, uni_cfg.params, df, net_cm)

        print(f"  GROSS  legacy   {_fmt(leg_g)}")
        print(f"  GROSS  unified  {_fmt(uni_g)}")
        print(f"  NET[{cm_label}] legacy   {_fmt(leg_n)}")
        print(f"  NET[{cm_label}] unified  {_fmt(uni_n)}")

        uw, uo_cagr, uo_ret = _wfe(sym, uni_type, uni_cfg.params)
        lw, lo_cagr, lo_ret = _wfe(sym, gen_type, leg_cfg.params)
        print(f"  OOS    unified  WFE={uw}  oos_cagr={uo_cagr}  oos_ret%={uo_ret}")
        print(f"  OOS    legacy   WFE={lw}  oos_cagr={lo_cagr}  oos_ret%={lo_ret}")

        gross_win = _wins(uni_g, leg_g)
        net_win = _wins(uni_n, leg_n)
        sample_ok = uni_g["round_trips"] >= MIN_SAMPLE
        oos_ok = (isinstance(uo_cagr, (int, float)) and uo_cagr > 0) or \
                 (isinstance(uw, (int, float)) and uw >= WFE_FLOOR)

        advance = gross_win and net_win and sample_ok and oos_ok
        verdict = "ADVANCE" if advance else "HOLD-ON-LEGACY"
        reasons = []
        reasons.append(("gross_win" if gross_win else "gross_FAIL"))
        reasons.append(("net_win" if net_win else "net_FAIL"))
        reasons.append((f"sample_ok({uni_g['round_trips']})" if sample_ok else f"sample_LOW({uni_g['round_trips']})"))
        reasons.append(("oos_ok" if oos_ok else "oos_WEAK"))
        flag = ""
        if advance and paper_note:
            flag = f"  | NOTE: {paper_note}"
        if high_risk:
            flag += "  | HIGHEST-RISK: needs strictest review even if ADVANCE"
        print(f"  -> VERDICT: {verdict}  [{', '.join(reasons)}]{flag}")
        summary.append((sym, verdict, ", ".join(reasons) + (" | " + paper_note if (advance and paper_note) else "")))

    print("\n" + "=" * 96)
    print("PHASE-A VERDICT SUMMARY")
    print("=" * 96)
    for sym, verdict, why in summary:
        print(f"  {sym:<11} {verdict:<15} {why}")
    print("\nSTOP — Phase A complete. Do NOT proceed to GATE 2 without explicit approval.")


if __name__ == "__main__":
    main()
