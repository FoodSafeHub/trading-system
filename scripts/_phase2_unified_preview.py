#!/usr/bin/env python
"""Read-only Phase 2 preview: unified compare-all + calibrated-vs-inherited A/B.

No writes. Compares each calibrated unified profile (current saved params) against
its inherited baseline (the pre-calibration migrated params, from the precalib
backup) on the SAME data, so we can see which calibrations actually justify keeping.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.market_data.provider import get_ohlcv
from app.services.backtest.engine import run_backtest
from app.services.scanner.scanner_service import _make_unified_configs
from app.services.backtest.scanner_profiles import get_param_overrides
from app.api.routes.backtest import _trade_metrics

PRECALIB = "app/services/backtest/scanner_profiles.backup.precalib_20260529_113341.json"
SYMS = ["NVDA", "HERITGFOOD", "AAPL", "BHARTIARTL", "FANG", "RELIANCE"]
PAIRS = [("rsi2_reversion", "NVDA"), ("rsi2_reversion", "HERITGFOOD"),
         ("trend_pullback", "AAPL"), ("trend_pullback", "BHARTIARTL"),
         ("trend_pullback", "FANG"), ("trend_pullback", "RELIANCE")]

# Pure factory defaults (a symbol with no profile / no alias predecessor).
DEFAULTS = {c.type: dict(c.params) for c in _make_unified_configs("ZQXWNOPROF")}
precalib = json.load(open(PRECALIB, encoding="utf-8"))

dfs = {}
for s in SYMS:
    try:
        dfs[s] = get_ohlcv(s, period="5y")
    except Exception as e:
        dfs[s] = None
        print(f"FETCH FAIL {s}: {e}")


def _m(stype, sym, params):
    r = run_backtest(strategy_name=f"{stype}:{sym}", symbol=sym, strategy_type=stype,
                     params=params, period="5y", initial_capital=10_000.0, quantity=0,
                     df=dfs[sym].copy())
    return _trade_metrics(r), r.total_return_pct


print("================ A/B: inherited baseline vs calibrated (current saved) ================")
for stype, sym in PAIRS:
    if dfs.get(sym) is None:
        print(f"{stype}:{sym}  NO DATA"); continue
    inh_ov = (precalib.get(f"{stype}:{sym}") or {}).get("param_overrides", {})
    cur_ov = get_param_overrides(stype, sym)
    inh = {**DEFAULTS[stype], **inh_ov}
    cur = {**DEFAULTS[stype], **cur_ov}
    im, ir = _m(stype, sym, inh)
    cm, cr = _m(stype, sym, cur)
    same = inh_ov == cur_ov
    d_ret = cr - ir
    d_exp = cm["expectancy_pct"] - im["expectancy_pct"]
    if same:
        verdict = "UNCHANGED (kept inherited; calibration found no gain)"
    elif d_ret > 0.01 or d_exp > 0.01:
        verdict = "KEEP (calibrated beats inherited)"
    else:
        verdict = "REVERT? (calibrated does NOT beat inherited on replay)"
    print(f"{stype}:{sym}")
    print(f"  inherited   ret%={ir:8.1f}  wr%={im['win_rate_pct']:5.1f}  exp%={im['expectancy_pct']:6.2f}  rt={im['round_trips']:>3}  ov={inh_ov}")
    print(f"  calibrated  ret%={cr:8.1f}  wr%={cm['win_rate_pct']:5.1f}  exp%={cm['expectancy_pct']:6.2f}  rt={cm['round_trips']:>3}  ov={cur_ov}")
    print(f"  delta ret={d_ret:+.1f}pp  exp={d_exp:+.2f}  -> {verdict}")
    print()

print("================ Unified compare-all (ranked by return, 5y) ================")
for sym in SYMS:
    if dfs.get(sym) is None:
        print(f"{sym}: NO DATA"); continue
    rows = []
    for c in _make_unified_configs(sym):
        m, ret = _m(c.type, sym, c.params)
        rows.append((c.type, m, ret))
    rows.sort(key=lambda x: x[2], reverse=True)
    print(f"=== {sym} ===")
    for t, m, ret in rows:
        print(f"  {t:<18} rt={m['round_trips']:>3}  ret%={ret:8.1f}  wr%={m['win_rate_pct']:5.1f}  "
              f"exp%={m['expectancy_pct']:6.2f}  PF={str(m['profit_factor']):>5}  maxDD%={m['max_drawdown_pct']:5.1f}")
    print()
print("DONE (read-only; no profiles written)")
