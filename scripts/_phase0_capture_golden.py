#!/usr/bin/env python
"""ONE-TIME Phase 0 golden-baseline capture. Run against PRISTINE code BEFORE
any engine/rules/exits edits. Writes JSON fixtures the parity tests load.

  python scripts/_phase0_capture_golden.py

Outputs:
  tests/golden/engine_baseline.json      (run_backtest snapshots, cost-free)
  tests/golden/chandelier_baseline.json  (legacy _apply_chandelier_overlay outputs)

Safe to delete after Phase 0 lands; the fixtures it writes are what matter.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.golden.synth import (  # noqa: E402
    ENGINE_CASES, CHANDELIER_CASES, make_ohlcv, build_chandelier_inputs,
    result_to_snapshot, overlay_to_snapshot,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden")


def capture_engine() -> dict:
    from app.services.backtest.engine import run_backtest
    df = make_ohlcv()
    out = {}
    for case in ENGINE_CASES:
        r = run_backtest(
            strategy_name="golden:" + case["name"],
            symbol="TEST",
            strategy_type=case["strategy_type"],
            params=case["params"],
            period="golden",
            initial_capital=100_000.0,
            quantity=0,
            df=df.copy(),
        )
        out[case["name"]] = result_to_snapshot(r)
    return out


def capture_chandelier() -> dict:
    from app.services.strategy.models import StrategySignal
    from app.services.strategy.rules import _apply_chandelier_overlay, PositionState
    out = {}
    for case in CHANDELIER_CASES:
        prices, ohlcv, params, sig_kwargs, c = build_chandelier_inputs(case)
        sig = StrategySignal(**sig_kwargs)
        pos = (PositionState(entry_price=c["entry"], highest_close=c["highest_close"])
               if c["has_position"] else None)
        result = _apply_chandelier_overlay(sig, prices, ohlcv, params, pos)
        out[case["name"]] = overlay_to_snapshot(result)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    eng = capture_engine()
    with open(os.path.join(OUT_DIR, "engine_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(eng, f, indent=2)
    chan = capture_chandelier()
    with open(os.path.join(OUT_DIR, "chandelier_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(chan, f, indent=2)
    # Console summary (cp1252-safe: no unicode).
    for name, snap in eng.items():
        print(f"ENGINE {name}: trades={snap['total_trades']} "
              f"final={snap['final_capital']} ret%={snap['total_return_pct']}")
    for name, snap in chan.items():
        print(f"CHANDELIER {name}: {snap['direction']} "
              f"trail_keys={[k for k in snap['indicators'] if 'trail' in k or 'chand' in k]}")
    print("WROTE tests/golden/engine_baseline.json + chandelier_baseline.json")


if __name__ == "__main__":
    main()
