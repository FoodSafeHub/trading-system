"""Math guards for the M1 target-allocation engine (advisory only).

Pin the cap-and-redistribute weighting and the gap-closing contribution split,
independent of live market data.
"""
from __future__ import annotations

import pandas as pd

from app.services.m1.targets import (
    _cap_and_redistribute, _volatility, _allocate_contribution,
    TargetPie, TargetSlice,
)


def test_cap_and_redistribute_respects_cap_and_sums_to_one():
    # One dominant score must be clamped to the cap; rest absorb the overflow.
    # 12 names @ 10% cap is feasible (12*0.10 = 1.2 >= 1).
    scores = {"A": 100.0}
    for i in range(11):
        scores[f"N{i}"] = 1.0
    w = _cap_and_redistribute(scores, cap=0.10)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v <= 0.10 + 1e-9 for v in w.values())
    assert abs(w["A"] - 0.10) < 1e-9          # the whale is capped at 10%


def test_cap_and_redistribute_uncapped_passthrough():
    # Already-feasible weights should normalize without clamping.
    w = _cap_and_redistribute({"A": 2, "B": 1, "C": 1}, cap=0.60)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert abs(w["A"] - 0.5) < 1e-6
    assert abs(w["B"] - 0.25) < 1e-6


def test_cap_too_tight_falls_back_to_equal():
    # cap*n < 1 is infeasible → equal split (best achievable under the cap).
    w = _cap_and_redistribute({"A": 5, "B": 1}, cap=0.10)
    assert abs(w["A"] - 0.5) < 1e-9 and abs(w["B"] - 0.5) < 1e-9


def test_zero_scores_yield_zero_weights():
    w = _cap_and_redistribute({"A": 0, "B": 0}, cap=0.10)
    assert w == {"A": 0.0, "B": 0.0}


def test_volatility_higher_for_choppier_series():
    steady = pd.Series([100 + i * 0.1 for i in range(120)], dtype=float)
    choppy = pd.Series([100 + (5 if i % 2 else -5) for i in range(120)], dtype=float)
    v_steady = _volatility(steady)
    v_choppy = _volatility(choppy)
    assert v_choppy > v_steady


def _slice(sym, cur_val, cur_w, tgt_w):
    return TargetSlice(
        symbol=sym, current_value=cur_val, current_weight=cur_w,
        target_weight=tgt_w, drift=tgt_w - cur_w, score=1.0,
        direction="HOLD", momentum_rs=None, volatility=None, laggard=False,
    )


def test_contribution_closes_underweight_gaps_and_never_sells():
    # Pie A is under target, Pie B is over target → all new money goes to A.
    pa = TargetPie(name="A", current_value=100, current_weight=0.50,
                   target_weight=0.80, drift=0.30, score=2.0,
                   slices=[_slice("X", 100, 1.0, 1.0)])
    pb = TargetPie(name="B", current_value=100, current_weight=0.50,
                   target_weight=0.20, drift=-0.30, score=0.5,
                   slices=[_slice("Y", 100, 1.0, 1.0)])
    _allocate_contribution([pa, pb], port_total=200.0, contribution=100.0,
                           max_stock_weight=0.10)
    # Over-weight pie gets nothing; under-weight pie gets it all (add-only).
    assert pb.suggested_dollars == 0.0
    assert abs(pa.suggested_dollars - 100.0) < 0.05
    # And it flows to the slice inside A.
    assert abs(pa.slices[0].suggested_dollars - 100.0) < 0.05


def test_no_contribution_is_a_noop():
    p = TargetPie(name="A", current_value=100, current_weight=1.0,
                  target_weight=1.0, drift=0.0, score=1.0,
                  slices=[_slice("X", 100, 1.0, 1.0)])
    _allocate_contribution([p], port_total=100.0, contribution=0.0,
                           max_stock_weight=0.10)
    assert p.suggested_dollars == 0.0
