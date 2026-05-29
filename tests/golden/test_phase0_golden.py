"""Phase 0 golden parity gate.

These tests prove the Phase 0 scaffolding introduced ZERO behaviour change with
default settings. They compare against fixtures captured from PRISTINE code by
scripts/_phase0_capture_golden.py. If any of these fail, Phase 1 must not begin.
"""
import json
import os

import pytest

from tests.golden.synth import (
    ENGINE_CASES, CHANDELIER_CASES, make_ohlcv, build_chandelier_inputs,
    result_to_snapshot, overlay_to_snapshot,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> dict:
    path = os.path.join(_HERE, name)
    if not os.path.exists(path):
        pytest.skip(f"baseline fixture missing: {name} (run scripts/_phase0_capture_golden.py)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda c: c["name"])
def test_engine_default_matches_baseline(case):
    from app.services.backtest.engine import run_backtest
    base = _load("engine_baseline.json")[case["name"]]
    r = run_backtest(
        strategy_name="golden:" + case["name"], symbol="TEST",
        strategy_type=case["strategy_type"], params=case["params"],
        period="golden", initial_capital=100_000.0, quantity=0, df=make_ohlcv().copy(),
    )
    assert result_to_snapshot(r) == base


@pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda c: c["name"])
def test_engine_explicit_zero_matches_baseline(case):
    from app.services.backtest.engine import run_backtest
    from app.services.backtest.costs import ZERO
    base = _load("engine_baseline.json")[case["name"]]
    r = run_backtest(
        strategy_name="golden:" + case["name"], symbol="TEST",
        strategy_type=case["strategy_type"], params=case["params"],
        period="golden", initial_capital=100_000.0, quantity=0,
        df=make_ohlcv().copy(), cost_model=ZERO,
    )
    assert result_to_snapshot(r) == base


@pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda c: c["name"])
def test_us_cost_is_a_drag(case):
    from app.services.backtest.engine import run_backtest
    from app.services.backtest.costs import US_DEFAULT
    df = make_ohlcv()
    r0 = run_backtest(strategy_name="g", symbol="TEST", strategy_type=case["strategy_type"],
                      params=case["params"], period="golden", initial_capital=100_000.0,
                      quantity=0, df=df.copy())
    ru = run_backtest(strategy_name="g", symbol="TEST", strategy_type=case["strategy_type"],
                      params=case["params"], period="golden", initial_capital=100_000.0,
                      quantity=0, df=df.copy(), cost_model=US_DEFAULT)
    assert ru.final_capital <= r0.final_capital + 1e-9


@pytest.mark.parametrize("case", CHANDELIER_CASES, ids=lambda c: c["name"])
def test_chandelier_overlay_parity(case):
    """The new apply_exit_overlay legacy branch must reproduce the pristine
    _apply_chandelier_overlay output exactly."""
    from app.services.strategy.exits import apply_exit_overlay
    from app.services.strategy.models import StrategySignal
    from app.services.strategy.rules import PositionState
    base = _load("chandelier_baseline.json")[case["name"]]
    prices, ohlcv, params, sig_kwargs, cc = build_chandelier_inputs(case)
    sig = StrategySignal(**sig_kwargs)
    pos = (PositionState(entry_price=cc["entry"], highest_close=cc["highest_close"])
           if cc["has_position"] else None)
    out = apply_exit_overlay(sig, prices, ohlcv, params, pos)
    assert overlay_to_snapshot(out) == base


def test_walkforward_zero_cost_identity():
    import dataclasses
    from app.services.backtest.walkforward_v2 import _run_slice
    from app.services.backtest.costs import ZERO
    df = make_ohlcv(n=400)
    warm, test = df.iloc[:250], df.iloc[250:]
    params = {"rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 70,
              "sma_trend": 200, "exit_sma": 5, "atr_skip_threshold": 5.0}
    a = _run_slice("rsi2_mean_reversion", "TEST", params, test, warm, 100_000.0)
    b = _run_slice("rsi2_mean_reversion", "TEST", params, test, warm, 100_000.0, ZERO)
    assert dataclasses.asdict(a) == dataclasses.asdict(b)
