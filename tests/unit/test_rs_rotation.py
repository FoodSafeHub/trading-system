"""Phase 1: relative-strength rotation service + harness."""
import numpy as np
import pandas as pd
import pytest

from app.services.strategy.rs_rotation import (
    rank_relative_strength, backtest_rs_rotation,
)
from app.services.backtest.costs import US_DEFAULT


def _series(drift, n=400, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.015, n)
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Close": close}, index=idx)


def _basket():
    return {
        "STRONG": _series(0.0015, seed=1),
        "MILD": _series(0.0006, seed=2),
        "FLAT": _series(0.0, seed=3),
        "WEAK": _series(-0.0008, seed=4),
        "SPY": _series(0.0004, seed=9),
    }


def test_rank_orders_by_relative_strength():
    data = _basket()
    ranked = rank_relative_strength(["STRONG", "MILD", "FLAT", "WEAK"], "SPY", data=data)
    syms = [r.symbol for r in ranked]
    assert syms[0] == "STRONG"
    assert syms.index("STRONG") < syms.index("WEAK")


def test_weak_name_flagged_below_sma200():
    data = _basket()
    ranked = {r.symbol: r for r in rank_relative_strength(["STRONG", "WEAK"], "SPY", data=data)}
    assert ranked["STRONG"].above_sma200 is True
    assert ranked["WEAK"].above_sma200 is False


def test_backtest_runs_and_returns_curve():
    data = _basket()
    res = backtest_rs_rotation(["STRONG", "MILD", "FLAT", "WEAK"], "SPY",
                               top_n=2, rebalance_days=21, data=data)
    assert res.n_rebalances > 0
    assert len(res.equity_curve) > 0
    assert res.equity_curve[0]["equity"] > 0


def test_cost_model_is_a_drag():
    data = _basket()
    base = backtest_rs_rotation(["STRONG", "MILD", "FLAT", "WEAK"], "SPY",
                                top_n=2, rebalance_days=21, data=data)
    costed = backtest_rs_rotation(["STRONG", "MILD", "FLAT", "WEAK"], "SPY",
                                  top_n=2, rebalance_days=21, data=data, cost_model=US_DEFAULT)
    assert costed.final_capital <= base.final_capital + 1e-6


def test_insufficient_data_raises():
    short = {"A": _series(0.001, n=50, seed=1), "SPY": _series(0.0004, n=50, seed=9)}
    with pytest.raises(ValueError):
        backtest_rs_rotation(["A"], "SPY", data=short)
