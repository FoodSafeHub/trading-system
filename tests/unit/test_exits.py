"""Unit tests for the exit-policy overlay layer (Phase 0).

Legacy-parity is covered by tests/golden/test_phase0_golden.py. Here we test the
NEW policy path branches and the precedence contract (exit_policy overrides
trail_enabled; entries and position-less calls are never altered).
"""
import numpy as np
import pandas as pd

from app.services.strategy.exits import apply_exit_overlay, ExitPolicy
from app.services.strategy.models import StrategySignal
from app.services.strategy.rules import PositionState
from tests.golden.synth import build_chandelier_inputs


def _sig(direction="HOLD"):
    return StrategySignal(symbol="T", direction=direction, strength=0.5,
                          indicators={"src": "test"}, strategy_name="t")


def _frame(closes):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = pd.Series([float(x) for x in closes], index=idx)
    df = pd.DataFrame({"Open": c.values, "High": c.values + 1.0,
                       "Low": c.values - 1.0, "Close": c.values,
                       "Volume": np.full(len(closes), 1e6)}, index=idx)
    return c, df


# ── Passthrough / no-op contract ───────────────────────────────────────────────

def test_position_none_is_noop():
    c, df = _frame(np.linspace(100, 110, 30))
    out = apply_exit_overlay(_sig("SELL"), c, df, {"exit_policy": {"trail": "chandelier"}}, None)
    assert out.direction == "SELL" and "trail_exit" not in out.indicators


def test_no_policy_no_trail_is_noop():
    c, df = _frame(np.linspace(100, 110, 30))
    pos = PositionState(entry_price=100.0, highest_close=110.0)
    out = apply_exit_overlay(_sig("SELL"), c, df, {}, pos)
    assert out.direction == "SELL" and out.indicators == {"src": "test"}


# ── Precedence: exit_policy overrides trail_enabled ────────────────────────────

def test_exit_policy_overrides_trail_enabled():
    # Inputs where the LEGACY path would convert SELL->HOLD (in profit, above
    # chandelier). With an (inert) exit_policy present, the legacy path must be
    # skipped, so the SELL passes straight through.
    _, df, _, sig_kwargs, cc = build_chandelier_inputs(
        {"name": "x", "n": 30, "c_now": 109.0, "entry": 100.0, "highest_close": 110.0,
         "direction": "SELL", "has_position": True, "trail_enabled": True})
    c, _ = _frame(list(np.linspace(104, 109, 30)))
    pos = PositionState(entry_price=100.0, highest_close=110.0)
    params = {"exit_policy": {"trail": "none"}, "trail_enabled": True,
              "trail_trigger_pct": 3.0, "atr_trail_mult": 3.0, "atr_trail_period": 22}
    out = apply_exit_overlay(StrategySignal(**sig_kwargs), c, df, params, pos)
    assert out.direction == "SELL"            # not held -> legacy path was bypassed
    assert "trail_hold" not in out.indicators


# ── Policy branch: time stop ───────────────────────────────────────────────────

def test_time_stop_forces_exit():
    c, df = _frame(np.linspace(100, 101, 30))
    pos = PositionState(entry_price=100.0, highest_close=101.0, bars_held=12)
    out = apply_exit_overlay(_sig("HOLD"), c, df, {"exit_policy": {"time_stop_bars": 5}}, pos)
    assert out.direction == "SELL" and out.indicators.get("time_stop") == 12


def test_time_stop_not_yet_reached():
    c, df = _frame(np.linspace(100, 101, 30))
    pos = PositionState(entry_price=100.0, highest_close=101.0, bars_held=3)
    out = apply_exit_overlay(_sig("HOLD"), c, df, {"exit_policy": {"time_stop_bars": 5}}, pos)
    assert out.direction == "HOLD" and "time_stop" not in out.indicators


# ── Policy branch: trend failure ───────────────────────────────────────────────

def test_trend_fail_ema_forces_exit():
    # Falling prices: close sits below EMA(50) -> trend failed.
    c, df = _frame(np.linspace(200, 100, 80))
    pos = PositionState(entry_price=200.0, highest_close=200.0, bars_held=10)
    out = apply_exit_overlay(_sig("HOLD"), c, df, {"exit_policy": {"trend_fail": "ema:50"}}, pos)
    assert out.direction == "SELL" and out.indicators.get("trend_fail") == "ema:50"


def test_trend_ok_no_exit():
    c, df = _frame(np.linspace(100, 200, 80))  # rising: close above EMA50
    pos = PositionState(entry_price=100.0, highest_close=200.0, bars_held=10)
    out = apply_exit_overlay(_sig("HOLD"), c, df, {"exit_policy": {"trend_fail": "ema:50"}}, pos)
    assert out.direction == "HOLD"


# ── Policy branch: chandelier trail ────────────────────────────────────────────

def test_policy_chandelier_forced_sell():
    _, df, _, sig_kwargs, cc = build_chandelier_inputs(
        {"name": "x", "n": 30, "c_now": 103.0, "entry": 99.0, "highest_close": 112.0,
         "direction": "HOLD", "has_position": True, "trail_enabled": True})
    c, _ = _frame(list(np.linspace(98, 103, 30)))
    pos = PositionState(entry_price=99.0, highest_close=112.0, bars_held=10)
    policy = {"trail": "chandelier", "atr_mult": 3.0, "atr_period": 22, "trigger_pct": 3.0}
    out = apply_exit_overlay(StrategySignal(**sig_kwargs), c, df, {"exit_policy": policy}, pos)
    assert out.direction == "SELL" and out.indicators.get("trail_exit") is True


def test_policy_chandelier_rides_winner():
    _, df, _, sig_kwargs, cc = build_chandelier_inputs(
        {"name": "x", "n": 30, "c_now": 109.0, "entry": 100.0, "highest_close": 110.0,
         "direction": "SELL", "has_position": True, "trail_enabled": True})
    c, _ = _frame(list(np.linspace(104, 109, 30)))
    pos = PositionState(entry_price=100.0, highest_close=110.0, bars_held=10)
    policy = {"trail": "chandelier", "atr_mult": 3.0, "atr_period": 22, "trigger_pct": 3.0}
    out = apply_exit_overlay(StrategySignal(**sig_kwargs), c, df, {"exit_policy": policy}, pos)
    assert out.direction == "HOLD" and out.indicators.get("trail_hold") is True


def test_partial_is_parsed_not_honoured():
    # `partial` present must not crash and must not change a non-triggering exit.
    c, df = _frame(np.linspace(100, 101, 30))
    pos = PositionState(entry_price=100.0, highest_close=101.0, bars_held=1)
    policy = {"trail": "none", "partial": [{"r": 1.5, "pct": 0.5}]}
    out = apply_exit_overlay(_sig("HOLD"), c, df, {"exit_policy": policy}, pos)
    assert out.direction == "HOLD"


def test_exit_policy_from_params_ignores_unknown_keys():
    p = ExitPolicy.from_params({"exit_policy": {"trail": "atr", "bogus": 1}})
    assert p is not None and p.trail == "atr"
    assert ExitPolicy.from_params({}) is None
