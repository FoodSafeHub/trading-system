"""
Tests for the trade-count cap behavior in RiskGovernor.

Background: max_trades_per_day used to be a hard kill-switch defaulting to 6,
which silently halted the autotrader mid-session. There is no regulatory cap on
the *number* of intraday round-trips, so the default is now 0 = unlimited, and a
positive value is an opt-in throttle. The loss-limit and consecutive-loss kill
switches are real risk controls and must keep firing.
"""
from __future__ import annotations

from app.services.strategy.daytrading.brain.risk_governor import RiskGovernor


def _wins(n: int) -> list[dict]:
    return [{"pnl": 5.0} for _ in range(n)]


def _losses(n: int) -> list[dict]:
    return [{"pnl": -5.0} for _ in range(n)]


def test_default_is_unlimited_trades():
    g = RiskGovernor()
    assert g.config["max_trades_per_day"] == 0
    state = g.build_risk_state(_wins(100), initial_capital=10_000.0)
    assert not state.kill_switch_triggered
    assert g.check_can_trade(state).allowed


def test_zero_cap_never_blocks_on_count():
    g = RiskGovernor(config={"max_trades_per_day": 0})
    state = g.build_risk_state(_wins(500), initial_capital=10_000.0)
    assert not state.kill_switch_triggered


def test_positive_cap_is_opt_in_and_fires():
    g = RiskGovernor(config={"max_trades_per_day": 3})
    state = g.build_risk_state(_wins(3), initial_capital=10_000.0)
    assert state.kill_switch_triggered
    assert "opt-in limit 3" in state.kill_switch_reason
    assert not g.check_can_trade(state).allowed


def test_positive_cap_allows_below_limit():
    g = RiskGovernor(config={"max_trades_per_day": 5})
    state = g.build_risk_state(_wins(4), initial_capital=10_000.0)
    assert not state.kill_switch_triggered


def test_consecutive_loss_kill_switch_still_fires_under_unlimited():
    # Even with unlimited trade count, 3 losses in a row must stop trading.
    g = RiskGovernor(config={"max_trades_per_day": 0, "max_consecutive_losses": 3})
    state = g.build_risk_state(_losses(3), initial_capital=10_000.0)
    assert state.kill_switch_triggered
    assert "consecutive losses" in state.kill_switch_reason


def test_daily_loss_kill_switch_still_fires_under_unlimited():
    g = RiskGovernor(config={"max_trades_per_day": 0, "max_daily_loss_pct": 2.0})
    # One big loss = -3% of 10k -> below the -2% floor.
    state = g.build_risk_state([{"pnl": -300.0}], initial_capital=10_000.0)
    assert state.kill_switch_triggered
    assert "Daily loss limit" in state.kill_switch_reason


def test_max_open_positions_is_configurable():
    g = RiskGovernor(config={"max_open_positions": 3})
    state = g.build_risk_state(_wins(1), initial_capital=10_000.0, open_positions=2)
    # 2 open < 3 max -> still allowed
    assert g.check_can_trade(state).allowed
    state2 = g.build_risk_state(_wins(1), initial_capital=10_000.0, open_positions=3)
    assert not g.check_can_trade(state2).allowed
