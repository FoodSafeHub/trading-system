"""Unit tests for the managed-exit engine (bot-managed exits, no resting stops).

Pins:
  - parse_alert_levels: parsing, sign-fixing, dedup, shallow→deep order.
  - drawdown_alert_pass: fires the deepest crossed level once, dedups while
    below it, re-arms only after the hysteresis recovery, escalates severity.
  - heartbeat write/read used by the invariant watchdog.
  - red_hold_check: holds a red SELL once per episode, fail-open on unknown
    cost / disabled flag, passes green positions through.
  - run_once: clears a red-hold on the green flip and marks gone positions
    exited.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db import Base
from app.models.managed_exit_state import (
    MODE_EXITED,
    MODE_MONITORING,
    MODE_RED_HOLD,
    ManagedExitState,
)
from app.models.settings import AppSetting
from app.services.execution import managed_exit_engine as eng


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[ManagedExitState.__table__, AppSetting.__table__],
    )
    TestSession = sessionmaker(bind=engine)
    with patch.object(eng, "SessionLocal", TestSession):
        with TestSession() as db:
            yield db


@pytest.fixture()
def managed_on(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "schwab_managed_exits_enabled", True)
    monkeypatch.setattr(s, "red_hold_enabled", True)
    monkeypatch.setattr(s, "managed_exit_drawdown_alert_levels", "-8,-12,-20")
    monkeypatch.setattr(s, "managed_exit_alert_rearm_pct", 2.0)
    with patch("app.services.markets.is_india_symbol", return_value=False):
        yield s


# ── parse_alert_levels ──────────────────────────────────────────────────────

def test_parse_alert_levels_orders_shallow_to_deep():
    assert eng.parse_alert_levels("-8,-12,-20") == [-8.0, -12.0, -20.0]


def test_parse_alert_levels_fixes_signs_and_dedups():
    # A "8" typo means -8; junk entries are dropped; duplicates collapse.
    assert eng.parse_alert_levels("8, -8, x, , -20") == [-8.0, -20.0]


def test_parse_alert_levels_empty():
    assert eng.parse_alert_levels("") == []


# ── drawdown_alert_pass ─────────────────────────────────────────────────────

def _state(db, symbol="AAPL", avg_cost=100.0, **kw):
    st = eng.upsert_state(db, symbol, avg_cost=avg_cost, **kw)
    db.commit()
    return st


def test_drawdown_fires_deepest_crossed_level_once(db_session, managed_on):
    st = _state(db_session)
    with patch("app.services.notifications.bus.notify_suppression") as mock_n:
        # -13% crosses -8 and -12 → one alert at -12.
        fired = eng.drawdown_alert_pass(db_session, st, 87.0)
        assert fired == -12.0
        assert mock_n.call_count == 1
        # Still below -12 → dedup, nothing fires.
        assert eng.drawdown_alert_pass(db_session, st, 87.5) is None
        assert mock_n.call_count == 1


def test_drawdown_escalates_deeper_level(db_session, managed_on):
    st = _state(db_session)
    with patch("app.services.notifications.bus.notify_suppression") as mock_n:
        assert eng.drawdown_alert_pass(db_session, st, 91.0) == -8.0
        # Plunge through -20 → escalates (SEVERE) even though -8 already fired.
        assert eng.drawdown_alert_pass(db_session, st, 79.0) == -20.0
        assert mock_n.call_count == 2
        assert "SEVERE" in mock_n.call_args.kwargs["detail"]


def test_drawdown_rearm_hysteresis(db_session, managed_on):
    st = _state(db_session)
    with patch("app.services.notifications.bus.notify_suppression") as mock_n:
        assert eng.drawdown_alert_pass(db_session, st, 91.0) == -8.0
        # Oscillation just above the level: -7.9% is NOT ≥ -8 + 2 → no re-arm.
        assert eng.drawdown_alert_pass(db_session, st, 92.1) is None
        assert eng.drawdown_alert_pass(db_session, st, 91.9) is None
        assert mock_n.call_count == 1
        # Real recovery to -5% (≥ -6) re-arms; a re-plunge alerts again.
        assert eng.drawdown_alert_pass(db_session, st, 95.0) is None
        assert eng.drawdown_alert_pass(db_session, st, 91.0) == -8.0
        assert mock_n.call_count == 2


def test_drawdown_never_places_orders(db_session, managed_on):
    # No broker surface at all — pass only touches DB + notifications.
    st = _state(db_session)
    with patch("app.services.notifications.bus.notify_suppression"):
        eng.drawdown_alert_pass(db_session, st, 79.0)
    assert st.last_alert_level == -20.0


# ── heartbeat ───────────────────────────────────────────────────────────────

def test_heartbeat_write_and_age(db_session):
    assert eng.heartbeat_age_seconds(db_session) is None
    eng._write_heartbeat(db_session)
    age = eng.heartbeat_age_seconds(db_session)
    assert age is not None and age < 5.0


# ── red_hold_check ──────────────────────────────────────────────────────────

def test_red_hold_blocks_and_notifies_once(db_session, managed_on):
    with patch("app.services.notifications.bus.notify_suppression") as mock_n:
        held = eng.red_hold_check(
            db_session, "AAPL", 95.0, signal_price=96.0, avg_cost=100.0,
        )
        assert held is True
        st = eng.get_state(db_session, "AAPL")
        assert st.mode == MODE_RED_HOLD
        assert mock_n.call_count == 1
        # Same episode, next cycle: still held, NO second notification.
        assert eng.red_hold_check(
            db_session, "AAPL", 94.0, signal_price=96.0, avg_cost=100.0,
        ) is True
        assert mock_n.call_count == 1


def test_red_hold_passes_green_position(db_session, managed_on):
    assert eng.red_hold_check(
        db_session, "AAPL", 105.0, avg_cost=100.0,
    ) is False


def test_red_hold_fails_open_on_unknown_cost(db_session, managed_on):
    with patch.object(eng, "fifo_avg_costs", return_value={}):
        assert eng.red_hold_check(db_session, "AAPL", 95.0) is False


def test_red_hold_disabled_flag_passes_through(db_session, managed_on, monkeypatch):
    monkeypatch.setattr(get_settings(), "red_hold_enabled", False)
    assert eng.red_hold_check(
        db_session, "AAPL", 95.0, avg_cost=100.0,
    ) is False


def test_red_hold_off_when_master_switch_off(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "schwab_managed_exits_enabled", False)
    assert eng.red_hold_check(
        db_session, "AAPL", 95.0, avg_cost=100.0,
    ) is False


# ── run_once ────────────────────────────────────────────────────────────────

def test_run_once_clears_red_hold_on_green_flip(db_session, managed_on):
    now = datetime.now(timezone.utc)
    _state(
        db_session, "AAPL", avg_cost=100.0, mode=MODE_RED_HOLD,
        red_hold_since=now - timedelta(days=2),
        red_hold_notified_at=now - timedelta(days=2),
    )
    with patch("app.services.notifications.bus.notify_suppression") as mock_n, \
         patch("app.services.strategy.scheduler._symbol_market_open", return_value=True), \
         patch.object(eng, "fifo_avg_costs", return_value={"AAPL": 100.0}):
        out = eng.run_once({"AAPL": 10.0}, {"AAPL": 101.0})
    assert out["status"] == "ok"
    assert out["red_holds_cleared"] == ["AAPL"]
    st = eng.get_state(db_session, "AAPL")
    assert st.mode == MODE_MONITORING
    assert st.red_hold_since is None and st.red_hold_notified_at is None
    reasons = [c.kwargs.get("reason") for c in mock_n.call_args_list]
    assert "red_hold_cleared" in reasons


def test_run_once_marks_gone_positions_exited(db_session, managed_on):
    _state(db_session, "ZM", avg_cost=70.0, mode="trail_armed")
    with patch("app.services.notifications.bus.notify_suppression"), \
         patch("app.services.strategy.scheduler._symbol_market_open", return_value=True), \
         patch.object(eng, "fifo_avg_costs", return_value={}):
        eng.run_once({}, {})
    assert eng.get_state(db_session, "ZM").mode == MODE_EXITED


def test_run_once_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(get_settings(), "schwab_managed_exits_enabled", False)
    assert eng.run_once({}, {}) == {"status": "disabled"}
