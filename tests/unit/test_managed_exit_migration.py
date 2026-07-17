"""Migration endpoint behavior: cancel bot-placed resting stops safely.

Pins:
  - dry_run lists candidates without cancelling anything.
  - Execute seeds managed_exit_state BEFORE cancelling (protection level must
    survive the switch), then cancels and VERIFIES; only a verified cancel
    counts.
  - An unverified cancel is reported as failed and nothing is lost.
  - Manual (non-bot) stops are listed for review, never touched.
  - Flag off → RuntimeError (the route maps it to 409).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db import Base
from app.models.managed_exit_state import ManagedExitState
from app.models.orders import Order
from app.services.execution import managed_exit_engine as eng


class _FakeBroker:
    name = "schwab"
    supports_native_trailing_stop = True

    def __init__(self, resting, cancel_verifies=True):
        self._resting = list(resting)
        self._cancel_verifies = cancel_verifies
        self.cancel_calls = []
        self.events = []          # ordering probe: state-seed vs cancel

    async def authenticate(self):
        return True

    async def get_accounts(self):
        return [SimpleNamespace(account_id="ACC1")]

    async def list_orders(self, account_id, status=None):
        return list(self._resting) if status == "working" else []

    async def cancel_order(self, broker_order_id, account_id):
        self.cancel_calls.append(broker_order_id)
        self.events.append(("cancel", broker_order_id))
        return True

    async def get_order(self, broker_order_id, account_id):
        return SimpleNamespace(
            broker_order_id=broker_order_id,
            status="cancelled" if self._cancel_verifies else "working",
        )


def _resting(symbol="AAPL", boid="bo-1", order_type="TRAILING_STOP", stop=None):
    return SimpleNamespace(
        symbol=symbol, side="SELL", order_type=order_type,
        broker_order_id=boid, stop_price=stop, quantity=10.0,
    )


@pytest.fixture()
def db_env(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine, tables=[ManagedExitState.__table__, Order.__table__],
    )
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(get_settings(), "schwab_managed_exits_enabled", True)
    with patch.object(eng, "SessionLocal", TestSession), \
         patch("app.services.markets.is_india_symbol", return_value=False), \
         patch("app.services.notifications.bus.notify_suppression"), \
         patch("app.services.audit.service.AuditService.log", lambda *a, **k: None):
        yield TestSession


def _seed_bot_order(TestSession, boid="bo-1", key="trailstop-buy-1"):
    with TestSession() as db:
        db.add(Order(
            symbol="AAPL", side="SELL", order_type="TRAILING_STOP",
            quantity=10.0, status="submitted", broker_order_id=boid,
            idempotency_key=key, source="scheduler", broker="default",
        ))
        db.commit()


def _run(broker, dry_run):
    with patch("app.services.brokers.factory.get_broker", return_value=broker):
        return eng.migrate_from_resting_stops(dry_run=dry_run)


def test_flag_off_raises(monkeypatch):
    monkeypatch.setattr(get_settings(), "schwab_managed_exits_enabled", False)
    with pytest.raises(RuntimeError):
        eng.migrate_from_resting_stops(dry_run=True)


def test_dry_run_lists_without_cancelling(db_env):
    _seed_bot_order(db_env)
    broker = _FakeBroker([_resting()])
    out = _run(broker, dry_run=True)
    assert out["dry_run"] is True
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["bot_placed"] is True
    assert broker.cancel_calls == []


def test_execute_seeds_state_before_verified_cancel(db_env):
    _seed_bot_order(db_env)
    broker = _FakeBroker([_resting(stop=97.5, order_type="STOP")])

    real_upsert = eng.upsert_state

    def _tracking_upsert(db, symbol, **fields):
        broker.events.append(("seed", symbol))
        return real_upsert(db, symbol, **fields)

    with patch.object(eng, "upsert_state", _tracking_upsert):
        out = _run(broker, dry_run=False)

    assert out["cancelled"] == ["AAPL:bo-1"]
    assert out["failed"] == []
    # Protection level survives: state seeded BEFORE the cancel call.
    seed_i = broker.events.index(("seed", "AAPL"))
    cancel_i = broker.events.index(("cancel", "bo-1"))
    assert seed_i < cancel_i
    with db_env() as db:
        st = db.query(ManagedExitState).filter_by(symbol="AAPL").one()
        assert st.target_stop == 97.5


def test_unverified_cancel_counts_as_failed(db_env):
    _seed_bot_order(db_env)
    broker = _FakeBroker([_resting()], cancel_verifies=False)
    out = _run(broker, dry_run=False)
    assert out["cancelled"] == []
    assert len(out["failed"]) == 1
    assert out["failed"][0]["error"] == "cancel not verified"


def test_manual_stop_left_alone(db_env):
    # No matching Order row → not bot-placed → listed, never cancelled.
    broker = _FakeBroker([_resting(boid="manual-9")])
    out = _run(broker, dry_run=False)
    assert broker.cancel_calls == []
    assert out["candidates"][0]["action"] == "manual_review"
