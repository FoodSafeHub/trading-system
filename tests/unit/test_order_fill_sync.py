"""Broker → DB order reconciliation (closes show up automatically).

The PnL page derives from filled Order rows. A position that closed at the
broker — e.g. a broker-native TRAILING_STOP that filled, or a manual close —
left no filled SELL Order, so FIFO never paired the close and the position
looked "open" forever. order_sync ingests these orphan fills.

These tests pin:
  - An orphan broker fill (no matching DB row) is INGESTED as a filled Order.
  - It then pairs with an existing filled BUY to produce a realized round-trip.
  - An existing submitted row matching a broker fill is flipped to filled.
  - Non-fill statuses don't create phantom rows.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orders import Order
from app.schemas.orders import OrderStatusResponse


def _bo(boid, symbol, side, qty, status, fill_price=None, close_time=None):
    return OrderStatusResponse(
        broker_order_id=boid, symbol=symbol, side=side, order_type="TRAILING_STOP",
        quantity=qty, status=status, fill_price=fill_price,
        raw={"closeTime": close_time} if close_time else {},
    )


class _FakeBroker:
    name = "schwab"

    def __init__(self, orders):
        self._orders = orders

    async def authenticate(self):
        return True

    async def get_accounts(self):
        return [SimpleNamespace(account_id="ACCT1")]

    async def list_orders(self, account_id, status=None):
        return list(self._orders)


@pytest.fixture
def db_session():
    # Import the models we touch so their tables register on Base.metadata,
    # then create the full schema (orders + realized_trades + signals, etc.).
    from app.models import orders, realized_trades, signals  # noqa: F401
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session


def _patch_env(db_session, broker):
    import app.services.reconciliation.order_sync as mod
    return (
        patch.object(mod, "SessionLocal", db_session),
        patch.object(mod, "get_broker", lambda: broker),
        # No India assignments → only the global broker runs.
        patch("app.services.strategy.scheduler._has_india_assignments", lambda: False),
        # Avoid the realized-trade sync touching a different SessionLocal.
        patch("app.services.pnl.store.sync_realized_trades", lambda db: (0, None)),
    )


def test_orphan_native_trail_fill_is_ingested(db_session):
    from app.services.reconciliation.order_sync import sync_broker_orders_once

    broker = _FakeBroker([
        _bo("T-1", "AAPL", "SELL", 10, "filled", fill_price=190.0),
    ])
    patches = _patch_env(db_session, broker)
    with patches[0], patches[1], patches[2], patches[3]:
        result = sync_broker_orders_once()

    assert result["created"] == 1
    with db_session() as db:
        rows = db.query(Order).filter_by(broker_order_id="T-1").all()
        assert len(rows) == 1
        o = rows[0]
        assert o.status == "filled"
        assert o.side == "SELL"
        assert o.fill_price == pytest.approx(190.0)
        assert o.source == "broker_reconcile"


def test_existing_submitted_row_flipped_to_filled(db_session):
    from app.services.reconciliation.order_sync import sync_broker_orders_once

    with db_session() as db:
        db.add(Order(
            broker="schwab", broker_order_id="T-2", symbol="MSFT", side="SELL",
            order_type="TRAILING_STOP", quantity=5, status="submitted", is_paper=False,
        ))
        db.commit()

    broker = _FakeBroker([_bo("T-2", "MSFT", "SELL", 5, "filled", fill_price=410.0)])
    patches = _patch_env(db_session, broker)
    with patches[0], patches[1], patches[2], patches[3]:
        result = sync_broker_orders_once()

    assert result["updated"] == 1
    with db_session() as db:
        o = db.query(Order).filter_by(broker_order_id="T-2").one()
        assert o.status == "filled"
        assert o.fill_price == pytest.approx(410.0)


def test_orphan_fill_pairs_into_realized_trade(db_session):
    """An ingested SELL fill must pair with an existing filled BUY so FIFO
    produces a realized round-trip (this is what makes PnL update)."""
    import app.services.reconciliation.order_sync as mod
    from app.services.reconciliation.order_sync import sync_broker_orders_once
    from app.services.pnl.fifo import compute_fifo

    # Seed a filled BUY so the orphan SELL has something to close.
    with db_session() as db:
        db.add(Order(
            broker="schwab", broker_order_id="B-1", symbol="NVDA", side="BUY",
            order_type="MARKET", quantity=10, status="filled", is_paper=False,
            fill_price=100.0,
        ))
        db.commit()

    broker = _FakeBroker([_bo("S-1", "NVDA", "SELL", 10, "filled", fill_price=120.0)])
    # Let the REAL sync_realized_trades run against our in-memory DB.
    with patch.object(mod, "SessionLocal", db_session), \
         patch.object(mod, "get_broker", lambda: broker), \
         patch("app.services.strategy.scheduler._has_india_assignments", lambda: False):
        result = sync_broker_orders_once()

    assert result["created"] == 1
    with db_session() as db:
        fifo = compute_fifo(db)
        assert len(fifo.closed) == 1
        t = fifo.closed[0]
        assert t.symbol == "NVDA"
        assert t.realized_pnl == pytest.approx((120.0 - 100.0) * 10)


def test_non_fill_status_creates_no_row(db_session):
    from app.services.reconciliation.order_sync import sync_broker_orders_once

    broker = _FakeBroker([_bo("W-1", "TSLA", "SELL", 3, "working")])
    patches = _patch_env(db_session, broker)
    with patches[0], patches[1], patches[2], patches[3]:
        result = sync_broker_orders_once()

    assert result["created"] == 0
    assert result["updated"] == 0
    with db_session() as db:
        assert db.query(Order).filter_by(broker_order_id="W-1").count() == 0
