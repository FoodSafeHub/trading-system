"""Idempotency guard for ExitService.tighten_trail_on_sell.

The scheduler re-evaluates every ~30s and a SELL condition usually persists for
many cycles, so tighten_trail_on_sell is called repeatedly for the same symbol.
Before the guard, each call cancelled the resting SELL stop and re-placed a fresh
one — resetting a broker-native trail's ratchet back to the current price every
cycle (so it could never climb) and churning cancel/replace orders on Zerodha.

These tests pin the corrected behavior:
  - If a healthy resting SELL STOP/TRAILING_STOP already exists, the call is a
    no-op (no cancel, no place) and returns True.
  - With force_replace=True, the resting stop IS cancelled and re-placed.
  - With NO resting stop, a fresh trail is placed as normal.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.schemas.account import Quote
from app.schemas.orders import OrderStatusResponse
from app.services.execution.service import ExecutionService


class _FakeBroker:
    """Minimal broker that records cancel/place calls and returns a fixed
    set of working orders. supports_native_trailing_stop=True so the native
    path is exercised."""

    name = "fake"
    supports_native_trailing_stop = True

    def __init__(self, working_orders):
        self._working = working_orders
        self.cancel_calls = []
        self.place_calls = []

    async def list_orders(self, account_id, status=None):
        return list(self._working)

    async def cancel_order(self, broker_order_id, account_id):
        self.cancel_calls.append(broker_order_id)
        return True

    async def get_quotes(self, symbols):
        return {s.upper(): Quote(symbol=s.upper(), last=100.0) for s in symbols}

    async def place_order(self, order, account_id):
        self.place_calls.append(order)
        return OrderStatusResponse(
            broker_order_id="new-123",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            status="submitted",
        )


def _resting_trail(symbol="AAPL"):
    return SimpleNamespace(
        symbol=symbol, side="SELL", order_type="TRAILING_STOP",
        broker_order_id="resting-1", stop_price=None,
    )


def _svc(broker):
    svc = ExecutionService.__new__(ExecutionService)
    svc.broker = broker
    # Stub persistence so we don't touch the DB.
    svc._persist_order = lambda *a, **k: SimpleNamespace(id=1)
    svc._update_order_status = lambda *a, **k: None
    return svc


@pytest.fixture(autouse=True)
def _no_ohlcv():
    # Keep peak lookup from hitting the network/provider.
    with patch("app.services.market_data.provider.get_ohlcv", side_effect=Exception("no data")):
        yield


async def test_skips_when_already_protected():
    broker = _FakeBroker(working_orders=[_resting_trail("AAPL")])
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0,
    )

    assert ok is True
    assert broker.cancel_calls == []   # nothing cancelled
    assert broker.place_calls == []    # nothing re-placed — existing trail kept


async def test_force_replace_cancels_and_replaces():
    broker = _FakeBroker(working_orders=[_resting_trail("AAPL")])
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0,
        force_replace=True,
    )

    assert ok is True
    assert broker.cancel_calls == ["resting-1"]   # old stop cancelled
    assert len(broker.place_calls) == 1           # fresh trail placed


async def test_places_fresh_when_unprotected():
    broker = _FakeBroker(working_orders=[])  # nothing resting
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0,
    )

    assert ok is True
    assert broker.cancel_calls == []
    assert len(broker.place_calls) == 1


async def test_zero_signal_price_falls_back_to_current():
    """A missing signal_price (0.0) must NOT silently zero out the floor — it
    falls back to the current quote so the floor stays meaningful, and an order
    is still placed."""
    broker = _FakeBroker(working_orders=[])  # current quote = 100.0
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=0.0,
    )

    assert ok is True
    assert len(broker.place_calls) == 1  # protective order placed despite 0 signal
