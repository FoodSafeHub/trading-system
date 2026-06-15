"""Behavior + idempotency guard for ExecutionService.tighten_trail_on_sell.

Approach C: after the assigned strategy's SELL signal the bot does NOT market-
sell. It waits until price clears the arm gate (signal + floor_buffer_pct%), then
places/ratchets a bot-managed STOP floored at signal + floor_buffer_pct% so a
reversal still exits in profit. The scheduler re-evaluates every cycle, so the
call must be idempotent: only ratchet the stop UP, never churn cancel/replace.

These tests pin:
  - Below the arm gate → pending arm: returns True, places nothing.
  - Above the arm gate, unprotected → places a floored STOP.
  - Floor is a hard minimum: stop = max(price − trail%, signal × (1+buf%)).
  - A resting STOP already at/above the target → no-op (no ratchet churn).
  - A higher target → cancel the resting stop and re-place (ratchet up).
  - force_replace=True always cancels and re-places.
  - signal_price=0.0 falls back to the current quote (floor stays meaningful).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.schemas.account import Quote
from app.schemas.orders import OrderStatusResponse
from app.services.execution.service import ExecutionService


class _FakeBroker:
    """Minimal broker that records cancel/place calls, returns a fixed set of
    working orders, and quotes a configurable current price."""

    name = "fake"
    supports_native_trailing_stop = True

    def __init__(self, working_orders, price=100.0):
        self._working = working_orders
        self._price = price
        self.cancel_calls = []
        self.place_calls = []

    async def list_orders(self, account_id, status=None):
        return list(self._working)

    async def cancel_order(self, broker_order_id, account_id):
        self.cancel_calls.append(broker_order_id)
        return True

    async def get_quotes(self, symbols):
        return {s.upper(): Quote(symbol=s.upper(), last=self._price) for s in symbols}

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


def _resting_stop(symbol="AAPL", stop_price=100.0):
    return SimpleNamespace(
        symbol=symbol, side="SELL", order_type="STOP",
        broker_order_id="resting-1", stop_price=stop_price,
    )


def _resting_native(symbol="AAPL"):
    return SimpleNamespace(
        symbol=symbol, side="SELL", order_type="TRAILING_STOP",
        broker_order_id="resting-native", stop_price=None,
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


async def test_pending_arm_below_gate_places_nothing():
    # signal 100 → floor/arm gate = 100.25; current 100.0 < gate → pending arm.
    broker = _FakeBroker(working_orders=[], price=100.0)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0,
    )

    assert ok is True                  # pending arm is not a failure
    assert broker.place_calls == []    # nothing placed below the gate
    assert broker.cancel_calls == []


async def test_floored_stop_while_trail_below_floor():
    # Just above the gate: current 100.30, trail 100.30×0.98=98.29 < floor 100.25
    # → still bot-managed static STOP, and the floor governs (stop = 100.25),
    # not the sub-floor trail level. No native handoff yet.
    broker = _FakeBroker(working_orders=[], price=100.30)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert len(broker.place_calls) == 1
    placed = broker.place_calls[0]
    assert placed.order_type == "STOP"
    assert placed.stop_price == pytest.approx(100.25, abs=0.01)  # floor wins


async def test_hands_off_to_native_when_trail_clears_floor():
    # current 105: trail level 105×0.98=102.90 ≥ floor 100.25 → hand off to a
    # broker-native TRAILING_STOP (tick-by-tick, can't violate the floor now).
    broker = _FakeBroker(working_orders=[], price=105.0)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert len(broker.place_calls) == 1
    placed = broker.place_calls[0]
    assert placed.order_type == "TRAILING_STOP"
    assert placed.trail_value == pytest.approx(2.0)


async def test_no_native_broker_stays_static_above_floor():
    # Same prices, but broker has NO native trail → must stay on a floored STOP.
    broker = _FakeBroker(working_orders=[], price=105.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert broker.place_calls[0].order_type == "STOP"
    assert broker.place_calls[0].stop_price == pytest.approx(102.90, abs=0.01)


async def test_healthy_native_left_alone():
    # A native trail already resting + trail level above floor → leave it (the
    # broker self-ratchets; re-placing would reset its trigger to current price).
    broker = _FakeBroker(working_orders=[_resting_native("AAPL")], price=105.0)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert broker.cancel_calls == []
    assert broker.place_calls == []


async def test_no_ratchet_when_resting_static_at_or_above_target():
    # No native broker; resting STOP 103 already >= target (105×0.98=102.90).
    broker = _FakeBroker(working_orders=[_resting_stop("AAPL", 103.0)], price=105.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert broker.cancel_calls == []   # nothing cancelled
    assert broker.place_calls == []    # nothing re-placed


async def test_trail_measured_from_peak_not_current_on_pullback():
    """The AMAL bug: price ran to a peak then pulled back. The stop must ratchet
    off the PEAK, not the (lower) current price. With OHLCV unavailable (autouse
    patch), the peak is reconstructed from the resting STOP's implied peak:
    a resting STOP at 107.80 implies peak 107.80/0.98=110.0, so even though the
    current price is only 104, the trail level stays 110×0.98=107.80 — the stop
    is NOT lowered to 104×0.98=101.92."""
    broker = _FakeBroker(working_orders=[_resting_stop("AAPL", 107.80)], price=104.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    # Resting 107.80 already == target reconstructed from its own implied peak →
    # no downward move, no churn. The key assertion: nothing was lowered.
    assert broker.cancel_calls == []
    assert broker.place_calls == []


async def test_ratchets_up_when_target_higher():
    # No native broker; resting STOP 100.50 < target 102.90 → ratchet up.
    # (resting 100.50 implies peak 102.55; current 105 implies peak 105 — the
    # higher current wins, target = 105×0.98 = 102.90.)
    broker = _FakeBroker(working_orders=[_resting_stop("AAPL", 100.50)], price=105.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    assert broker.cancel_calls == ["resting-1"]
    assert len(broker.place_calls) == 1
    assert broker.place_calls[0].stop_price == pytest.approx(102.90, abs=0.01)


async def test_force_replace_cancels_and_replaces():
    broker = _FakeBroker(working_orders=[_resting_native("AAPL")], price=105.0)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0,
        trail_pct=2.0, force_replace=True,
    )

    assert ok is True
    assert broker.cancel_calls == ["resting-native"]   # replaced despite healthy
    assert len(broker.place_calls) == 1


async def test_zero_signal_price_falls_back_to_current():
    """A missing signal_price (0.0) must NOT silently zero out the floor — it
    falls back to the current quote. With signal==current, the arm gate isn't
    cleared yet (current == floor base, < floor), so it's pending arm: returns
    True, places nothing. The point is the floor stays meaningful, not zeroed."""
    broker = _FakeBroker(working_orders=[], price=100.0)
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=0.0,
    )

    assert ok is True
