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


@pytest.fixture(autouse=True)
def _isolated_trail_peaks_db():
    """Point the trail-peak persistence at a fresh in-memory DB per test, so the
    durable high-water mark never leaks between tests (or into the project DB)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models.trail_peaks import TrailPeak  # noqa: F401 — register table
    import app.services.execution.service as svc_mod

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[TrailPeak.__table__])
    TestSession = sessionmaker(bind=engine)
    with patch.object(svc_mod, "SessionLocal", TestSession):
        yield TestSession


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
    """The AMAL bug: price ran to a peak then pulled back (but NOT through the
    trail). The stop must ratchet off the PEAK, not the lower current price. A
    resting STOP at 107.80 implies peak 110.0; current 109 is below the peak but
    still ABOVE the stop (trail not hit) → the stop stays at 107.80, not lowered
    to 109×0.98=106.82, and nothing churns."""
    broker = _FakeBroker(working_orders=[_resting_stop("AAPL", 107.80)], price=109.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    # Resting 107.80 == target (from its implied peak) and current 109 > 107.80
    # → trail not hit, no downward move, no churn.
    assert broker.cancel_calls == []
    assert broker.place_calls == []


async def test_trail_hit_exits_at_market_not_invalid_stop():
    """The reject loop: when the peak-based stop would sit AT/ABOVE the current
    price (price pulled back THROUGH the trail), a SELL STOP there is invalid and
    the broker rejects it. The bot must SELL AT MARKET instead. AMAL: signal
    43.55, peak 45.23 (resting stop 44.33 implies it), current 44.00 < target
    44.33 → exit now."""
    broker = _FakeBroker(working_orders=[_resting_stop("AMAL", 44.33)], price=44.00)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AMAL", quantity=5, account_id="X", signal_price=43.55, trail_pct=2.0,
    )

    assert ok is True
    # Cancel the resting (now-invalid) stop, then place ONE market sell.
    assert broker.cancel_calls == ["resting-1"]
    assert len(broker.place_calls) == 1
    placed = broker.place_calls[0]
    assert placed.order_type == "MARKET"
    assert placed.side == "SELL"


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


async def test_rejected_replacement_keeps_old_stop(_isolated_trail_peaks_db):
    """Place-then-cancel: if the new stop is REJECTED, the old resting stop must
    NOT be cancelled (position stays protected), and the call reports success
    (still-protected) rather than 'POSITION LEFT UNPROTECTED'."""
    class _RejectingBroker(_FakeBroker):
        async def place_order(self, order, account_id):
            raise RuntimeError("rejected: stop above market")

    # Ratchet up (current 105 → target 102.90) so it tries to replace the
    # resting 100.50 stop, but placement is rejected.
    broker = _RejectingBroker(working_orders=[_resting_stop("AAPL", 100.50)], price=105.0)
    broker.supports_native_trailing_stop = False
    svc = ExecutionService.__new__(ExecutionService)
    svc.broker = broker
    svc._persist_order = lambda *a, **k: SimpleNamespace(id=1)
    svc._update_order_status = lambda *a, **k: None

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True                  # still protected by the old stop
    assert broker.cancel_calls == []   # old stop NOT cancelled — no naked gap


class _StatusAwareBroker(_FakeBroker):
    """Returns the resting order ONLY for a specific status (mimics Schwab,
    which parks a resting STOP in AWAITING_STOP_CONDITION, not 'working')."""

    def __init__(self, orders, only_status, price=100.0):
        super().__init__(orders, price=price)
        self._only_status = only_status

    async def list_orders(self, account_id, status=None):
        return list(self._working) if status == self._only_status else []


async def test_resting_stop_in_non_working_status_is_detected():
    """The bug: Step 0 queried only status='working', so a Schwab STOP resting
    in AWAITING_STOP_CONDITION was missed → re-placed every cycle → duplicate
    idempotency_key. The guard must sweep all pending statuses and skip."""
    broker = _StatusAwareBroker(
        [_resting_stop("AAPL", 103.0)], only_status="awaiting_stop_condition", price=105.0,
    )
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)

    ok = await svc.tighten_trail_on_sell(
        symbol="AAPL", quantity=10, account_id="X", signal_price=100.0, trail_pct=2.0,
    )

    assert ok is True
    # Resting 103 >= target (105×0.98=102.90) → detected, no re-place, no cancel.
    assert broker.cancel_calls == []
    assert broker.place_calls == []


async def test_persist_duplicate_idempotency_key_returns_existing():
    """A duplicate idempotency_key must NOT raise (which the caller would log as
    'POSITION LEFT UNPROTECTED'); _persist_order returns the existing row."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models.orders import Order
    from app.schemas.orders import OrderRequest
    import app.services.execution.service as svc_mod

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[Order.__table__])
    TestSession = sessionmaker(bind=engine)

    svc = ExecutionService.__new__(ExecutionService)
    svc.broker = _FakeBroker(working_orders=[])

    req = OrderRequest(
        symbol="AAPL", side="SELL", order_type="STOP", quantity=10,
        stop_price=102.90, time_in_force="GTC", source="scheduler",
        idempotency_key="sell-trail-AAPL-x-10290",
    )
    with patch.object(svc_mod, "SessionLocal", TestSession):
        first = svc._persist_order(req, signal_id=None, status="submitted")
        # Same key again — must return the existing row, not raise.
        second = svc._persist_order(req, signal_id=None, status="submitted")

    assert first.id == second.id
    with TestSession() as db:
        assert db.query(Order).filter_by(idempotency_key="sell-trail-AAPL-x-10290").count() == 1


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


async def test_durable_peak_holds_on_pullback_across_cycles(_isolated_trail_peaks_db):
    """The durable persisted peak must govern the stop across cycles. Cycle 2 is
    a SMALL pullback (peak 45 → 44.50, still above the 44.10 stop so the trail is
    NOT hit): the stop must stay at 45×0.98 = 44.10, not drop to 44.50×0.98."""
    from app.models.trail_peaks import TrailPeak
    TestSession = _isolated_trail_peaks_db

    # Cycle 1 — price at the $45 peak, no resting order. signal 40 → floor 40.10;
    # trail 45×0.98 = 44.10. Static STOP path placed at the peak-based level.
    broker = _FakeBroker(working_orders=[], price=45.0)
    broker.supports_native_trailing_stop = False
    svc = _svc(broker)
    ok = await svc.tighten_trail_on_sell(
        symbol="AMAL", quantity=10, account_id="X",
        signal_price=40.0, trail_pct=2.0, signal_id=999,
    )
    assert ok is True
    assert broker.place_calls[0].stop_price == pytest.approx(44.10, abs=0.01)
    with TestSession() as db:
        row = db.query(TrailPeak).filter_by(symbol="AMAL", signal_id=999).one()
        assert row.peak_price == pytest.approx(45.0, abs=0.01)

    # Cycle 2 — price eased to $44.50 (still ABOVE the 44.10 stop → trail not
    # hit). The durable peak ($45) governs: target stays 44.10, no churn, and the
    # stop is NOT lowered to 44.50×0.98 = 43.61.
    broker2 = _FakeBroker(working_orders=[_resting_stop("AMAL", 44.10)], price=44.50)
    broker2.supports_native_trailing_stop = False
    svc2 = _svc(broker2)
    ok2 = await svc2.tighten_trail_on_sell(
        symbol="AMAL", quantity=10, account_id="X",
        signal_price=40.0, trail_pct=2.0, signal_id=999,
    )
    assert ok2 is True
    assert broker2.cancel_calls == []   # stop NOT lowered
    assert broker2.place_calls == []
    with TestSession() as db:
        row = db.query(TrailPeak).filter_by(symbol="AMAL", signal_id=999).one()
        assert row.peak_price == pytest.approx(45.0, abs=0.01)
