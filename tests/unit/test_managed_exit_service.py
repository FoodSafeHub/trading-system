"""Managed-mode behavior of ExecutionService (bot-managed exits).

Pins the core promise: with schwab_managed_exits_enabled=True NO protective
order is ever placed at the broker —
  - tighten_trail_on_sell above the arm gate records a software trail
    (trail_armed + target_stop) and places nothing; stray resting stops are
    cancelled (each one is sweep risk);
  - repeated cycles stay order-free and the software target only ratchets up;
  - the trail-hit branch still market-sells exactly once (the ONLY executor);
  - a RED position at SELL time is red-held: no order at all;
  - _submit_protective_stop after a BUY fill places nothing and records
    monitoring.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.schemas.account import Quote
from app.schemas.orders import OrderRequest, OrderStatusResponse
from app.services.execution.service import ExecutionService


class _FakeBroker:
    name = "fake"
    supports_native_trailing_stop = True

    def __init__(self, working_orders=None, price=100.0):
        self._working = list(working_orders or [])
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

    async def get_order(self, broker_order_id, account_id):
        return OrderStatusResponse(
            broker_order_id=broker_order_id, symbol="AAPL", side="SELL",
            order_type="MARKET", quantity=1, status="filled",
        )


def _svc(broker):
    svc = ExecutionService.__new__(ExecutionService)
    svc.broker = broker
    svc._persist_order = lambda *a, **k: SimpleNamespace(id=1)
    svc._update_order_status = lambda *a, **k: None
    svc._handle_fill = lambda *a, **k: None

    async def _confirm(*a, **k):
        return None

    svc._confirm_broker_status = _confirm
    return svc


@pytest.fixture(autouse=True)
def _no_ohlcv():
    with patch("app.services.market_data.provider.get_ohlcv",
               side_effect=Exception("no data")):
        yield


@pytest.fixture(autouse=True)
def _isolated_db():
    """Fresh in-memory DB (trail peaks + managed state) per test, patched into
    both the execution service and the managed-exit engine."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models.managed_exit_state import ManagedExitState
    from app.models.trail_peaks import TrailPeak
    import app.services.execution.managed_exit_engine as eng_mod
    import app.services.execution.service as svc_mod

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[TrailPeak.__table__, ManagedExitState.__table__],
    )
    TestSession = sessionmaker(bind=engine)
    with patch.object(svc_mod, "SessionLocal", TestSession), \
         patch.object(eng_mod, "SessionLocal", TestSession):
        yield TestSession


@pytest.fixture()
def managed_on(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "schwab_managed_exits_enabled", True)
    monkeypatch.setattr(s, "red_hold_enabled", True)
    with patch("app.services.markets.is_india_symbol", return_value=False):
        yield s


def _get_state(TestSession, symbol):
    from app.models.managed_exit_state import ManagedExitState
    with TestSession() as db:
        return (
            db.query(ManagedExitState)
            .filter(ManagedExitState.symbol == symbol)
            .one_or_none()
        )


def _green(symbol="AAPL"):
    """FIFO cost far below the quote so the red-hold gate stays out of the way."""
    return patch(
        "app.services.execution.managed_exit_engine.fifo_avg_costs",
        return_value={symbol: 1.0},
    )


async def test_managed_mode_places_no_orders_and_records_trail(
    managed_on, _isolated_db,
):
    # signal 100, current 103 → armed; trail should live in DB, not at broker.
    broker = _FakeBroker(price=103.0)
    svc = _svc(broker)
    with _green():
        ok = await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
    assert ok is True
    assert broker.place_calls == []          # the whole point: nothing rests
    st = _get_state(_isolated_db, "AAPL")
    assert st is not None and st.mode == "trail_armed"
    # peak 103 × 0.98 = 100.94 > floor 100 → target = trail level.
    assert st.target_stop == pytest.approx(100.94, abs=0.01)


async def test_managed_mode_cancels_stray_resting_stops(managed_on, _isolated_db):
    stray = SimpleNamespace(
        symbol="AAPL", side="SELL", order_type="STOP",
        broker_order_id="stray-1", stop_price=99.0,
    )
    broker = _FakeBroker(working_orders=[stray], price=103.0)
    svc = _svc(broker)
    with _green():
        ok = await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
    assert ok is True
    assert broker.cancel_calls == ["stray-1"]   # sweep risk removed
    assert broker.place_calls == []


async def test_managed_mode_repeated_cycles_stay_order_free_and_ratchet_up(
    managed_on, _isolated_db,
):
    broker = _FakeBroker(price=103.0)
    svc = _svc(broker)
    with _green():
        await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
        first = _get_state(_isolated_db, "AAPL").target_stop
        # Price runs up → target must ratchet UP; still zero broker orders.
        broker._price = 110.0
        await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
        second = _get_state(_isolated_db, "AAPL").target_stop
        # Pullback (above the trail) → target holds, does NOT drop.
        broker._price = 109.0
        await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
        third = _get_state(_isolated_db, "AAPL").target_stop
    assert broker.place_calls == []
    assert second > first
    assert third == pytest.approx(second)


async def test_trail_hit_market_sells_once(managed_on, _isolated_db):
    broker = _FakeBroker(price=103.0)
    svc = _svc(broker)
    with _green():
        await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
        assert broker.place_calls == []
        # Reversal through the software trail (target ~100.94) → market exit.
        broker._price = 100.5
        ok = await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
    assert ok is True
    assert len(broker.place_calls) == 1
    assert broker.place_calls[0].order_type == "MARKET"
    assert broker.place_calls[0].side == "SELL"
    assert _get_state(_isolated_db, "AAPL").mode == "exited"


async def test_red_position_is_held_no_orders(managed_on, _isolated_db):
    # Cost 120 vs current 103 → RED: no trail, no market sell, nothing.
    broker = _FakeBroker(price=103.0)
    svc = _svc(broker)
    with patch(
        "app.services.execution.managed_exit_engine.fifo_avg_costs",
        return_value={"AAPL": 120.0},
    ), patch("app.services.notifications.bus.notify_suppression") as mock_n:
        ok = await svc.tighten_trail_on_sell(
            symbol="AAPL", quantity=10, account_id="X",
            signal_price=100.0, trail_pct=2.0,
        )
    assert ok is True
    assert broker.place_calls == []
    assert broker.cancel_calls == []
    st = _get_state(_isolated_db, "AAPL")
    assert st.mode == "red_hold"
    assert mock_n.call_count == 1


async def test_protective_stop_skipped_in_managed_mode(managed_on, _isolated_db, monkeypatch):
    # Even with auto protective stops enabled, managed mode rests nothing.
    monkeypatch.setattr(get_settings(), "auto_protective_stop_enabled", True)
    broker = _FakeBroker(price=100.0)
    svc = _svc(broker)
    buy = OrderRequest(symbol="AAPL", side="BUY", order_type="MARKET",
                       quantity=10, source="scheduler", idempotency_key="buy-1")
    fill = OrderStatusResponse(
        broker_order_id="b1", symbol="AAPL", side="BUY", order_type="MARKET",
        quantity=10, filled_quantity=10, fill_price=100.0, status="filled",
    )
    await svc._submit_protective_stop(buy, "X", fill)
    assert broker.place_calls == []
    st = _get_state(_isolated_db, "AAPL")
    assert st is not None and st.mode == "monitoring"
    assert st.avg_cost == 100.0


async def test_flag_off_keeps_legacy_native_path(_isolated_db, monkeypatch):
    # Master switch off → original behavior (native trail placed on BUY fill).
    s = get_settings()
    monkeypatch.setattr(s, "schwab_managed_exits_enabled", False)
    monkeypatch.setattr(s, "auto_protective_stop_enabled", True)
    monkeypatch.setattr(s, "trailing_stop_enabled", True)
    broker = _FakeBroker(price=100.0)
    svc = _svc(broker)
    buy = OrderRequest(symbol="AAPL", side="BUY", order_type="MARKET",
                       quantity=10, source="scheduler", idempotency_key="buy-1")
    fill = OrderStatusResponse(
        broker_order_id="b1", symbol="AAPL", side="BUY", order_type="MARKET",
        quantity=10, filled_quantity=10, fill_price=100.0, status="filled",
    )
    await svc._submit_protective_stop(buy, "X", fill)
    assert len(broker.place_calls) == 1
    assert broker.place_calls[0].order_type == "TRAILING_STOP"
