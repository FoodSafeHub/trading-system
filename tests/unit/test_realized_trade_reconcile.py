"""sync_realized_trades must RECONCILE per SELL, not just insert-new.

FIFO pairing depends on the full BUY history. If a backdated BUY is ingested
later (e.g. an order reconcile/backfill pulls in a >30-day-old lot), the SAME
sell re-pairs against different buy lots/quantities. A pure insert-only sync
would leave the old pairing rows alongside the new ones — double-counting closed
shares and overstating realized P&L (the AMAL bug: a 12-share sell showed 17
closed shares). This pins the self-healing behaviour.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orders import Order
from app.models.realized_trades import RealizedTrade
from app.services.pnl.store import sync_realized_trades


@pytest.fixture
def db_session():
    from app.models import orders, realized_trades, signals  # noqa: F401
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def _order(oid, boid, sym, side, qty, price, at):
    return Order(
        id=oid, broker="schwab", broker_order_id=boid, symbol=sym, side=side,
        order_type="MARKET", quantity=qty, status="filled", is_paper=False,
        fill_price=price, filled_at=at,
    )


def test_backdated_buy_repairs_without_double_count(db_session):
    t_jun3 = datetime(2026, 6, 3, tzinfo=timezone.utc)
    t_jun15 = datetime(2026, 6, 15, tzinfo=timezone.utc)
    t_may19 = datetime(2026, 5, 19, tzinfo=timezone.utc)

    # Phase 1: only the June BUY (12) + the June SELL (12) exist.
    with db_session() as db:
        db.add(_order(38, "b-38", "AMAL", "BUY", 12, 41.275, t_jun3))
        db.add(_order(103, "s-103", "AMAL", "SELL", 12, 44.2388, t_jun15))
        db.commit()
        sync_realized_trades(db)

    with db_session() as db:
        rows = db.query(RealizedTrade).filter_by(symbol="AMAL").all()
        assert sum(r.quantity for r in rows) == pytest.approx(12.0)  # 12 closed

    # Phase 2: a BACKDATED May BUY (5) is ingested (the backfill case). FIFO now
    # drains May(5)+June(7) against the same 12-share sell. The table must
    # RECONCILE to 12 closed shares total, not 17.
    with db_session() as db:
        db.add(_order(132, "b-132", "AMAL", "BUY", 5, 40.43, t_may19))
        db.commit()
        sync_realized_trades(db)

    with db_session() as db:
        rows = db.query(RealizedTrade).filter_by(symbol="AMAL").all()
        total_qty = sum(r.quantity for r in rows)
        total_pnl = sum(r.realized_pnl for r in rows)
        assert total_qty == pytest.approx(12.0)        # NOT 17 — no double count
        # 5@40.43 + 7@41.275 sold @44.2388
        expected = (44.2388 - 40.43) * 5 + (44.2388 - 41.275) * 7
        assert total_pnl == pytest.approx(expected, abs=0.01)
        # And the open lot is the remaining 5 of the June lot.
        # (verified via compute_fifo open_lots)
        from app.services.pnl.fifo import compute_fifo
        open_amal = [l for l in compute_fifo(db).open_lots if l.symbol == "AMAL"]
        assert len(open_amal) == 1
        assert open_amal[0].quantity == pytest.approx(5.0)
        assert open_amal[0].buy_price == pytest.approx(41.275)


def test_stable_history_does_not_churn(db_session):
    """When nothing changed, a second sync must not delete/re-insert rows."""
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with db_session() as db:
        db.add(_order(1, "b1", "KO", "BUY", 5, 78.665, t0))
        db.add(_order(2, "s2", "KO", "SELL", 5, 81.03, t0 + timedelta(days=10)))
        db.commit()
        sync_realized_trades(db)
        ids_before = sorted(r.id for r in db.query(RealizedTrade).all())

    with db_session() as db:
        inserted, _ = sync_realized_trades(db)
        ids_after = sorted(r.id for r in db.query(RealizedTrade).all())

    assert inserted == 0          # no churn
    assert ids_before == ids_after  # same rows, not re-created
