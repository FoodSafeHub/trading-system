from __future__ import annotations

"""Persistence layer for realized round-trips.

Pass 1 recomputed FIFO from the orders table on every /pnl/* call. That's
fine for small histories but scales linearly with the order count. Pass 2
materializes each round-trip into `realized_trades` once and lets the
endpoints read directly from that table.

The sync is incremental: we run FIFO over current orders, then INSERT only
the (sell_order_id, buy_order_id) pairs that aren't already in the table.
A SELL that drains two BUY lots produces two rows — that's expected, the
unique constraint catches duplicates if the sync is called twice.
"""

import logging

from sqlalchemy.orm import Session

from app.models.realized_trades import RealizedTrade
from app.services.pnl.fifo import ClosedTrade, FifoResult, compute_fifo

logger = logging.getLogger(__name__)


def sync_realized_trades(db: Session) -> tuple[int, FifoResult]:
    """Materialize any new FIFO round-trips into `realized_trades`.

    Returns (rows_inserted, fifo_result). The FifoResult is reused by callers
    that also need open lots (e.g. /pnl/summary's unrealized leg).
    """
    fifo = compute_fifo(db)

    if not fifo.closed:
        return 0, fifo

    existing: set[tuple[int, int]] = {
        (row.sell_order_id, row.buy_order_id)
        for row in db.query(
            RealizedTrade.sell_order_id, RealizedTrade.buy_order_id
        ).all()
    }

    inserted = 0
    for t in fifo.closed:
        key = (t.sell_order_id, t.buy_order_id)
        if key in existing:
            continue
        db.add(_to_row(t))
        inserted += 1

    if inserted:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("[pnl] realized_trades commit failed: %s", exc)
            return 0, fifo

    return inserted, fifo


def _to_row(t: ClosedTrade) -> RealizedTrade:
    return RealizedTrade(
        symbol=t.symbol,
        quantity=t.quantity,
        buy_price=t.buy_price,
        sell_price=t.sell_price,
        buy_at=t.buy_at,
        sell_at=t.sell_at,
        realized_pnl=t.realized_pnl,
        realized_pct=t.realized_pct,
        hold_days=t.hold_days,
        broker=t.broker,
        buy_order_id=t.buy_order_id,
        sell_order_id=t.sell_order_id,
        buy_signal_id=t.buy_signal_id,
        sell_signal_id=t.sell_signal_id,
        buy_strategy=t.buy_strategy,
        sell_strategy=t.sell_strategy,
        is_paper=t.is_paper,
    )


def row_to_closed_trade(row: RealizedTrade) -> ClosedTrade:
    """Reverse projection so aggregate.py helpers can stay ClosedTrade-shaped."""
    return ClosedTrade(
        symbol=row.symbol,
        quantity=row.quantity,
        buy_price=row.buy_price,
        sell_price=row.sell_price,
        buy_at=row.buy_at,
        sell_at=row.sell_at,
        realized_pnl=row.realized_pnl,
        realized_pct=row.realized_pct,
        hold_days=row.hold_days,
        broker=row.broker,
        buy_order_id=row.buy_order_id,
        sell_order_id=row.sell_order_id,
        buy_signal_id=row.buy_signal_id,
        sell_signal_id=row.sell_signal_id,
        buy_strategy=row.buy_strategy,
        sell_strategy=row.sell_strategy,
        is_paper=row.is_paper,
    )


def load_closed_trades(db: Session) -> list[ClosedTrade]:
    """Read every persisted round-trip back as ClosedTrades, oldest first."""
    rows = db.query(RealizedTrade).order_by(RealizedTrade.sell_at.asc()).all()
    return [row_to_closed_trade(r) for r in rows]
