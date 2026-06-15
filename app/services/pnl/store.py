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
    """Materialize FIFO round-trips into `realized_trades`, RECONCILING per SELL.

    Returns (rows_inserted, fifo_result). The FifoResult is reused by callers
    that also need open lots (e.g. /pnl/summary's unrealized leg).

    Why reconcile and not just insert-new: FIFO pairing depends on the full
    BUY history. If a BACKDATED buy is ingested later (e.g. a >30-day-old lot
    pulled in by an order reconcile/backfill), the SAME sell re-pairs against
    different buy lots and quantities. A pure insert-only sync then leaves the
    OLD pairing rows in place alongside the new ones — double-counting the
    closed quantity and overstating realized P&L (the AMAL bug: a 12-share sell
    showed 17 closed shares). So for every SELL present in the fresh FIFO walk
    we DELETE its persisted rows and re-insert the current pairing — the table
    is then always exactly what the FIFO walk produces.
    """
    fifo = compute_fifo(db)

    if not fifo.closed:
        return 0, fifo

    # Group the fresh closed trades by their sell_order_id.
    fresh_by_sell: dict[int, list[ClosedTrade]] = {}
    for t in fifo.closed:
        fresh_by_sell.setdefault(t.sell_order_id, []).append(t)

    # Persisted (sell, buy, qty) per affected sell — detect divergence so we
    # only churn rows that actually changed (avoids rewriting the whole table
    # every call).
    affected_sells = list(fresh_by_sell.keys())
    persisted = (
        db.query(RealizedTrade)
        .filter(RealizedTrade.sell_order_id.in_(affected_sells))
        .all()
    )
    persisted_by_sell: dict[int, list[RealizedTrade]] = {}
    for r in persisted:
        persisted_by_sell.setdefault(r.sell_order_id, []).append(r)

    def _sig(rows, qty_attr="quantity", buy_attr="buy_order_id"):
        # Order-independent signature of the (buy_order_id, rounded qty) pairs.
        return sorted(
            (getattr(x, buy_attr), round(float(getattr(x, qty_attr)), 6)) for x in rows
        )

    inserted = 0
    changed = False
    to_add: list = []
    for sell_id, fresh_rows in fresh_by_sell.items():
        old_rows = persisted_by_sell.get(sell_id, [])
        if _sig(old_rows) == _sig(fresh_rows):
            continue  # already consistent — no churn
        # Diverged (or new): replace this sell's rows with the fresh pairing.
        for r in old_rows:
            db.delete(r)
        for t in fresh_rows:
            to_add.append(_to_row(t))
            inserted += 1
        changed = True

    if changed:
        # Flush the DELETEs before the INSERTs so a re-paired (sell, buy) key
        # doesn't collide with the stale row under the UNIQUE(sell,buy)
        # constraint (the old row is gone by the time we insert the new one).
        db.flush()
        for row in to_add:
            db.add(row)
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
