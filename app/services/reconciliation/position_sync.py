from __future__ import annotations

"""Position reconciliation — snapshot the broker's live positions to the DB.

The `position_snapshots` table had no writer; nothing ever recorded what the
broker actually held versus what our orders implied. This job closes that gap:
on a schedule (and on demand) it pulls `broker.get_positions()` for the active
broker and writes one row per holding into `position_snapshots`, tagged with
the broker name and a single batch timestamp.

It is read-only against the broker (no orders placed) and append-only against
the DB, so it is safe to run frequently. Drift between snapshots and our own
order-derived position view is left for a dashboard/alert to surface — this
module's only job is to make the broker's truth durable.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.positions import PositionSnapshot
from app.services.brokers.factory import get_broker

logger = logging.getLogger(__name__)


def sync_positions_once() -> int:
    """Fetch live positions from the active broker and persist a snapshot batch.

    Returns the number of position rows written. Never raises — a broker or DB
    error is logged and 0 is returned so a scheduled caller keeps running.
    """
    loop = asyncio.new_event_loop()
    try:
        broker = get_broker()
        loop.run_until_complete(broker.authenticate())
        accounts = loop.run_until_complete(broker.get_accounts())
        account_id = accounts[0].account_id if accounts else ""
        positions = loop.run_until_complete(broker.get_positions(account_id))
    except Exception as exc:
        logger.warning("[position_sync] broker fetch failed: %s", exc)
        return 0
    finally:
        loop.close()

    if not positions:
        logger.debug("[position_sync] broker reported no open positions")
        return 0

    snapshotted_at = datetime.now(tz=timezone.utc)
    broker_name = getattr(broker, "name", "") or ""
    is_paper = "paper" in broker_name and not any(
        b in broker_name for b in ("schwab", "webull", "zerodha")
    )

    rows = [
        PositionSnapshot(
            broker=pos.broker or broker_name,
            symbol=pos.symbol.upper(),
            quantity=pos.quantity,
            average_cost=pos.average_cost,
            current_price=pos.current_price,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
            is_paper=is_paper,
            snapshotted_at=snapshotted_at,
        )
        for pos in positions
    ]
    try:
        with SessionLocal() as db:
            db.add_all(rows)
            db.commit()
    except Exception as exc:
        logger.warning("[position_sync] snapshot commit failed: %s", exc)
        return 0

    logger.info("[position_sync] snapshotted %d positions from %s", len(rows), broker_name)
    return len(rows)
