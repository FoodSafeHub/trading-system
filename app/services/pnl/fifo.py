from __future__ import annotations

"""FIFO round-trip matcher for realized P/L.

Walks the orders table chronologically per symbol, pairs BUY lots with SELL
fills using first-in-first-out, and emits one ClosedTrade per closed lot
(or per closed slice of a lot when a SELL closes only part of one). Open
lots — anything still long at the end of the history — are returned as
OpenLot rows so the caller can compute unrealized P/L against a live quote.

Source data is the orders table filtered to status="filled" with non-null
fill_price and quantity > 0. We deliberately use the order-level fill_price
rather than aggregating executions: brokers nearly always fill MARKET orders
in a single execution, and orders.fill_price is the canonical value the rest
of the system already trusts. Partial fills (status="partial") are skipped
in Pass 1 — they're rare in this codebase and surface as open quantity that
the broker still owns.

SHORTS: this matcher only handles long round trips (BUY then SELL). A SELL
with no covering BUY history is logged + skipped so we don't manufacture
phantom realized P/L. Add covering logic in Pass 2 if shorting is ever
turned on.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sqlalchemy.orm import Session

from app.models.orders import Order
from app.models.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class ClosedTrade:
    """One realized round trip — a SELL that closed (some or all of) a BUY lot."""
    symbol: str
    quantity: float
    buy_price: float
    sell_price: float
    buy_at: datetime
    sell_at: datetime
    realized_pnl: float
    realized_pct: float          # (sell - buy) / buy * 100
    hold_days: float
    broker: str
    buy_order_id: int
    sell_order_id: int
    buy_signal_id: int | None
    sell_signal_id: int | None
    buy_strategy: str | None     # joined from signals.strategy_name
    sell_strategy: str | None
    is_paper: bool


@dataclass
class OpenLot:
    """A BUY lot (or remainder of one) that has not yet been closed by a SELL."""
    symbol: str
    quantity: float
    buy_price: float
    buy_at: datetime
    broker: str
    buy_order_id: int
    buy_signal_id: int | None
    buy_strategy: str | None
    is_paper: bool


@dataclass
class FifoResult:
    closed: list[ClosedTrade]
    open_lots: list[OpenLot]


def _filled_orders(db: Session) -> list[Order]:
    """Pull every order that produced a real fill, ordered chronologically.

    We sort by filled_at (the broker's confirmation timestamp) and fall back
    to created_at when filled_at is null. id is the final tiebreaker so the
    walk is deterministic across multiple fills in the same second.
    """
    rows = (
        db.query(Order)
        .filter(Order.status == "filled")
        .filter(Order.fill_price.is_not(None))
        .filter(Order.quantity > 0)
        .order_by(Order.filled_at.asc(), Order.created_at.asc(), Order.id.asc())
        .all()
    )
    return rows


def _strategy_index(db: Session, signal_ids: Iterable[int]) -> dict[int, str]:
    """Bulk-fetch strategy_name for the signal ids referenced by these orders."""
    ids = [s for s in signal_ids if s]
    if not ids:
        return {}
    rows = db.query(Signal.id, Signal.strategy_name).filter(Signal.id.in_(ids)).all()
    return {sid: name for sid, name in rows}


def compute_fifo(db: Session) -> FifoResult:
    """Replay every filled order through a per-symbol FIFO queue.

    Returns realized round trips and any still-open long lots.
    """
    orders = _filled_orders(db)
    strat = _strategy_index(db, (o.signal_id for o in orders))

    # symbol -> deque of {qty, price, at, order_id, signal_id, broker, is_paper}
    lots: dict[str, deque[dict]] = {}
    closed: list[ClosedTrade] = []

    for o in orders:
        sym = (o.symbol or "").upper()
        if not sym:
            continue
        qty = float(o.quantity or 0)
        px = float(o.fill_price or 0)
        if qty <= 0 or px <= 0:
            continue
        at = o.filled_at or o.created_at

        if o.side == "BUY":
            lots.setdefault(sym, deque()).append({
                "qty": qty,
                "price": px,
                "at": at,
                "order_id": o.id,
                "signal_id": o.signal_id,
                "broker": o.broker,
                "is_paper": bool(o.is_paper),
            })
            continue

        if o.side != "SELL":
            continue

        # SELL — drain BUY lots in FIFO order until this SELL is consumed.
        remaining = qty
        queue = lots.get(sym)
        if not queue:
            logger.warning(
                "[pnl] SELL %s qty=%.4f order_id=%d has no prior BUY history — "
                "skipping (would imply a short or external position)",
                sym, qty, o.id,
            )
            continue

        while remaining > 1e-9 and queue:
            lot = queue[0]
            take = min(lot["qty"], remaining)
            buy_price = lot["price"]
            sell_price = px
            pnl = (sell_price - buy_price) * take
            pct = ((sell_price - buy_price) / buy_price * 100.0) if buy_price > 0 else 0.0
            buy_at: datetime = lot["at"]
            sell_at: datetime = at or buy_at
            hold_days = max((sell_at - buy_at).total_seconds() / 86400.0, 0.0) \
                if (sell_at and buy_at) else 0.0

            closed.append(ClosedTrade(
                symbol=sym,
                quantity=take,
                buy_price=buy_price,
                sell_price=sell_price,
                buy_at=buy_at,
                sell_at=sell_at,
                realized_pnl=pnl,
                realized_pct=pct,
                hold_days=hold_days,
                broker=o.broker or lot["broker"] or "",
                buy_order_id=lot["order_id"],
                sell_order_id=o.id,
                buy_signal_id=lot["signal_id"],
                sell_signal_id=o.signal_id,
                buy_strategy=strat.get(lot["signal_id"]) if lot["signal_id"] else None,
                sell_strategy=strat.get(o.signal_id) if o.signal_id else None,
                is_paper=bool(o.is_paper) or lot["is_paper"],
            ))

            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                queue.popleft()

        if remaining > 1e-9:
            logger.warning(
                "[pnl] SELL %s order_id=%d unmatched %.4f shares — no BUY lots left",
                sym, o.id, remaining,
            )

    open_lots: list[OpenLot] = []
    for sym, queue in lots.items():
        for lot in queue:
            if lot["qty"] <= 1e-9:
                continue
            open_lots.append(OpenLot(
                symbol=sym,
                quantity=lot["qty"],
                buy_price=lot["price"],
                buy_at=lot["at"],
                broker=lot["broker"] or "",
                buy_order_id=lot["order_id"],
                buy_signal_id=lot["signal_id"],
                buy_strategy=strat.get(lot["signal_id"]) if lot["signal_id"] else None,
                is_paper=lot["is_paper"],
            ))

    return FifoResult(closed=closed, open_lots=open_lots)
