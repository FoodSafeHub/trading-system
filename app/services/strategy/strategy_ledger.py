"""Per-strategy share ledger.

The broker reports one aggregate position per symbol, but a symbol may be held by
several strategy assignments at once. This module maintains StrategyPosition rows so
each strategy's own held quantity is known — letting a SELL fired by one strategy
touch only that strategy's lot. See app.models.strategy_positions for the table and
the approved plan for the rationale.

Updates come from two places:
  * the scheduler, right after it places a BUY / tight-trail SELL (apply_fill), and
  * the order_sync reconciliation job, as a backstop for fills the scheduler didn't
    drive directly (native-trail exits, manual closes ingested from the broker).
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.strategy_positions import StrategyPosition

logger = logging.getLogger(__name__)


def get_held(
    symbol: str,
    system: str,
    strategy_name: str,
    broker: str = "default",
    db: Session | None = None,
) -> float:
    """Return this strategy's ledger quantity for the symbol (0.0 if none)."""
    own_session = db is None
    db = db or SessionLocal()
    try:
        row = (
            db.query(StrategyPosition)
            .filter_by(
                symbol=symbol.upper(),
                system=system,
                strategy_name=strategy_name,
                broker=broker or "default",
            )
            .first()
        )
        return float(row.held_qty) if row else 0.0
    finally:
        if own_session:
            db.close()


def apply_fill(
    symbol: str,
    system: str,
    strategy_name: str,
    side: str,
    qty: float,
    price: float | None = None,
    broker: str = "default",
    db: Session | None = None,
) -> float:
    """Apply a filled order to the strategy's ledger and return the new held qty.

    BUY adds, SELL subtracts (clamped at 0 — a strategy can never go negative in the
    ledger, even if a broker-side oddity reports more sold than we tracked). avg_price
    is volume-weighted on BUYs; left unchanged on SELLs.
    """
    symbol = symbol.upper()
    broker = broker or "default"
    side = (side or "").upper()
    qty = abs(float(qty or 0.0))
    if qty <= 0:
        return get_held(symbol, system, strategy_name, broker, db=db)

    own_session = db is None
    db = db or SessionLocal()
    try:
        row = (
            db.query(StrategyPosition)
            .filter_by(
                symbol=symbol,
                system=system,
                strategy_name=strategy_name,
                broker=broker,
            )
            .first()
        )
        if row is None:
            row = StrategyPosition(
                symbol=symbol,
                system=system,
                strategy_name=strategy_name,
                broker=broker,
                held_qty=0.0,
                avg_price=None,
            )
            db.add(row)

        if side == "BUY":
            prev_qty = float(row.held_qty or 0.0)
            new_qty = prev_qty + qty
            if price and price > 0:
                prev_cost = prev_qty * float(row.avg_price or price)
                row.avg_price = (prev_cost + qty * price) / new_qty if new_qty > 0 else price
            row.held_qty = new_qty
        elif side == "SELL":
            row.held_qty = max(0.0, float(row.held_qty or 0.0) - qty)
            if row.held_qty == 0.0:
                row.avg_price = None
        else:
            logger.warning("[strategy_ledger] ignoring fill with side=%r", side)

        if own_session:
            db.commit()
        new_held = float(row.held_qty)
        return new_held
    finally:
        if own_session:
            db.close()
