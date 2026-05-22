from __future__ import annotations

"""Realized + unrealized P/L endpoints.

All endpoints compute on the fly off the orders table — no persisted P/L
state, no caching. Cost is one orders query + one signals query per call;
fine for the ~10s of K orders this system is sized for. If that ever
becomes a hot path we'll add a persisted realized_trades table (Pass 2).
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas._serializers import serialize_et
from app.services.brokers.factory import get_broker
from app.services.pnl.aggregate import (
    by_strategy,
    by_symbol,
    equity_curve,
    open_position_pnl,
    summarize,
)
from app.services.pnl.fifo import compute_fifo
from app.services.pnl.store import load_closed_trades, sync_realized_trades

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl", tags=["pnl"])


# ── Response models ────────────────────────────────────────────────────────

class GroupSummaryOut(BaseModel):
    key: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    total_realized_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float]
    best_trade: float
    worst_trade: float
    avg_hold_days: float


class ClosedTradeOut(BaseModel):
    symbol: str
    quantity: float
    buy_price: float
    sell_price: float
    buy_at: datetime
    sell_at: datetime
    realized_pnl: float
    realized_pct: float
    hold_days: float
    broker: str
    buy_order_id: int
    sell_order_id: int
    buy_strategy: Optional[str]
    sell_strategy: Optional[str]
    is_paper: bool

    @field_serializer("buy_at", "sell_at")
    def _ser_ts(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class EquityPointOut(BaseModel):
    at: datetime
    realized_pnl: float
    peak_pnl: float
    drawdown: float       # peak_pnl - realized_pnl, always >= 0
    drawdown_pct: float | None  # drawdown / peak_pnl when peak > 0, else None

    @field_serializer("at")
    def _ser_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class OpenPositionOut(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    last_price: Optional[float]
    market_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pct: Optional[float]
    broker: str
    is_paper: bool


class SummaryOut(BaseModel):
    realized: GroupSummaryOut
    total_unrealized_pnl: float
    open_position_count: int
    closed_trade_count: int
    has_live_prices: bool


# ── Helpers ────────────────────────────────────────────────────────────────

def _refresh(db: Session):
    """Sync new round-trips into realized_trades, then return (closed, fifo).

    The FifoResult is reused for open lots (no separate query). The closed list
    is the canonical persisted history — same numbers across calls regardless of
    what the live walk would produce.
    """
    _inserted, fifo = sync_realized_trades(db)
    closed = load_closed_trades(db)
    return closed, fifo


def _resolve_last_prices(symbols: list[str]) -> dict[str, float]:
    """Pull last/ask/bid from the active broker for the given symbols.

    Returns {} on any failure — the unrealized columns will surface as None
    rather than 0, so the UI can tell "we don't know" from "zero".
    """
    if not symbols:
        return {}
    import asyncio
    try:
        broker = get_broker()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(broker.authenticate())
            quotes = loop.run_until_complete(broker.get_quotes(symbols))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[pnl] quote fetch failed: %s", exc)
        return {}

    out: dict[str, float] = {}
    for sym, q in (quotes or {}).items():
        price = getattr(q, "last", None) or getattr(q, "ask", None) or getattr(q, "bid", None)
        if price and price > 0:
            out[sym.upper()] = float(price)
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/summary", response_model=SummaryOut)
def pnl_summary(
    include_unrealized: bool = True,
    db: Session = Depends(get_db),
):
    """Top-line numbers: realized so far, unrealized on open positions, counts."""
    closed, fifo = _refresh(db)
    realized = summarize(closed, key="ALL")
    last_prices: dict[str, float] = {}
    if include_unrealized and fifo.open_lots:
        last_prices = _resolve_last_prices(sorted({l.symbol for l in fifo.open_lots}))
    open_rows, total_unrealized = open_position_pnl(fifo.open_lots, last_prices)
    return SummaryOut(
        realized=GroupSummaryOut(**asdict(realized)),
        total_unrealized_pnl=total_unrealized,
        open_position_count=len(open_rows),
        closed_trade_count=len(closed),
        has_live_prices=bool(last_prices),
    )


@router.get("/by-symbol", response_model=List[GroupSummaryOut])
def pnl_by_symbol(db: Session = Depends(get_db)):
    """Realized P/L grouped by symbol, best to worst."""
    closed, _ = _refresh(db)
    return [GroupSummaryOut(**asdict(g)) for g in by_symbol(closed)]


@router.get("/by-strategy", response_model=List[GroupSummaryOut])
def pnl_by_strategy(db: Session = Depends(get_db)):
    """Realized P/L grouped by the BUY-side strategy attribution."""
    closed, _ = _refresh(db)
    return [GroupSummaryOut(**asdict(g)) for g in by_strategy(closed)]


@router.get("/closed-trades", response_model=List[ClosedTradeOut])
def pnl_closed_trades(
    symbol: str | None = None,
    strategy: str | None = None,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    """The realized round-trip log. Most recent SELL first."""
    if limit < 1 or limit > 5000:
        raise HTTPException(400, "limit must be between 1 and 5000")
    closed, _ = _refresh(db)
    rows = closed
    if symbol:
        sym = symbol.upper().strip()
        rows = [t for t in rows if t.symbol == sym]
    if strategy:
        rows = [t for t in rows if (t.buy_strategy or "(unattributed)") == strategy]
    rows = sorted(rows, key=lambda t: t.sell_at, reverse=True)[:limit]
    return [
        ClosedTradeOut(
            symbol=t.symbol, quantity=t.quantity, buy_price=t.buy_price,
            sell_price=t.sell_price, buy_at=t.buy_at, sell_at=t.sell_at,
            realized_pnl=t.realized_pnl, realized_pct=t.realized_pct,
            hold_days=t.hold_days, broker=t.broker,
            buy_order_id=t.buy_order_id, sell_order_id=t.sell_order_id,
            buy_strategy=t.buy_strategy, sell_strategy=t.sell_strategy,
            is_paper=t.is_paper,
        )
        for t in rows
    ]


@router.get("/equity-curve", response_model=List[EquityPointOut])
def pnl_equity_curve(bucket: str = "trade", db: Session = Depends(get_db)):
    """Cumulative realized P/L over time + running drawdown.

    `bucket=trade` (default) emits one point per closed trade.
    `bucket=day` collapses to one point per calendar day (last close of each
    day) — cleaner for long histories.
    """
    if bucket not in {"trade", "day"}:
        raise HTTPException(400, "bucket must be 'trade' or 'day'")
    closed, _ = _refresh(db)
    points = equity_curve(closed)

    if bucket == "day" and points:
        by_day: dict = {}
        for p in points:
            key = p.at.date()
            by_day[key] = p  # last point of each day wins (curve is monotonic in time)
        points = sorted(by_day.values(), key=lambda p: p.at)

    out: list[EquityPointOut] = []
    peak = 0.0
    for p in points:
        if p.realized_pnl > peak:
            peak = p.realized_pnl
        dd = max(peak - p.realized_pnl, 0.0)
        dd_pct = (dd / peak) if peak > 0 else None
        out.append(EquityPointOut(
            at=p.at,
            realized_pnl=p.realized_pnl,
            peak_pnl=peak,
            drawdown=dd,
            drawdown_pct=dd_pct,
        ))
    return out


@router.get("/open-positions", response_model=List[OpenPositionOut])
def pnl_open_positions(db: Session = Depends(get_db)):
    """Open long lots aggregated per symbol, with unrealized P/L vs last quote."""
    _, fifo = _refresh(db)
    last_prices: dict[str, float] = {}
    if fifo.open_lots:
        last_prices = _resolve_last_prices(sorted({l.symbol for l in fifo.open_lots}))
    rows, _total = open_position_pnl(fifo.open_lots, last_prices)
    return [OpenPositionOut(**r) for r in rows]
