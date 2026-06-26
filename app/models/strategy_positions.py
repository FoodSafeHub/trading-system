from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class StrategyPosition(Base):
    """
    Per-strategy share ledger.

    The broker reports only ONE aggregate position per symbol, but a symbol can be
    held concurrently by several strategy assignments (see
    app.models.assignments.SymbolStrategyAssignment). This table is our source of
    truth for "how many shares does strategy X own" so that a SELL fired by one
    strategy only touches that strategy's lot — never another strategy's shares.

    held_qty is maintained from filled orders, attributed back to the assignment via
    the Signal.strategy_name -> Order.signal_id chain (see strategy_ledger.apply_fill
    and the order_sync reconciliation backstop).
    """
    __tablename__ = "strategy_positions"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    system: Mapped[str] = mapped_column(String(32), primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    # The broker the lot lives on. Mirrors the assignment's resolved route so a
    # symbol traded on two brokers by the same strategy stays separated.
    broker: Mapped[str] = mapped_column(String(32), primary_key=True, default="default")

    held_qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
