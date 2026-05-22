from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class RealizedTrade(Base):
    """One FIFO-matched round-trip. Persisted so /pnl/* doesn't recompute on every call.

    A single SELL can produce multiple rows (one per BUY lot it consumed), so the
    natural key is (sell_order_id, buy_order_id) — that pair is unique across the
    log because a given BUY lot can only be drained once per SELL.
    """

    __tablename__ = "realized_trades"
    __table_args__ = (
        UniqueConstraint("sell_order_id", "buy_order_id", name="uq_realized_buy_sell"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    buy_price: Mapped[float] = mapped_column(Float, nullable=False)
    sell_price: Mapped[float] = mapped_column(Float, nullable=False)
    buy_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sell_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pct: Mapped[float] = mapped_column(Float, nullable=False)
    hold_days: Mapped[float] = mapped_column(Float, nullable=False)
    broker: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    buy_order_id: Mapped[int] = mapped_column(Integer, nullable=False)
    sell_order_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    buy_signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sell_signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buy_strategy: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    sell_strategy: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
