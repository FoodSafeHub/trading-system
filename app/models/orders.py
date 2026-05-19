from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)      # BUY | SELL
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # MARKET | LIMIT | STOP
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending | previewed | submitted | filled | partial | cancelled | rejected | error
    is_paper: Mapped[bool] = mapped_column(default=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preview_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    # Origin of the order. Set explicitly by every code path that submits orders so
    # we can always tell a manual click from a scheduler/autotrader/scanner fire.
    # Values: "manual" | "scheduler" | "autotrader" | "scanner" | "unknown_pre_migration"
    source: Mapped[str] = mapped_column(String(32), default="manual", nullable=False)


class OrderPreview(Base):
    __tablename__ = "order_previews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, nullable=False)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    buying_power_effect: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
