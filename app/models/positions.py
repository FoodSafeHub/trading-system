from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    average_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_paper: Mapped[bool] = mapped_column(default=True)
    snapshotted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
