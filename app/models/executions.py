from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Execution(Base):
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, nullable=False)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_execution_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # fill | partial_fill | cancel | reject | error | status_update
    quantity_filled: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
