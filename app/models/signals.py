from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY | SELL | HOLD
    strength: Mapped[float] = mapped_column(Float, default=1.0)  # 0.0–1.0
    price_at_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicators_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON snapshot
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    acted_on: Mapped[bool] = mapped_column(default=False)
    order_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
