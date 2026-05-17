from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ScanResult(Base):
    __tablename__ = "scan_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scan_run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    universe: Mapped[str] = mapped_column(String(32), nullable=False)  # watchlist|sp500|nasdaq100|custom
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)   # BUY | SELL
    score: Mapped[float] = mapped_column(Float, default=0.0)            # 0–100 composite
    strategies_agreeing: Mapped[int] = mapped_column(Integer, default=1)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    indicators_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_traded: Mapped[bool] = mapped_column(default=False)
