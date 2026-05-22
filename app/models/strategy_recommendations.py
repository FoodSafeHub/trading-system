from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class StrategyRecommendation(Base):
    """The historically best strategy for a symbol, derived from Compare All.

    One row per symbol — recomputing for a symbol overwrites the previous row.
    The metrics on this row mirror what the green "Recommended strategy" banner
    on the Backtest page shows, so the scanner's badge tooltip can render the
    same numbers without re-running the backtest.
    """

    __tablename__ = "strategy_recommendations"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    period: Mapped[str] = mapped_column(String(8), nullable=False)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    win_rate_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
