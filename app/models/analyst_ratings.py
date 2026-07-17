from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AnalystRating(Base):
    """Cached Wall-Street analyst view of one symbol (yfinance-sourced).

    One row per symbol, overwritten on each refresh — the dashboard reads the
    cache, never yfinance directly (analyst data changes slowly and yfinance
    is rate-limited/flaky). Symbols with no coverage (most NSE names) still
    get a row with a `note`, so the page shows "no coverage" instead of a
    silent hole.
    """

    __tablename__ = "analyst_ratings"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    # Why this symbol is in the universe (holdings ∪ enabled assignments).
    is_holding: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_assigned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    # yfinance info["recommendationKey"], e.g. "buy", "hold", "strong_buy".
    recommendation_key: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Current-month analyst counts from recommendations_summary (period "0m").
    strong_buy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sell: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strong_sell: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analyst_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    target_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    # (target_mean / current_price − 1) × 100 at refresh time.
    upside_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # JSON list of the most recent upgrades/downgrades:
    # [{"firm", "action", "from_grade", "to_grade", "date"}]
    upgrades_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # e.g. "no analyst coverage" for thin NSE names.
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
