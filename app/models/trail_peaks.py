from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrailPeak(Base):
    """Durable high-water mark for an armed Approach C tight trail.

    A trailing stop must ratchet off the PEAK price seen since the assigned
    strategy's SELL signal — never the current price — so on a pullback the stop
    holds where it was instead of dropping with price. Recomputing that peak from
    OHLCV every cycle is fragile (provider gaps, stale bars, intraday-vs-daily
    granularity). This table persists it: one row per (symbol, signal_id), bumped
    each cycle to max(stored_peak, current high). It is the authoritative source
    the trail measures against; OHLCV/resting-stop are only seeds for the first
    write.

    Keyed by signal_id (the anchor SELL signal) so a NEW exit signal after a
    re-entry starts a fresh peak rather than inheriting the prior position's.
    """

    __tablename__ = "trail_peaks"
    __table_args__ = (UniqueConstraint("symbol", "signal_id", name="uq_trail_peak_symbol_signal"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # The anchor SELL Signal.id this peak is tracked against (None if unknown).
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Price the trail signal fired at — for context / audit.
    signal_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The high-water mark itself (the value the trail ratchets off).
    peak_price: Mapped[float] = mapped_column(Float, nullable=False)
    # When the peak was last advanced (so a stale peak is visible).
    peak_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
    )
