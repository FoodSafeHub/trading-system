from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Lifecycle modes for a bot-managed (no-resting-order) position.
MODE_MONITORING = "monitoring"      # held, no SELL signal yet — drawdown alerts only
MODE_PENDING_ARM = "pending_arm"    # SELL signal fired, price below arm gate
MODE_TRAIL_ARMED = "trail_armed"    # software trail live; target_stop is the exit level
MODE_RED_HOLD = "red_hold"          # SELL fired while red — held for manual review / green flip
MODE_EXITED = "exited"              # trail hit → market sell confirmed


class ManagedExitState(Base):
    """Durable per-symbol state for bot-managed exits (no resting broker stops).

    When schwab_managed_exits_enabled is on, US swing positions keep NO
    protective order at the broker — a resting stop is exactly what a
    fear-selloff sweep triggers. Instead the 60s fast-trail loop evaluates
    this row: the software trail level lives in target_stop, the red-hold
    gate in mode, and drawdown-alert dedup in last_alert_level. The row must
    survive process restarts — losing it would silently drop the trail level
    and alert dedup — hence a table, not memory.

    One row per symbol (the swing path holds one net position per symbol;
    per-lot granularity stays in the FIFO ledger).
    """

    __tablename__ = "managed_exit_state"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    # Broker route the position lives at ("default" = Schwab). Scope guard so a
    # future non-Schwab managed mode can't collide.
    broker: Mapped[str] = mapped_column(String(16), nullable=False, default="default")
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default=MODE_MONITORING)

    # Anchor SELL signal (mirrors trail_peaks keying) + trail parameters.
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signal_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trail_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    floor: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The software stop level — what used to be the resting STOP's trigger.
    target_stop: Mapped[float | None] = mapped_column(Float, nullable=True)

    # FIFO avg cost at last eval — the red/green line and drawdown-alert base.
    avg_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Red-hold episode tracking. notified_at doubles as the once-per-episode
    # notification dedup; both cleared when the position turns green.
    red_hold_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    red_hold_notified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Deepest drawdown level (e.g. -12.0) already alerted; re-armed with
    # hysteresis by the engine so threshold oscillation can't spam.
    last_alert_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_alert_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Per-symbol heartbeat — the invariant watchdog treats a stale value during
    # market hours as "position not under monitoring".
    last_eval_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
    )
