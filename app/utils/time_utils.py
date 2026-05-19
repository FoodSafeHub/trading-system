from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def now_in_tz(tz: ZoneInfo) -> datetime:
    return datetime.now(tz=tz)


def is_market_hours(start: str, end: str, tz: ZoneInfo) -> bool:
    """Return True if current time is within [start, end) in the given timezone."""
    now = now_in_tz(tz)
    start_t = time.fromisoformat(start)
    end_t = time.fromisoformat(end)
    current_t = now.time().replace(second=0, microsecond=0)
    # Also exclude weekends
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return start_t <= current_t < end_t


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def now_et() -> datetime:
    return datetime.now(tz=ET)


def to_et(dt: datetime | None) -> datetime | None:
    """Convert any datetime to ET. Naive datetimes are assumed to be UTC
    (matches our storage convention from `datetime.utcnow`)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def format_et(dt: datetime | None, fmt: str = "%Y-%m-%dT%H:%M:%S%z") -> str | None:
    """Render a datetime as an ET string. Returns None if dt is None."""
    et = to_et(dt)
    return et.strftime(fmt) if et else None
