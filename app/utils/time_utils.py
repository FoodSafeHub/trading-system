from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


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
    return datetime.now(tz=ZoneInfo("UTC"))
