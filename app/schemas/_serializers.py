"""Shared pydantic serializers for API schemas.

All persisted timestamps in this codebase are stored in UTC (often as naive
datetimes from `datetime.utcnow`). At API boundaries we surface them in
America/New_York (ET) so that operators reading responses don't have to
mentally convert from UTC.
"""
from __future__ import annotations

from datetime import datetime

from app.utils.time_utils import to_et


def serialize_et(dt: datetime | None) -> str | None:
    """Render a stored datetime as an ET-localized ISO 8601 string.

    Naive datetimes are assumed UTC (the storage convention for this project).
    Returns ``None`` if the input is ``None`` so Optional fields stay null.
    """
    et = to_et(dt)
    return et.isoformat() if et else None
