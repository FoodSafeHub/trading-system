"""Twelve Data circuit breaker.

Once TD reports daily quota exhaustion ("out of API credits"), every subsequent
TD call wastes ~0.5–1s on a doomed HTTP round-trip. During a full-universe
scan that adds up to minutes of wasted time per scan.

This module tracks the exhausted state in-process. Callers should:
  1. Check ``is_tripped()`` before sending a TD request — short-circuit if True.
  2. Call ``note_response_text(text)`` with any TD response body that may
     contain the credit-exhausted marker.

The breaker auto-resets at the next UTC midnight (TD's quota rolls over daily).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_tripped_until_utc_date: str | None = None

_EXHAUSTED_MARKER = "out of api credits"


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_tripped() -> bool:
    """True if TD is currently considered exhausted for today (UTC)."""
    global _tripped_until_utc_date
    with _lock:
        if _tripped_until_utc_date is None:
            return False
        if _tripped_until_utc_date != _utc_date_str():
            _tripped_until_utc_date = None
            _log.info("[td_breaker] reset for new UTC day")
            return False
        return True


def trip() -> None:
    """Mark TD as exhausted for the rest of the current UTC day."""
    global _tripped_until_utc_date
    with _lock:
        today = _utc_date_str()
        if _tripped_until_utc_date != today:
            _tripped_until_utc_date = today
            _log.warning("[td_breaker] tripped — skipping TD until next UTC day")


def note_response_text(text: str | None) -> None:
    """Inspect a TD response body and trip the breaker if it signals quota exhaustion."""
    if not text:
        return
    if _EXHAUSTED_MARKER in text.lower():
        trip()


def status() -> dict:
    """For debugging / health endpoints."""
    with _lock:
        return {
            "tripped": _tripped_until_utc_date is not None
                       and _tripped_until_utc_date == _utc_date_str(),
            "tripped_for_utc_date": _tripped_until_utc_date,
            "current_utc_date": _utc_date_str(),
        }
