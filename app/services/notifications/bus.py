"""Single chokepoint for emitting user-facing notifications.

Called from the scheduler + scanner signal paths when a BUY/SELL fires on a
symbol that has an active assignment. Writes to the notifications table (the
dashboard's notification log) and best-effort fires a Windows toast.

Design choice: assignment gating happens INSIDE the bus, so callers don't have
to repeat the lookup. They just call notify_signal(symbol, ...) and the bus
decides whether to emit anything.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.assignments import SymbolStrategyAssignment
from app.models.notifications import Notification

logger = logging.getLogger(__name__)


def _is_assigned(db: Session, symbol: str) -> bool:
    sym = (symbol or "").upper().strip()
    if not sym:
        return False
    row = (
        db.query(SymbolStrategyAssignment)
        .filter_by(symbol=sym, enabled=True)
        .first()
    )
    return row is not None


def _toast(title: str, body: str) -> None:
    """Best-effort Windows toast. Silent on non-Windows or if winotify missing."""
    try:
        from winotify import Notification as _WinNotif, audio  # type: ignore
    except Exception:
        return
    try:
        n = _WinNotif(
            app_id="Trading System",
            title=title,
            msg=body,
            duration="short",
        )
        n.set_audio(audio.Default, loop=False)
        n.show()
    except Exception as exc:
        logger.debug("toast failed: %s", exc)


def notify_signal(
    *,
    symbol: str,
    direction: str,
    strategy: str,
    source: str,
    price: Optional[float] = None,
    extra: Optional[str] = None,
    gated: bool = True,
) -> Optional[int]:
    """Emit a signal notification IF the symbol is in an enabled assignment.

    gated=False bypasses the assignment check — used by paths that EXECUTED a
    real order (consensus can trade unassigned symbols; the day-trading
    autotrader's symbols aren't assignments). A filled order must always
    notify; only discovery-style signals stay assignment-gated.

    Returns the new notification id, or None if gated out.
    """
    symbol = (symbol or "").upper().strip()
    direction = (direction or "").upper().strip()
    if direction not in ("BUY", "SELL"):
        return None

    try:
        with SessionLocal() as db:
            if gated and not _is_assigned(db, symbol):
                return None
            title = f"{direction} signal: {symbol}"
            body_parts = [f"Strategy: {strategy}", f"Source: {source}"]
            if price:
                body_parts.append(f"Price: ${price:,.2f}")
            if extra:
                body_parts.append(extra)
            body = " · ".join(body_parts)

            row = Notification(
                kind="signal",
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                source=source,
                price=price,
                title=title,
                body=body,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            new_id = row.id
    except Exception as exc:
        logger.warning("notify_signal db write failed: %s", exc)
        return None

    _toast(title, body)
    return new_id


def notify_suppression(
    *,
    symbol: str,
    reason: str,
    detail: str,
    source: str = "scheduler",
    direction: Optional[str] = None,
    toast: bool = False,
) -> Optional[int]:
    """Emit a notification when the automation correctly-but-silently declined
    to act — the events that bite because nothing happened: a cash-limited or
    regime-capped BUY skip, or a trailing-stop that failed to arm.

    NOT gated on assignments (a suppression is by definition about an assigned
    symbol) and toast defaults OFF (these are informational, not alarms) —
    callers pass toast=True for the ones that need attention (e.g. an unarmed
    protective stop). Best-effort: never raises into the caller.

    reason  : short machine-ish tag, e.g. "cash_limited", "regime_cap",
              "trail_unarmed", "insufficient_cash".
    detail  : human-readable one-liner for the notification body.
    """
    symbol = (symbol or "").upper().strip()
    title = f"Skipped {direction or ''} {symbol}: {reason}".strip()
    try:
        with SessionLocal() as db:
            row = Notification(
                kind="suppress",
                symbol=symbol or None,
                direction=(direction or "").upper() or None,
                strategy=None,
                source=source,
                price=None,
                title=title[:256],
                body=detail,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            new_id = row.id
    except Exception as exc:
        logger.warning("notify_suppression db write failed: %s", exc)
        return None

    if toast:
        _toast(title, detail)
    return new_id


def notify_m1(
    *,
    title: str,
    body: str,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    price: Optional[float] = None,
    toast: bool = True,
) -> Optional[int]:
    """Emit an M1-portfolio notification — NOT gated on assignments.

    The M1 advisor's symbols aren't broker assignments, so notify_signal would
    suppress them. This path always writes (source='m1') and optionally toasts.
    """
    try:
        with SessionLocal() as db:
            row = Notification(
                kind="signal",
                symbol=(symbol or "").upper() or None,
                direction=(direction or "").upper() or None,
                strategy="m1_advisor",
                source="m1",
                price=price,
                title=title[:256],
                body=body,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            new_id = row.id
    except Exception as exc:
        logger.warning("notify_m1 db write failed: %s", exc)
        return None

    if toast:
        _toast(title, body)
    return new_id
