from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.notifications import Notification
from app.schemas._serializers import serialize_et

router = APIRouter(prefix="/notifications", tags=["notifications"])


class NotificationOut(BaseModel):
    id: int
    kind: str
    symbol: Optional[str]
    direction: Optional[str]
    strategy: Optional[str]
    source: Optional[str]
    price: Optional[float]
    title: str
    body: Optional[str]
    created_at: datetime
    read_at: Optional[datetime]

    model_config = {"from_attributes": True}

    @field_serializer("created_at", "read_at")
    def _ser_dt(self, dt: Optional[datetime]) -> Optional[str]:
        return serialize_et(dt) if dt else None


@router.get("", response_model=List[NotificationOut])
def list_notifications(
    limit: int = 50,
    unread_only: bool = False,
    db: Session = Depends(get_db),
):
    q = db.query(Notification).order_by(Notification.created_at.desc())
    if unread_only:
        q = q.filter(Notification.read_at.is_(None))
    return q.limit(limit).all()


@router.get("/unread-count")
def unread_count(db: Session = Depends(get_db)):
    n = db.query(Notification).filter(Notification.read_at.is_(None)).count()
    return {"unread": n}


@router.post("/{notification_id}/read")
def mark_read(notification_id: int, db: Session = Depends(get_db)):
    row = db.query(Notification).filter_by(id=notification_id).first()
    if not row:
        raise HTTPException(404, "notification not found")
    if row.read_at is None:
        row.read_at = datetime.now(tz=timezone.utc)
        db.commit()
    return {"id": row.id, "read_at": row.read_at}


@router.post("/mark-all-read")
def mark_all_read(db: Session = Depends(get_db)):
    now = datetime.now(tz=timezone.utc)
    updated = (
        db.query(Notification)
        .filter(Notification.read_at.is_(None))
        .update({Notification.read_at: now}, synchronize_session=False)
    )
    db.commit()
    return {"marked_read": updated}


@router.delete("/{notification_id}")
def delete_notification(notification_id: int, db: Session = Depends(get_db)):
    row = db.query(Notification).filter_by(id=notification_id).first()
    if not row:
        raise HTTPException(404, "notification not found")
    db.delete(row)
    db.commit()
    return {"deleted": notification_id}
