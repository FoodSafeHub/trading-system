from __future__ import annotations

"""Wall-Street analyst ratings for holdings + assigned symbols.

Read endpoints serve the analyst_ratings cache (never yfinance directly —
it's rate-limited and slow); POST /ratings/recompute refreshes the cache on
demand. A scheduler job refreshes it every analyst_ratings_refresh_hours.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.analyst_ratings import AnalystRating
from app.schemas._serializers import serialize_et

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ratings", tags=["ratings"])


class RatingOut(BaseModel):
    symbol: str
    is_holding: bool
    is_assigned: bool
    current_price: Optional[float]
    recommendation_key: Optional[str]
    strong_buy: Optional[int]
    buy: Optional[int]
    hold: Optional[int]
    sell: Optional[int]
    strong_sell: Optional[int]
    analyst_count: Optional[int]
    target_mean: Optional[float]
    target_high: Optional[float]
    target_low: Optional[float]
    target_median: Optional[float]
    upside_pct: Optional[float]
    upgrades: list[dict] = []
    note: Optional[str]
    computed_at: datetime

    @field_serializer("computed_at")
    def _ser_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class RecomputeIn(BaseModel):
    # Empty/omitted → refresh the full universe (holdings ∪ assignments).
    symbols: list[str] = []


def _to_out(row: AnalystRating) -> RatingOut:
    upgrades: list[dict] = []
    if row.upgrades_json:
        try:
            upgrades = json.loads(row.upgrades_json)
        except Exception:
            upgrades = []
    return RatingOut(
        symbol=row.symbol,
        is_holding=bool(row.is_holding),
        is_assigned=bool(row.is_assigned),
        current_price=row.current_price,
        recommendation_key=row.recommendation_key,
        strong_buy=row.strong_buy,
        buy=row.buy,
        hold=row.hold,
        sell=row.sell,
        strong_sell=row.strong_sell,
        analyst_count=row.analyst_count,
        target_mean=row.target_mean,
        target_high=row.target_high,
        target_low=row.target_low,
        target_median=row.target_median,
        upside_pct=row.upside_pct,
        upgrades=upgrades,
        note=row.note,
        computed_at=row.computed_at,
    )


@router.get("", response_model=List[RatingOut])
def list_ratings(db: Session = Depends(get_db)):
    """Every cached rating — holdings first, then assigned-only, A-Z."""
    rows = db.query(AnalystRating).all()
    rows.sort(key=lambda r: (not r.is_holding, r.symbol))
    return [_to_out(r) for r in rows]


@router.get("/{symbol}", response_model=RatingOut | None)
def get_rating(symbol: str, db: Session = Depends(get_db)):
    row = db.query(AnalystRating).filter_by(symbol=symbol.upper().strip()).first()
    return _to_out(row) if row else None


@router.post("/recompute")
def recompute(payload: RecomputeIn | None = None, db: Session = Depends(get_db)):
    """Refresh the cache from yfinance (full universe when no symbols given).

    Sync handler on purpose: the universe build creates its own event loop
    for the broker fan-out (FastAPI runs sync routes in a threadpool).
    """
    from app.services.research.analyst_ratings import refresh_all

    symbols = payload.symbols if payload else []
    return refresh_all(db, symbols or None)
