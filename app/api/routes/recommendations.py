from __future__ import annotations

"""Cached "historically best strategy per symbol" — derived from Compare All.

The scanner shows what's firing NOW. The Backtest page's Compare All shows
what historically WORKS on a name. These are different questions: today's
firing strategy may have a terrible 5-year track record, and vice versa.

This module persists the winner from a Compare All run so the scanner can
star rows where today's firing strategy matches the historical winner —
without re-running the backtest every time the scanner is opened.

Recomputation is a manual or background trigger (POST /recommendations/recompute),
not on-read, because a Compare All takes 30–90s per symbol.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.api.routes.backtest import backtest_custom_compare_all
from app.db import get_db
from app.models.strategy_recommendations import StrategyRecommendation
from app.schemas._serializers import serialize_et
from app.services.recommendations.winner import pick_winner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationOut(BaseModel):
    symbol: str
    strategy_name: str
    period: str
    total_trades: int | None
    win_rate_pct: float | None
    profit_factor: float | None
    total_return_pct: float | None
    sharpe_ratio: float | None
    computed_at: datetime

    @field_serializer("computed_at")
    def _ser_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class RecomputeIn(BaseModel):
    symbols: list[str]
    period: str = "5y"
    initial_capital: float = 100_000.0


def _upsert(db: Session, sym: str, period: str, winner: dict) -> StrategyRecommendation:
    row = db.query(StrategyRecommendation).filter_by(symbol=sym).first()
    if row is None:
        row = StrategyRecommendation(symbol=sym)
        db.add(row)
    row.strategy_name = winner["strategy_name"]
    row.period = period
    row.total_trades = winner.get("total_trades")
    row.win_rate_pct = winner.get("win_rate_pct")
    pf = winner.get("profit_factor")
    # Compare All emits inf for "all wins, no losses" — clamp to None for JSON.
    row.profit_factor = None if pf in (None, float("inf"), float("-inf")) else float(pf)
    row.total_return_pct = winner.get("total_return_pct")
    row.sharpe_ratio = winner.get("sharpe_ratio")
    row.computed_at = datetime.utcnow()
    return row


@router.post("/recompute/{symbol}", response_model=RecommendationOut | None)
def recompute_one(
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    db: Session = Depends(get_db),
):
    """Run Compare All on one symbol and persist the winner.

    Returns the new recommendation row, or null if no strategy met the
    minimum-trades guardrail. ~30–90s per call depending on history depth.
    """
    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(400, "symbol is required")
    try:
        rows = backtest_custom_compare_all(sym, period=period, initial_capital=initial_capital)
    except Exception as exc:
        logger.exception("[recommendations] Compare All failed for %s", sym)
        raise HTTPException(500, f"Compare All failed for {sym}: {exc}")

    winner = pick_winner(rows or [])
    if winner is None:
        return None

    row = _upsert(db, sym, period, winner)
    db.commit()
    db.refresh(row)
    return RecommendationOut.model_validate(row, from_attributes=True)


@router.post("/recompute", response_model=List[RecommendationOut])
def recompute_many(
    payload: RecomputeIn,
    db: Session = Depends(get_db),
):
    """Bulk recompute — runs Compare All for each symbol sequentially.

    Symbols that produce no rankable winner are skipped (not error'd) so a
    bad symbol in the middle doesn't kill the whole batch.
    """
    out: list[StrategyRecommendation] = []
    for raw in payload.symbols:
        sym = (raw or "").upper().strip()
        if not sym:
            continue
        try:
            rows = backtest_custom_compare_all(
                sym, period=payload.period, initial_capital=payload.initial_capital,
            )
        except Exception as exc:
            logger.warning("[recommendations] Compare All failed for %s: %s", sym, exc)
            continue
        winner = pick_winner(rows or [])
        if winner is None:
            logger.info("[recommendations] No winner picked for %s (insufficient trades)", sym)
            continue
        out.append(_upsert(db, sym, payload.period, winner))
    db.commit()
    for row in out:
        db.refresh(row)
    return [RecommendationOut.model_validate(r, from_attributes=True) for r in out]


@router.get("", response_model=List[RecommendationOut])
def list_recommendations(db: Session = Depends(get_db)):
    """Every cached recommendation — keyed by symbol."""
    rows = db.query(StrategyRecommendation).order_by(StrategyRecommendation.symbol.asc()).all()
    return [RecommendationOut.model_validate(r, from_attributes=True) for r in rows]


@router.get("/{symbol}", response_model=RecommendationOut | None)
def get_one(symbol: str, db: Session = Depends(get_db)):
    """Cached winner for one symbol, or null if we've never computed it."""
    sym = symbol.upper().strip()
    row = db.query(StrategyRecommendation).filter_by(symbol=sym).first()
    if row is None:
        return None
    return RecommendationOut.model_validate(row, from_attributes=True)
