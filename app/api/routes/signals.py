from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.signals import Signal
from app.models.strategy_runs import StrategyRun
from app.schemas.signals import SignalOut, StrategyRunOut

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_model=List[SignalOut])
def list_signals(
    # Default raised from 50 to 500: a single 15-min scheduler cycle emits
    # one signal per (assigned-symbol, strategy) pair, which for a typical
    # configuration is already > 50. With the prior 50-row cap, symbols
    # scanned earliest in the cycle were silently evicted before the
    # dashboard could render them, making it look like only a handful of
    # symbols were being scanned. Hard ceiling of 5000 prevents accidental
    # full-table scans.
    limit: int = Query(500, ge=1, le=5000),
    symbol: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Signal).order_by(Signal.created_at.desc())
    if symbol:
        q = q.filter(Signal.symbol == symbol.upper())
    return q.limit(limit).all()


@router.get("/runs", response_model=List[StrategyRunOut])
def list_runs(limit: int = 20, db: Session = Depends(get_db)):
    return db.query(StrategyRun).order_by(StrategyRun.started_at.desc()).limit(limit).all()
