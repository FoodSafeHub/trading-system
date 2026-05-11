from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.signals import Signal
from app.models.strategy_runs import StrategyRun
from app.schemas.signals import SignalOut, StrategyRunOut

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_model=List[SignalOut])
def list_signals(limit: int = 50, symbol: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Signal).order_by(Signal.created_at.desc())
    if symbol:
        q = q.filter(Signal.symbol == symbol.upper())
    return q.limit(limit).all()


@router.get("/runs", response_model=List[StrategyRunOut])
def list_runs(limit: int = 20, db: Session = Depends(get_db)):
    return db.query(StrategyRun).order_by(StrategyRun.started_at.desc()).limit(limit).all()
