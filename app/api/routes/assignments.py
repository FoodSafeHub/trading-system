from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.assignments import SymbolStrategyAssignment
from app.schemas._serializers import serialize_et

router = APIRouter(prefix="/assignments", tags=["assignments"])

VALID_SYSTEMS = {"bollinger", "perplexity"}


class AssignmentIn(BaseModel):
    symbol: str
    system: str        # "bollinger" or "perplexity"
    strategy_name: str
    enabled: bool = True
    max_capital_usd: float | None = None   # None = use global account settings
    notes: str = ""


class AssignmentOut(BaseModel):
    symbol: str
    system: str
    strategy_name: str
    enabled: bool
    max_capital_usd: float | None
    notes: str | None
    assigned_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("assigned_at")
    def _ser_assigned_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


@router.get("", response_model=List[AssignmentOut])
def list_assignments(db: Session = Depends(get_db)):
    return db.query(SymbolStrategyAssignment).order_by(SymbolStrategyAssignment.symbol).all()


@router.post("", response_model=AssignmentOut)
def upsert_assignment(body: AssignmentIn, db: Session = Depends(get_db)):
    """Create or update the strategy assignment for a symbol."""
    if body.system not in VALID_SYSTEMS:
        raise HTTPException(400, f"system must be one of {sorted(VALID_SYSTEMS)}")
    symbol = body.symbol.upper().strip()
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol).first()
    if row:
        row.system = body.system
        row.strategy_name = body.strategy_name
        row.enabled = body.enabled
        row.max_capital_usd = body.max_capital_usd
        row.notes = body.notes
        row.assigned_at = datetime.now(tz=timezone.utc)
    else:
        row = SymbolStrategyAssignment(
            symbol=symbol,
            system=body.system,
            strategy_name=body.strategy_name,
            enabled=body.enabled,
            max_capital_usd=body.max_capital_usd,
            notes=body.notes,
            assigned_at=datetime.now(tz=timezone.utc),
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.patch("/{symbol}/toggle")
def toggle_assignment(symbol: str, enabled: bool, db: Session = Depends(get_db)):
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    row.enabled = enabled
    db.commit()
    return {"symbol": row.symbol, "enabled": row.enabled}


@router.patch("/{symbol}/cap")
def set_cap(symbol: str, max_capital_usd: float | None, db: Session = Depends(get_db)):
    """Update only the capital cap on an existing assignment."""
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    row.max_capital_usd = max_capital_usd if max_capital_usd and max_capital_usd > 0 else None
    db.commit()
    return {"symbol": row.symbol, "max_capital_usd": row.max_capital_usd}


@router.delete("/{symbol}")
def delete_assignment(symbol: str, db: Session = Depends(get_db)):
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    db.delete(row)
    db.commit()
    return {"deleted": symbol.upper()}
