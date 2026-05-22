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

VALID_SYSTEMS = {"bollinger", "perplexity", "scanner"}
# "default" means follow the global active_broker / trade_routing toggle.
# Any other value routes this symbol's orders to a specific broker adapter.
VALID_BROKERS = {"default", "paper", "schwab", "webull"}


class AssignmentIn(BaseModel):
    symbol: str
    system: str        # "bollinger" | "perplexity" | "scanner"
    strategy_name: str
    enabled: bool = True
    max_capital_usd: float | None = None   # None = use global account settings
    max_shares: float | None = None        # Fallback shares cap when no dollar cap
    broker: str = "default"                # "default" | "paper" | "schwab" | "webull"
    notes: str = ""


class AssignmentOut(BaseModel):
    symbol: str
    system: str
    strategy_name: str
    enabled: bool
    max_capital_usd: float | None
    max_shares: float | None
    broker: str
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
    if body.broker not in VALID_BROKERS:
        raise HTTPException(400, f"broker must be one of {sorted(VALID_BROKERS)}")
    symbol = body.symbol.upper().strip()
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol).first()
    if row:
        row.system = body.system
        row.strategy_name = body.strategy_name
        row.enabled = body.enabled
        row.max_capital_usd = body.max_capital_usd
        row.max_shares = body.max_shares
        row.broker = body.broker
        row.notes = body.notes
        row.assigned_at = datetime.now(tz=timezone.utc)
    else:
        row = SymbolStrategyAssignment(
            symbol=symbol,
            system=body.system,
            strategy_name=body.strategy_name,
            enabled=body.enabled,
            max_capital_usd=body.max_capital_usd,
            max_shares=body.max_shares,
            broker=body.broker,
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
def set_cap(symbol: str, max_capital_usd: float | None = None, db: Session = Depends(get_db)):
    """Update only the capital cap on an existing assignment.

    Omit the query param (or pass 0) to clear the cap.
    """
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    row.max_capital_usd = max_capital_usd if max_capital_usd and max_capital_usd > 0 else None
    db.commit()
    return {"symbol": row.symbol, "max_capital_usd": row.max_capital_usd}


@router.patch("/{symbol}/broker")
def set_broker(symbol: str, broker: str, db: Session = Depends(get_db)):
    """Update only the broker route for an existing assignment.

    "default" means follow the global active_broker / trade_routing toggle.
    Otherwise route this symbol's orders to a specific broker adapter.
    """
    if broker not in VALID_BROKERS:
        raise HTTPException(400, f"broker must be one of {sorted(VALID_BROKERS)}")
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    row.broker = broker
    db.commit()
    return {"symbol": row.symbol, "broker": row.broker}


@router.patch("/{symbol}/shares")
def set_shares(symbol: str, max_shares: float | None = None, db: Session = Depends(get_db)):
    """Update only the shares cap on an existing assignment.

    Shares cap is the fallback used by the scheduler only when max_capital_usd
    is empty — dollar cap wins whenever both are set.
    """
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    row.max_shares = max_shares if max_shares and max_shares > 0 else None
    db.commit()
    return {"symbol": row.symbol, "max_shares": row.max_shares}


@router.delete("/{symbol}")
def delete_assignment(symbol: str, db: Session = Depends(get_db)):
    row = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper()).first()
    if not row:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    db.delete(row)
    db.commit()
    return {"deleted": symbol.upper()}
