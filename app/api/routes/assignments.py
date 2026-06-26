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
VALID_BROKERS = {"default", "paper", "schwab", "webull", "zerodha"}


class AssignmentIn(BaseModel):
    symbol: str
    system: str        # "bollinger" | "perplexity" | "scanner"
    strategy_name: str
    enabled: bool = True
    max_capital_usd: float | None = None   # None = use global account settings
    max_shares: float | None = None        # Fallback shares cap when no dollar cap
    broker: str = "default"                # "default" | "paper" | "schwab" | "webull"
    notes: str = ""
    # Approach C tight trailing stop % — set during backtesting, stored per assignment.
    # None = use system default (2.0%). Range 1.0–10.0.
    tight_trail_pct: float | None = None


class AssignmentOut(BaseModel):
    symbol: str
    system: str
    strategy_name: str
    enabled: bool
    max_capital_usd: float | None
    max_shares: float | None
    broker: str
    notes: str | None
    tight_trail_pct: float | None = None
    assigned_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("assigned_at")
    def _ser_assigned_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


def _get_assignment(
    db: Session, symbol: str, system: str | None, strategy_name: str | None
) -> SymbolStrategyAssignment:
    """Resolve a single assignment by its (symbol, system, strategy_name) identity.

    A symbol may now have several assignments. When system/strategy_name are omitted
    and the symbol has exactly one assignment, that one is used (back-compat for the
    old single-assignment-per-symbol callers). Ambiguity raises 400.
    """
    q = db.query(SymbolStrategyAssignment).filter_by(symbol=symbol.upper())
    if system:
        q = q.filter_by(system=system)
    if strategy_name:
        q = q.filter_by(strategy_name=strategy_name)
    rows = q.all()
    if not rows:
        raise HTTPException(404, f"No assignment found for {symbol.upper()}")
    if len(rows) > 1:
        raise HTTPException(
            400,
            f"{symbol.upper()} has multiple assignments — specify system and "
            f"strategy_name to target one.",
        )
    return rows[0]


@router.get("", response_model=List[AssignmentOut])
def list_assignments(db: Session = Depends(get_db)):
    return (
        db.query(SymbolStrategyAssignment)
        .order_by(
            SymbolStrategyAssignment.symbol,
            SymbolStrategyAssignment.strategy_name,
        )
        .all()
    )


@router.post("", response_model=AssignmentOut)
def upsert_assignment(body: AssignmentIn, db: Session = Depends(get_db)):
    """Create or update one strategy assignment, keyed by (symbol, system, strategy_name).

    A symbol may have several assignments — this only touches the row matching the
    full triple, so adding a second strategy to a symbol no longer clobbers the first.
    """
    if body.system not in VALID_SYSTEMS:
        raise HTTPException(400, f"system must be one of {sorted(VALID_SYSTEMS)}")
    if body.broker not in VALID_BROKERS:
        raise HTTPException(400, f"broker must be one of {sorted(VALID_BROKERS)}")
    symbol = body.symbol.upper().strip()
    row = (
        db.query(SymbolStrategyAssignment)
        .filter_by(symbol=symbol, system=body.system, strategy_name=body.strategy_name)
        .first()
    )
    if row:
        row.system = body.system
        row.strategy_name = body.strategy_name
        row.enabled = body.enabled
        row.max_capital_usd = body.max_capital_usd
        row.max_shares = body.max_shares
        row.broker = body.broker
        row.notes = body.notes
        row.tight_trail_pct = body.tight_trail_pct
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
            tight_trail_pct=body.tight_trail_pct,
            assigned_at=datetime.now(tz=timezone.utc),
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.patch("/{symbol}/toggle")
def toggle_assignment(
    symbol: str, enabled: bool,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    row = _get_assignment(db, symbol, system, strategy_name)
    row.enabled = enabled
    db.commit()
    return {"symbol": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name, "enabled": row.enabled}


@router.patch("/{symbol}/cap")
def set_cap(
    symbol: str, max_capital_usd: float | None = None,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    """Update only the capital cap on an existing assignment.

    Omit the query param (or pass 0) to clear the cap.
    """
    row = _get_assignment(db, symbol, system, strategy_name)
    row.max_capital_usd = max_capital_usd if max_capital_usd and max_capital_usd > 0 else None
    db.commit()
    return {"symbol": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name, "max_capital_usd": row.max_capital_usd}


class BulkBrokerIn(BaseModel):
    # Explicit list of symbols to retag. Empty + auto_by_market=True means
    # "retag every existing assignment by its detected market".
    symbols: List[str] = []
    broker: str = "default"          # ignored when auto_by_market is True
    auto_by_market: bool = False     # India (NSE/BSE) -> zerodha, US -> "default"


@router.post("/bulk-broker")
def bulk_set_broker(body: BulkBrokerIn, db: Session = Depends(get_db)):
    """Set the broker route on many assignments at once.

    Two modes:
      * explicit  — set every listed symbol to `broker`.
      * auto_by_market — route each symbol to the broker its market implies:
        India (NSE/BSE) -> "zerodha", everything else -> "default" (so US
        symbols keep following the global toggle). When symbols is empty,
        applies to every existing assignment.
    """
    from app.services.markets import is_india_symbol

    if not body.auto_by_market and body.broker not in VALID_BROKERS:
        raise HTTPException(400, f"broker must be one of {sorted(VALID_BROKERS)}")

    if body.symbols:
        targets = {s.upper().strip() for s in body.symbols if s.strip()}
        rows = (
            db.query(SymbolStrategyAssignment)
            .filter(SymbolStrategyAssignment.symbol.in_(targets))
            .all()
        )
    else:
        rows = db.query(SymbolStrategyAssignment).all()

    updated = []
    for row in rows:
        if body.auto_by_market:
            # Only (re)route India symbols to Zerodha. Leave non-India symbols
            # exactly as they are so explicit Schwab/Webull/paper pins are never
            # clobbered — auto-route is additive, not destructive.
            if is_india_symbol(row.symbol):
                if row.broker != "zerodha":
                    row.broker = "zerodha"
                    updated.append({"symbol": row.symbol, "broker": row.broker})
            continue
        row.broker = body.broker
        updated.append({"symbol": row.symbol, "broker": row.broker})
    db.commit()
    return {"updated": updated, "count": len(updated)}


@router.patch("/{symbol}/broker")
def set_broker(
    symbol: str, broker: str,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    """Update only the broker route for an existing assignment.

    "default" means follow the global active_broker / trade_routing toggle.
    Otherwise route this symbol's orders to a specific broker adapter.
    """
    if broker not in VALID_BROKERS:
        raise HTTPException(400, f"broker must be one of {sorted(VALID_BROKERS)}")
    row = _get_assignment(db, symbol, system, strategy_name)
    row.broker = broker
    db.commit()
    return {"symbol": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name, "broker": row.broker}


@router.patch("/{symbol}/shares")
def set_shares(
    symbol: str, max_shares: float | None = None,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    """Update only the shares cap on an existing assignment.

    Shares cap is the fallback used by the scheduler only when max_capital_usd
    is empty — dollar cap wins whenever both are set.
    """
    row = _get_assignment(db, symbol, system, strategy_name)
    row.max_shares = max_shares if max_shares and max_shares > 0 else None
    db.commit()
    return {"symbol": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name, "max_shares": row.max_shares}


@router.patch("/{symbol}/trail")
def set_trail(
    symbol: str, tight_trail_pct: float | None = None,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    """Update only the Approach C tight trailing stop % on an existing assignment.

    Pass tight_trail_pct=0 or omit to reset to system default (2%).
    Valid range: 1.0–10.0. Values outside this range are clamped.
    """
    row = _get_assignment(db, symbol, system, strategy_name)
    if tight_trail_pct and tight_trail_pct > 0:
        row.tight_trail_pct = round(max(1.0, min(10.0, tight_trail_pct)), 2)
    else:
        row.tight_trail_pct = None  # reset to system default
    db.commit()
    return {"symbol": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name, "tight_trail_pct": row.tight_trail_pct}


@router.delete("/{symbol}")
def delete_assignment(
    symbol: str,
    system: str | None = None, strategy_name: str | None = None,
    db: Session = Depends(get_db),
):
    """Delete one assignment. With several assignments on the symbol, system and
    strategy_name are required to disambiguate which to remove."""
    row = _get_assignment(db, symbol, system, strategy_name)
    db.delete(row)
    db.commit()
    return {"deleted": row.symbol, "system": row.system,
            "strategy_name": row.strategy_name}
