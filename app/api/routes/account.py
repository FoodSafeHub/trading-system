from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.account import AccountSummary, Position, Quote
from app.services.brokers.factory import _build_one, get_broker

router = APIRouter(prefix="/account", tags=["account"])

_KNOWN_BROKERS = {"paper", "schwab", "webull", "zerodha"}


@router.get("/summary", response_model=List[AccountSummary])
async def get_account_summary():
    broker = get_broker()
    await broker.authenticate()
    return await broker.get_accounts()


@router.get("/positions", response_model=List[Position])
async def get_positions(account_id: str = ""):
    broker = get_broker()
    await broker.authenticate()
    accounts = await broker.get_accounts()
    if not accounts:
        raise HTTPException(status_code=404, detail="No accounts found")
    # If account_id is empty and we're on MultiBroker, get_positions("") fans
    # out across every broker and returns positions tagged with their broker.
    # On a single broker, we still need to pass the resolved account_id.
    if account_id:
        return await broker.get_positions(account_id)
    if getattr(broker, "name", "").startswith("multi:"):
        return await broker.get_positions("")
    return await broker.get_positions(accounts[0].account_id)


@router.get("/{broker}/summary", response_model=List[AccountSummary])
async def get_broker_account_summary(broker: str):
    """Accounts for a SPECIFIC broker, independent of the global routing toggle.

    Lets the India page (and any per-broker view) pull Zerodha data directly even
    when global routing points at the US brokers. Degrades to [] on auth failure
    so the UI shows 'not authenticated' rather than a 500.
    """
    broker = broker.lower().strip()
    if broker not in _KNOWN_BROKERS:
        raise HTTPException(status_code=404, detail=f"Unknown broker {broker!r}")
    try:
        b = _build_one(broker)
        await b.authenticate()
        return await b.get_accounts()
    except Exception:
        return []


@router.get("/{broker}/positions", response_model=List[Position])
async def get_broker_positions(broker: str):
    """Positions for a SPECIFIC broker, independent of the global routing toggle."""
    broker = broker.lower().strip()
    if broker not in _KNOWN_BROKERS:
        raise HTTPException(status_code=404, detail=f"Unknown broker {broker!r}")
    try:
        b = _build_one(broker)
        await b.authenticate()
        accounts = await b.get_accounts()
        if not accounts:
            return []
        return await b.get_positions(accounts[0].account_id)
    except Exception:
        return []


@router.get("/quotes", response_model=dict)
async def get_quotes(symbols: str):
    """Pass comma-separated symbols: ?symbols=AAPL,MSFT"""
    broker = get_broker()
    await broker.authenticate()
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return await broker.get_quotes(sym_list)
