from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.account import AccountSummary, Position, Quote
from app.services.brokers.factory import (
    _build_one,
    get_broker,
    get_position_brokers,
)

router = APIRouter(prefix="/account", tags=["account"])

_KNOWN_BROKERS = {"paper", "schwab", "webull", "zerodha"}


@router.get("/summary", response_model=List[AccountSummary])
async def get_account_summary():
    """Accounts across EVERY broker that could hold a position.

    The cockpit builds its per-broker tabs from whatever this returns, so it
    must enumerate all position-holding brokers (global route + per-assignment
    overrides — Webull, Zerodha) rather than only the global broker. Otherwise a
    broker off the active route (e.g. Webull under trade_routing="auto") never
    appears on the dashboard even though it holds positions and cash. Best-
    effort per broker: a failed auth degrades to skipping that broker, not a 500.
    """
    out: List[AccountSummary] = []
    for b in get_position_brokers():
        try:
            await b.authenticate()
            out.extend(await b.get_accounts())
        except Exception:
            continue
    return out


def _held_only(positions: List[Position]) -> List[Position]:
    """Drop flat (qty 0) lots so a sold symbol never shows as 'open'.

    Defense in depth: each broker adapter already filters zero-quantity lots,
    but a broker can return a recently-closed position with net qty 0 in its
    positions array. Filtering here guarantees the home page and any consumer
    of /account/positions only ever sees genuinely-held positions.
    """
    return [p for p in positions if (p.quantity or 0) != 0]


@router.get("/positions", response_model=List[Position])
async def get_positions(account_id: str = ""):
    # A specific account_id targets one broker (the global one owns it).
    if account_id:
        broker = get_broker()
        await broker.authenticate()
        return _held_only(await broker.get_positions(account_id))

    # No account_id → the cockpit's "all positions" call. Fan out across EVERY
    # position-holding broker (global route + per-assignment overrides — Webull,
    # Zerodha) so the dashboard shows holdings on brokers off the active route.
    # Each Position is tagged with its broker. Best-effort per broker.
    out: List[Position] = []
    for b in get_position_brokers():
        try:
            await b.authenticate()
            accts = await b.get_accounts()
            if not accts:
                continue
            out.extend(await b.get_positions(accts[0].account_id))
        except Exception:
            continue
    return _held_only(out)


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
        return _held_only(await b.get_positions(accounts[0].account_id))
    except Exception:
        return []


@router.get("/quotes", response_model=dict)
async def get_quotes(symbols: str):
    """Pass comma-separated symbols: ?symbols=AAPL,MSFT"""
    broker = get_broker()
    await broker.authenticate()
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return await broker.get_quotes(sym_list)
