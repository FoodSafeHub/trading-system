from __future__ import annotations

"""Upstox OAuth2 login — India MARKET-DATA only (orders run on Zerodha).

Standard authorization-code grant. The access token is single-day (~03:30 IST
expiry, no refresh), so this must be re-run each morning. Token is stored in
broker_tokens under broker="upstox" and read by app.services.marketdata.upstox_data.
"""

import secrets
import time

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.services.marketdata import upstox_data

router = APIRouter(prefix="/upstox", tags=["upstox-auth"])

# CSRF state nonces, same process-local pattern as schwab_auth.
_PENDING_STATES: dict[str, float] = {}
_STATE_TTL_SECONDS = 600


def _purge() -> None:
    now = time.time()
    for s, exp in list(_PENDING_STATES.items()):
        if exp < now:
            _PENDING_STATES.pop(s, None)


@router.get("/login")
async def upstox_login_start():
    """Step 1: get the Upstox login URL. Open it to authorize the data feed."""
    settings = get_settings()
    if not settings.upstox_api_key:
        raise HTTPException(status_code=400, detail="UPSTOX_API_KEY not configured in .env")
    _purge()
    state = secrets.token_urlsafe(32)
    _PENDING_STATES[state] = time.time() + _STATE_TTL_SECONDS
    url = upstox_data.get_login_url(state=state)
    return {
        "login_url": url,
        "instructions": (
            "Open this URL, log in to Upstox, and you'll be redirected back. "
            "This authorizes market data only — orders still go to Zerodha. "
            "Token lasts until ~03:30 IST; re-run daily."
        ),
    }


@router.get("/resolve/{symbol}")
async def upstox_resolve(symbol: str):
    """Validate a free-typed NSE ticker against the Upstox instrument map.

    Returns {symbol, tradeable, instrument_key}. Used by the dashboard so users
    can backtest ANY of the ~2,466 NSE equities, not just the curated tiers.
    """
    from app.services.marketdata import upstox_instruments as instr
    sym = (symbol or "").upper().strip()
    key = instr.resolve(sym)
    return {"symbol": sym, "tradeable": bool(key), "instrument_key": key}


@router.get("/universe/{tier}")
async def upstox_universe(tier: str):
    """Return the symbol list for an India tier: nifty50/100/200/500 or nse_all."""
    from app.services.scanner.universe_service import get_india_universe
    tier = (tier or "").lower().strip()
    if tier not in ("nifty50", "nifty100", "nifty200", "nifty500", "nse_all"):
        raise HTTPException(status_code=400, detail=f"Unknown India tier {tier!r}")
    syms = get_india_universe(tier)
    return {"tier": tier, "count": len(syms), "symbols": syms}


@router.get("/callback")
async def upstox_callback(code: str | None = None, state: str | None = None, request: Request = None):
    """Step 2: Upstox redirects here with the auth code. Exchange + store it."""
    _purge()
    if not state or _PENDING_STATES.pop(state, None) is None:
        raise HTTPException(status_code=400, detail="invalid or missing OAuth state")
    if not code:
        raise HTTPException(status_code=400, detail="missing authorization code")
    try:
        await upstox_data.exchange_code(code)
        return {"status": "success", "message": "Upstox data feed authorized. Token valid until ~03:30 IST tomorrow."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")
