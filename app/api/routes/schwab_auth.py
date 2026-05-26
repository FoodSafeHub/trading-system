import secrets
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.services.brokers.schwab import SchwabBroker

router = APIRouter(prefix="/schwab", tags=["schwab-auth"])

# CSRF: in-process state nonces. {state: expires_at_epoch}.
# Process-local is fine because the OAuth round-trip takes seconds and the user
# is the only one driving it. Restarting the API invalidates pending flows,
# which is the safe direction.
_PENDING_STATES: dict[str, float] = {}
_STATE_TTL_SECONDS = 600  # 10 min — enough for the user to log in to Schwab


def _purge_expired_states() -> None:
    now = time.time()
    for s, exp in list(_PENDING_STATES.items()):
        if exp < now:
            _PENDING_STATES.pop(s, None)


@router.get("/status")
async def schwab_status():
    """Report Schwab connection health without leaking token material.

    The dashboard polls this so an expired token shows as a clear 'reconnect'
    nudge instead of Schwab silently vanishing from the broker tabs (an expired
    token makes get_accounts() raise, which the Home page swallows).

    `state` is one of:
      - "connected"     : access token present and not past expiry
      - "expiring"       : within the refresh buffer window (auto-refresh should kick in)
      - "expired"        : past expiry — needs a working refresh token or re-auth
      - "disconnected"   : no token row at all — never authorized
      - "not_configured" : SCHWAB_CLIENT_ID missing from .env
    """
    settings = get_settings()
    if not settings.schwab_client_id:
        return {"state": "not_configured", "detail": "SCHWAB_CLIENT_ID not set in .env"}

    from app.db import SessionLocal
    from app.models.broker_tokens import BrokerToken

    with SessionLocal() as db:
        row = db.query(BrokerToken).filter_by(broker="schwab").first()

    if not row or not row.access_token:
        return {"state": "disconnected", "has_refresh_token": False, "expires_at": None}

    expiry = row.token_expiry
    if expiry and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    has_refresh = bool(row.refresh_token)

    if expiry is None:
        state = "connected"
        seconds_left = None
    else:
        seconds_left = (expiry - now).total_seconds()
        if seconds_left <= 0:
            state = "expired"
        elif seconds_left <= 300:  # mirrors TOKEN_REFRESH_BUFFER_SECONDS in schwab.py
            state = "expiring"
        else:
            state = "connected"

    return {
        "state": state,
        "has_refresh_token": has_refresh,
        "expires_at": expiry.isoformat() if expiry else None,
        "seconds_left": seconds_left,
        "account_number": settings.schwab_account_number or None,
    }


@router.get("/auth")
async def schwab_auth_start():
    """Step 1: Get the Schwab OAuth authorization URL. Visit this URL in your browser."""
    settings = get_settings()
    if not settings.schwab_client_id:
        raise HTTPException(status_code=400, detail="SCHWAB_CLIENT_ID not configured in .env")
    _purge_expired_states()
    state = secrets.token_urlsafe(32)
    _PENDING_STATES[state] = time.time() + _STATE_TTL_SECONDS
    broker = SchwabBroker()
    url = broker.get_authorization_url(state=state)
    return {"authorization_url": url, "instructions": "Open this URL in your browser to authorize the app."}


@router.get("/callback")
async def schwab_auth_callback(code: str, request: Request, state: str | None = None):
    """
    Step 2: Schwab redirects here after user authorizes.
    Exchanges the authorization code for tokens and stores them.
    """
    _purge_expired_states()
    if not state or _PENDING_STATES.pop(state, None) is None:
        raise HTTPException(status_code=400, detail="invalid or missing OAuth state")
    broker = SchwabBroker()
    try:
        await broker.exchange_code_for_tokens(code)
        return {"status": "success", "message": "Schwab tokens stored. You can now use the trading API."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")
