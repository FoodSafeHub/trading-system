from __future__ import annotations

"""Zerodha Kite Connect login flow.

Unlike Schwab's OAuth2, Kite's flow has no CSRF `state` round-trip: the user
opens the login URL, authenticates with Zerodha, and Kite redirects back to our
registered redirect_uri with `request_token` and `status`. We exchange the
request_token (with an api_secret-derived checksum) for a same-day access_token.

The access_token expires at ~07:30 IST daily with no refresh API, so this flow
must be re-run every morning before trading.
"""

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.brokers.zerodha import ZerodhaBroker

router = APIRouter(prefix="/zerodha", tags=["zerodha-auth"])


@router.get("/login")
async def zerodha_login_start():
    """Step 1: get the Kite login URL. Open it in your browser to authorize."""
    settings = get_settings()
    if not settings.zerodha_api_key:
        raise HTTPException(status_code=400, detail="ZERODHA_API_KEY not configured in .env")
    url = ZerodhaBroker().get_login_url()
    return {
        "login_url": url,
        "instructions": (
            "Open this URL, log in to Zerodha, and you'll be redirected back. "
            "The session lasts until ~07:30 IST tomorrow — re-run this daily."
        ),
    }


@router.get("/callback")
async def zerodha_callback(request_token: str | None = None, status: str | None = None):
    """Step 2: Kite redirects here. Exchange request_token for an access_token."""
    if status and status != "success":
        raise HTTPException(status_code=400, detail=f"Zerodha login not successful: status={status}")
    if not request_token:
        raise HTTPException(status_code=400, detail="missing request_token in callback")

    broker = ZerodhaBroker()
    try:
        await broker.exchange_request_token(request_token)
        return {
            "status": "success",
            "message": "Zerodha session established. Token is valid until ~07:30 IST tomorrow.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")
