from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(prefix="/webull", tags=["webull-auth"])


@router.get("/status")
async def webull_status():
    """Report Webull connection health.

    Webull uses HMAC-SHA1 request signing (app_key + app_secret), not OAuth,
    so there is no token to expire. 'connected' means credentials are present
    and a live account ping succeeded; 'not_configured' means the keys are
    missing from .env.

    state values:
      connected       — keys present, account reachable
      not_configured  — WEBULL_APP_KEY or WEBULL_APP_SECRET missing
      error           — keys present but account ping failed
    """
    settings = get_settings()

    if not settings.webull_app_key or not settings.webull_app_secret:
        return {
            "state": "not_configured",
            "detail": "WEBULL_APP_KEY and/or WEBULL_APP_SECRET not set in .env",
            "account_id": None,
        }

    # Attempt a live ping by fetching accounts.
    try:
        from app.services.brokers.webull import WebullBroker
        broker = WebullBroker()
        await broker.authenticate()
        accounts = await broker.get_accounts()
        acct_ids = [a.account_id for a in accounts] if accounts else []
        return {
            "state": "connected",
            "detail": f"{len(acct_ids)} account(s) found",
            "account_id": acct_ids[0] if acct_ids else settings.webull_account_id or None,
            "account_count": len(acct_ids),
        }
    except Exception as exc:
        return {
            "state": "error",
            "detail": str(exc),
            "account_id": settings.webull_account_id or None,
        }
