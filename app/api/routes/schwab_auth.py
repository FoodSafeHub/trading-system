from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.services.brokers.schwab import SchwabBroker

router = APIRouter(prefix="/schwab", tags=["schwab-auth"])


@router.get("/auth")
async def schwab_auth_start():
    """Step 1: Get the Schwab OAuth authorization URL. Visit this URL in your browser."""
    settings = get_settings()
    if not settings.schwab_client_id:
        raise HTTPException(status_code=400, detail="SCHWAB_CLIENT_ID not configured in .env")
    broker = SchwabBroker()
    url = broker.get_authorization_url()
    return {"authorization_url": url, "instructions": "Open this URL in your browser to authorize the app."}


@router.get("/callback")
async def schwab_auth_callback(code: str, request: Request):
    """
    Step 2: Schwab redirects here after user authorizes.
    Exchanges the authorization code for tokens and stores them.
    """
    broker = SchwabBroker()
    try:
        await broker.exchange_code_for_tokens(code)
        return {"status": "success", "message": "Schwab tokens stored. You can now use the trading API."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")
