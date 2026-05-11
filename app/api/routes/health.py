from fastapi import APIRouter
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    s = get_settings()
    return {
        "status": "ok",
        "broker": s.active_broker,
        "mode": "PAPER" if not s.is_live else "LIVE",
        "live_trading_enabled": s.live_trading_enabled,
        "live_trading_confirmed": s.live_trading_confirmed,
    }
