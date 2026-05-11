from fastapi import APIRouter

from app.schemas.risk import RiskStatusOut
from app.services.risk.engine import RiskEngine

router = APIRouter(prefix="/risk", tags=["risk"])
_risk = RiskEngine()


@router.get("/status", response_model=RiskStatusOut)
def risk_status():
    return _risk.get_status()


@router.post("/kill-switch")
def set_kill_switch(active: bool):
    """Activate (true) or deactivate (false) the trading kill switch."""
    _risk.set_kill_switch(active)
    return {"kill_switch_active": active}
