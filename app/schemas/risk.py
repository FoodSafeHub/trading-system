from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class RiskCheckResult(BaseModel):
    passed: bool
    blocked_reason: Optional[str] = None
    warnings: List[str] = []


class RiskStatusOut(BaseModel):
    kill_switch_active: bool
    live_trading_enabled: bool
    live_trading_confirmed: bool
    active_broker: str
    is_live: bool
    orders_today: int
    max_orders_per_day: int
    daily_loss_usd: float
    max_daily_loss_usd: float
    market_hours_active: bool
