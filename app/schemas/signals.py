from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_serializer

from app.schemas._serializers import serialize_et


class SignalOut(BaseModel):
    id: int
    strategy_name: str
    symbol: str
    direction: Literal["BUY", "SELL", "HOLD"]
    strength: float
    price_at_signal: Optional[float]
    indicators_json: Optional[str]
    created_at: datetime
    acted_on: bool
    order_id: Optional[int]

    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def _ser_created_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class StrategyRunOut(BaseModel):
    id: int
    strategy_name: str
    symbol: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    signals_generated: int
    error_message: Optional[str]

    model_config = {"from_attributes": True}

    @field_serializer("started_at", "completed_at")
    def _ser_dt(self, dt: datetime | None) -> str | None:
        return serialize_et(dt)
