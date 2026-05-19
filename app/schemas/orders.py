from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator

from app.schemas._serializers import serialize_et


class OrderRequest(BaseModel):
    """Broker-agnostic order request passed into the execution layer."""
    symbol: str = Field(..., min_length=1, max_length=16)
    side: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"] = "MARKET"
    quantity: float = Field(..., gt=0)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal["DAY", "GTC", "IOC", "FOK"] = "DAY"
    signal_id: Optional[int] = None
    idempotency_key: Optional[str] = None  # set by execution service

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.upper().strip()


class OrderPreviewResponse(BaseModel):
    broker: str
    estimated_cost: Optional[float] = None
    estimated_commission: Optional[float] = None
    buying_power_effect: Optional[float] = None
    raw: dict = Field(default_factory=dict)


class OrderStatusResponse(BaseModel):
    broker_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: float
    status: str
    fill_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    raw: dict = Field(default_factory=dict)


class OrderOut(BaseModel):
    id: int
    broker: str
    broker_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    is_paper: bool
    signal_id: Optional[int] = None
    preview_json: Optional[str] = None
    created_at: datetime
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    fill_price: Optional[float]
    error_message: Optional[str]

    model_config = {"from_attributes": True}

    @field_serializer("created_at", "submitted_at", "filled_at")
    def _ser_dt(self, dt: datetime | None) -> str | None:
        return serialize_et(dt)
