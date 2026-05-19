from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer

from app.schemas._serializers import serialize_et


class ScanConfig(BaseModel):
    universe: Literal["watchlist", "sp500", "nasdaq100", "custom"] = "watchlist"
    custom_symbols: List[str] = Field(default_factory=list)
    min_price: float = 5.0
    min_avg_volume: float = 500_000.0
    top_n: int = 5
    auto_trade_top: bool = False
    batch_size: int = 20


class ScanResultOut(BaseModel):
    id: int
    scan_run_id: str
    scanned_at: datetime
    universe: str
    symbol: str
    strategy_name: str
    direction: str
    score: float
    strategies_agreeing: int
    price: Optional[float]
    avg_volume: Optional[float]
    reason: Optional[str]
    auto_traded: bool

    class Config:
        from_attributes = True

    @field_serializer("scanned_at")
    def _ser_scanned_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class ScanSummary(BaseModel):
    scan_run_id: str
    scanned_at: datetime
    universe: str
    total_scanned: int
    total_passed_filters: int
    total_matches: int
    top_candidates: List[ScanResultOut]
    duration_seconds: float

    @field_serializer("scanned_at")
    def _ser_scanned_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)
