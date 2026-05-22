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
    # Direction filter applied BEFORE the top-N slice. ANY keeps the legacy
    # behavior (rank all matches together). BUY/SELL drops the other side
    # entirely so the top_n window is filled exclusively with the requested
    # direction — useful when you want, say, 20 BUY candidates and don't want
    # SELL signals crowding them out.
    scan_direction: Literal["ANY", "BUY", "SELL"] = "ANY"
    auto_trade_top: bool = False
    # When auto-trading, only act on signals in this direction.
    # ANY keeps the old behavior (top candidate fires regardless of side).
    auto_trade_direction: Literal["ANY", "BUY", "SELL"] = "ANY"
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
