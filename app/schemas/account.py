from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class AccountSummary(BaseModel):
    broker: str
    account_id: str
    account_type: Optional[str] = None
    buying_power: Optional[float] = None
    cash: Optional[float] = None
    equity: Optional[float] = None
    is_paper: bool


class Position(BaseModel):
    symbol: str
    quantity: float
    average_cost: Optional[float] = None
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class Quote(BaseModel):
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[str] = None
