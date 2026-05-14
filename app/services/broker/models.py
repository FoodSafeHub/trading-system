"""
Broker data models — shared by all broker implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


OrderStatus = Literal["PENDING", "PARTIAL", "FILLED", "REJECTED", "CANCELLED", "EXPIRED"]
OrderSide   = Literal["BUY", "SELL"]
OrderType   = Literal["market", "limit", "stop", "stop_limit"]


@dataclass
class Order:
    order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    limit_price: float | None
    stop_price: float | None

    # Fill state
    status: OrderStatus = "PENDING"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    remaining_qty: float = 0.0

    # Timestamps
    placed_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Metadata
    strategy: str = ""
    reason: str = ""                # why this order was placed
    rejection_reason: str = ""      # populated on REJECTED
    broker_response: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_qty == 0.0:
            self.remaining_qty = self.qty

    @property
    def is_complete(self) -> bool:
        return self.status in ("FILLED", "REJECTED", "CANCELLED", "EXPIRED")

    @property
    def is_partial(self) -> bool:
        return self.status == "PARTIAL" and self.filled_qty > 0

    @property
    def fill_pct(self) -> float:
        return self.filled_qty / self.qty * 100 if self.qty > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "remaining_qty": self.remaining_qty,
            "avg_fill_price": self.avg_fill_price,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "strategy": self.strategy,
            "reason": self.reason,
            "rejection_reason": self.rejection_reason,
            "placed_at": self.placed_at.isoformat() if self.placed_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class Position:
    symbol: str
    qty: float                   # positive = long, negative = short
    avg_entry_price: float
    current_price: float = 0.0
    strategy: str = ""
    opened_at: datetime = field(default_factory=datetime.utcnow)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def market_value(self) -> float:
        return abs(self.qty) * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.qty) * self.avg_entry_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.qty > 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.qty
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * abs(self.qty)
        self.unrealized_pnl_pct = (
            self.unrealized_pnl / self.cost_basis * 100
            if self.cost_basis > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": round(self.avg_entry_price, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "strategy": self.strategy,
        }


@dataclass
class Account:
    account_id: str
    equity: float               # total portfolio value
    cash: float                 # available cash
    buying_power: float         # cash × margin (usually 4× for PDT)
    day_trades_used: int = 0    # PDT: max 3 per 5 rolling days if < $25k
    initial_capital: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    broker: str = "paper"

    PDT_MIN_EQUITY = 25_000.0
    PDT_MAX_DAY_TRADES = 3

    @property
    def is_pdt_restricted(self) -> bool:
        """True if account is under $25k AND has used 3 day trades."""
        return self.equity < self.PDT_MIN_EQUITY and self.day_trades_used >= self.PDT_MAX_DAY_TRADES

    @property
    def pdt_warning(self) -> str | None:
        if self.equity < self.PDT_MIN_EQUITY:
            remaining = self.PDT_MAX_DAY_TRADES - self.day_trades_used
            if remaining <= 0:
                return f"PDT BLOCKED: account ${self.equity:,.0f} < $25k, 0 day trades remaining."
            return f"PDT WARNING: ${self.equity:,.0f} < $25k. {remaining} day trade(s) left this week."
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "buying_power": round(self.buying_power, 2),
            "day_trades_used": self.day_trades_used,
            "pdt_restricted": self.is_pdt_restricted,
            "pdt_warning": self.pdt_warning,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "broker": self.broker,
        }
