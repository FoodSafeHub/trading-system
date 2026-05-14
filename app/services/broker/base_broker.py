"""
Abstract broker interface.

All broker implementations (Alpaca, IBKR, Schwab, paper) must implement
every method here. The rest of the system only talks to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from app.services.broker.models import Account, Order, OrderSide, OrderType, Position


class BaseBroker(ABC):

    @abstractmethod
    def get_account(self) -> Account:
        """Return current account state including buying power and PDT count."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all currently open positions."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Return open position for a symbol, or None."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        strategy: str = "",
        reason: str = "",
    ) -> Order:
        """
        Submit an order. Returns an Order immediately with status=PENDING.
        The order fills asynchronously — poll get_order() or use callbacks.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancellation was accepted."""
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        new_qty: float | None = None,
        new_limit_price: float | None = None,
    ) -> Order:
        """Modify quantity or limit price of a pending order."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order | None:
        """Retrieve current state of an order by internal ID."""
        ...

    @abstractmethod
    def get_orders(self, status: str | None = None) -> list[Order]:
        """
        Return orders filtered by status.
        status=None → all orders today.
        """
        ...

    @abstractmethod
    def close_position(self, symbol: str, reason: str = "") -> Order | None:
        """Submit a market order to close the full position in symbol."""
        ...

    @abstractmethod
    def close_all_positions(self, reason: str = "EOD") -> list[Order]:
        """Submit market orders to close ALL open positions."""
        ...

    def register_fill_callback(self, callback: Callable[[Order], None]) -> None:
        """
        Register a function to call whenever an order is filled or updated.
        Default: no-op. Override in implementations that support streaming.
        """

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """True if this is a paper/simulation broker."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable broker name."""
        ...
