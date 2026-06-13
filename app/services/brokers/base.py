from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse


class BrokerBase(ABC):
    """Abstract interface every broker adapter must implement."""

    name: str = "base"

    # Whether the broker accepts a native TRAILING_STOP order type that it
    # ratchets server-side. Brokers that DON'T (e.g. Zerodha/Kite, which
    # removed trailing SL years ago) must set this False so the execution
    # layer places a static STOP instead and relies on the Chandelier job to
    # ratchet it. Defaults False so an unset adapter is treated as "no native
    # trail" — fail safe (a static stop), never a silent MARKET conversion.
    supports_native_trailing_stop: bool = False

    @abstractmethod
    async def authenticate(self) -> None:
        """Perform initial authentication / load stored tokens."""

    @abstractmethod
    async def refresh_token_if_needed(self) -> None:
        """Check token expiry and refresh if within the refresh window."""

    @abstractmethod
    async def get_accounts(self) -> List[AccountSummary]:
        """Return all accounts for the authenticated user."""

    @abstractmethod
    async def get_positions(self, account_id: str) -> List[Position]:
        """Return current open positions for an account."""

    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Return latest quotes for the given ticker symbols."""

    @abstractmethod
    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        """
        Return an order cost/commission estimate without submitting.
        Brokers that do not support preview should raise NotImplementedError.
        """

    @abstractmethod
    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        """Submit an order to the broker. Returns initial status."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        """Cancel an open order. Returns True on success."""

    @abstractmethod
    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        """Retrieve current status of a specific order."""

    @abstractmethod
    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        """List orders filtered by optional status string."""
