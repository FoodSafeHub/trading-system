from __future__ import annotations

"""
Webull OpenAPI adapter — SCAFFOLD / PLACEHOLDER.

TODO: Webull's OpenAPI (distinct from the unofficial reverse-engineered API)
was in limited availability / beta at time of writing. The endpoints below
are based on publicly available Webull OpenAPI documentation drafts.

Before going live:
  1. Register as a Webull OpenAPI developer at https://developer.webull.com
  2. Confirm the base URL, token endpoint, and order payload format.
  3. Replace all TODO blocks below with real implementations.
  4. Remove the NotImplementedError calls once implemented.

All methods raise NotImplementedError until implemented.
"""

import logging
from typing import Dict, List, Optional

from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)

# TODO: Confirm these URLs once Webull OpenAPI is fully launched
WEBULL_BASE_URL = "https://openapi.webull.com/v1"  # ASSUMPTION — verify before use
WEBULL_TOKEN_URL = "https://openapi.webull.com/v1/oauth/token"  # ASSUMPTION


class WebullBroker(BrokerBase):
    """
    Webull OpenAPI broker adapter.
    All methods are scaffolded but raise NotImplementedError.
    Implement each method once Webull OpenAPI credentials are available.
    """

    name = "webull"

    def __init__(self) -> None:
        self._access_token: str = ""
        self._refresh_token: str = ""

    async def authenticate(self) -> None:
        # TODO: Implement Webull OAuth / app-key authentication
        # TODO: Load tokens from DB or env (similar to SchwabBroker pattern)
        raise NotImplementedError(
            "Webull adapter not yet implemented. "
            "See app/services/brokers/webull.py for TODOs."
        )

    async def refresh_token_if_needed(self) -> None:
        # TODO: Check expiry and POST to WEBULL_TOKEN_URL with grant_type=refresh_token
        raise NotImplementedError("Webull: refresh_token_if_needed not implemented")

    async def get_accounts(self) -> List[AccountSummary]:
        # TODO: GET /accounts — return AccountSummary list
        raise NotImplementedError("Webull: get_accounts not implemented")

    async def get_positions(self, account_id: str) -> List[Position]:
        # TODO: GET /positions?accountId={account_id}
        raise NotImplementedError("Webull: get_positions not implemented")

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        # TODO: GET /quotes?symbols=AAPL,MSFT
        raise NotImplementedError("Webull: get_quotes not implemented")

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        # TODO: Check if Webull OpenAPI supports order preview / dry-run
        raise NotImplementedError("Webull: preview_order not implemented")

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        # TODO: POST /orders with Webull-specific payload
        # TODO: Map OrderRequest fields to Webull payload format
        raise NotImplementedError("Webull: place_order not implemented")

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        # TODO: DELETE /orders/{broker_order_id}
        raise NotImplementedError("Webull: cancel_order not implemented")

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        # TODO: GET /orders/{broker_order_id}
        raise NotImplementedError("Webull: get_order not implemented")

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        # TODO: GET /orders?accountId={account_id}&status={status}
        raise NotImplementedError("Webull: list_orders not implemented")
