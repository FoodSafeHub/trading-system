"""Multi-broker fan-out adapter.

Wraps two or more brokers behind the BrokerBase interface so a single
ExecutionService call places the order at every configured broker.

Failure semantics: each broker call is independent. A failure at one does
not prevent the order being attempted at the other. The returned status is
the first successful response (so callers see something coherent); per-broker
results are logged so attribution is visible.

Read methods (get_accounts, get_quotes, etc.) delegate to the *primary*
broker only — fan-out doesn't make sense for queries that return state.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)


class MultiBroker(BrokerBase):
    """Fan out write operations across multiple brokers; read from the primary."""

    name = "multi"

    def __init__(self, brokers: List[BrokerBase]) -> None:
        if not brokers:
            raise ValueError("MultiBroker requires at least one broker")
        self._brokers = brokers
        self._primary = brokers[0]
        self.name = "multi:" + "+".join(b.name for b in brokers)

    async def authenticate(self) -> None:
        results = await asyncio.gather(
            *(b.authenticate() for b in self._brokers), return_exceptions=True
        )
        for b, r in zip(self._brokers, results):
            if isinstance(r, Exception):
                logger.warning("[multi-broker] %s authenticate failed: %s", b.name, r)

    async def refresh_token_if_needed(self) -> None:
        results = await asyncio.gather(
            *(b.refresh_token_if_needed() for b in self._brokers), return_exceptions=True
        )
        for b, r in zip(self._brokers, results):
            if isinstance(r, Exception):
                logger.warning("[multi-broker] %s refresh failed: %s", b.name, r)

    async def get_accounts(self) -> List[AccountSummary]:
        """Fan out — concatenate accounts from every broker so the dashboard
        can show per-broker equity/cash side-by-side. A failure at one broker
        does not blank out the others.
        """
        results = await asyncio.gather(
            *(b.get_accounts() for b in self._brokers), return_exceptions=True
        )
        out: List[AccountSummary] = []
        for b, r in zip(self._brokers, results):
            if isinstance(r, Exception):
                logger.warning("[multi-broker] %s get_accounts failed: %s", b.name, r)
                continue
            out.extend(r)
        return out

    async def get_positions(self, account_id: str = "") -> List[Position]:
        """Fan out — pull each broker's first account and concatenate positions,
        tagged by broker so callers can group. When `account_id` is supplied it
        is forwarded only to the broker that owns it; brokers whose first
        account doesn't match are skipped silently.
        """
        async def _for_broker(b: BrokerBase) -> List[Position]:
            try:
                accts = await b.get_accounts()
            except Exception as exc:
                logger.warning("[multi-broker] %s get_accounts failed: %s", b.name, exc)
                return []
            if not accts:
                return []
            acct_id = accts[0].account_id
            if account_id and account_id != acct_id:
                return []
            try:
                return await b.get_positions(acct_id)
            except Exception as exc:
                logger.warning("[multi-broker] %s get_positions failed: %s", b.name, exc)
                return []

        results = await asyncio.gather(*(_for_broker(b) for b in self._brokers))
        out: List[Position] = []
        for rows in results:
            out.extend(rows)
        return out

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return await self._primary.get_quotes(symbols)

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        return await self._primary.preview_order(order, account_id)

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        """Fan out to every broker. Return the first successful status.

        If all fail, re-raise the primary's exception so the caller gets a
        coherent error path.
        """
        results = await asyncio.gather(
            *(b.place_order(order, account_id) for b in self._brokers),
            return_exceptions=True,
        )
        first_ok: Optional[OrderStatusResponse] = None
        primary_exc: Optional[BaseException] = None
        for b, r in zip(self._brokers, results):
            if isinstance(r, Exception):
                logger.error("[multi-broker] %s place_order failed: %s", b.name, r)
                if b is self._primary:
                    primary_exc = r
            else:
                logger.info(
                    "[multi-broker] %s placed %s %s qty=%s -> %s",
                    b.name, order.side, order.symbol, order.quantity,
                    getattr(r, "status", "ok"),
                )
                if first_ok is None:
                    first_ok = r
        if first_ok is not None:
            return first_ok
        if primary_exc is not None:
            raise primary_exc
        raise RuntimeError("[multi-broker] all brokers failed and primary had no exception")

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        return await self._primary.cancel_order(broker_order_id, account_id)

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        return await self._primary.get_order(broker_order_id, account_id)

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        return await self._primary.list_orders(account_id, status)
