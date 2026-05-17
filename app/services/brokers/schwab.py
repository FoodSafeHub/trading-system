from __future__ import annotations

"""
Charles Schwab Individual Trader API adapter.

Documentation reference: https://developer.schwab.com/products/trader-api--individual-
OAuth 2.0 authorization code flow with PKCE is used.

ASSUMPTIONS (marked where Schwab docs were ambiguous at time of writing):
  - Base URL: https://api.schwabapi.com/trader/v1
  - Token endpoint: https://api.schwabapi.com/v1/oauth/token
  - Auth endpoint: https://api.schwabapi.com/v1/oauth/authorize
  - Account hash is used (not raw account number) for all order endpoints — per Schwab docs.
  - Quotes endpoint returns JSON with symbol as key.
  - Order placement returns a Location header with the order ID.

TODO: Review and update endpoints against the latest Schwab API reference before going live.
"""

import base64
import logging
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)

SCHWAB_BASE_URL = "https://api.schwabapi.com/trader/v1"
SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TOKEN_REFRESH_BUFFER_SECONDS = 300  # refresh 5 min before expiry


class SchwabBroker(BrokerBase):
    name = "schwab"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._token_expiry: Optional[datetime] = None
        self._account_hash: str = ""  # Schwab uses hashed account ID in API calls

    # ── Auth ────────────────────────────────────────────────────────────────

    def get_authorization_url(self) -> str:
        """Return the URL the user must visit to authorize the app."""
        params = {
            "client_id": self._settings.schwab_client_id,
            "redirect_uri": self._settings.schwab_redirect_uri,
            "response_type": "code",
            "scope": "PlaceTrades AccountAccess MarketData",
        }
        return f"{SCHWAB_AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_tokens(self, authorization_code: str) -> None:
        """Exchange the OAuth authorization code for access + refresh tokens."""
        credentials = base64.b64encode(
            f"{self._settings.schwab_client_id}:{self._settings.schwab_client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                SCHWAB_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": self._settings.schwab_redirect_uri,
                },
                timeout=15,
            )
            resp.raise_for_status()
            self._store_tokens(resp.json())
            await self._save_tokens_to_db()
            logger.info("[schwab] Tokens exchanged successfully")

    async def authenticate(self) -> None:
        """Load tokens from DB or env, refresh if needed."""
        # Try loading from DB first
        stored = await self._load_tokens_from_db()
        if not stored:
            # Fall back to env vars (for initial bootstrap)
            self._access_token = self._settings.schwab_access_token
            self._refresh_token = self._settings.schwab_refresh_token
            if self._settings.schwab_token_expiry:
                self._token_expiry = datetime.fromisoformat(self._settings.schwab_token_expiry)

        if not self._access_token and not self._refresh_token:
            logger.warning(
                "[schwab] No tokens found. Run the OAuth flow first: "
                "GET /schwab/auth to get the authorization URL."
            )
            return

        await self.refresh_token_if_needed()

    async def refresh_token_if_needed(self) -> None:
        if not self._refresh_token:
            logger.warning("[schwab] No refresh token available")
            return

        now = datetime.now(tz=timezone.utc)
        expiry = self._token_expiry
        if expiry and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry and (expiry - now).total_seconds() > TOKEN_REFRESH_BUFFER_SECONDS:
            return  # Token still valid

        logger.info("[schwab] Refreshing access token...")
        credentials = base64.b64encode(
            f"{self._settings.schwab_client_id}:{self._settings.schwab_client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                SCHWAB_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
                timeout=15,
            )
            resp.raise_for_status()
            self._store_tokens(resp.json())
            await self._save_tokens_to_db()
            logger.info("[schwab] Access token refreshed")

    def _store_tokens(self, token_data: dict) -> None:
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token", self._refresh_token)
        expires_in = token_data.get("expires_in", 1800)
        self._token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)

    async def _load_tokens_from_db(self) -> bool:
        try:
            from app.db import SessionLocal
            from app.models.broker_tokens import BrokerToken

            with SessionLocal() as db:
                row = db.query(BrokerToken).filter_by(broker="schwab").first()
                if row:
                    self._access_token = row.access_token
                    self._refresh_token = row.refresh_token or ""
                    self._token_expiry = row.token_expiry
                    return True
        except Exception as exc:
            logger.warning("[schwab] Could not load tokens from DB: %s", exc)
        return False

    async def _save_tokens_to_db(self) -> None:
        try:
            from app.db import SessionLocal
            from app.models.broker_tokens import BrokerToken

            with SessionLocal() as db:
                row = db.query(BrokerToken).filter_by(broker="schwab").first()
                if not row:
                    row = BrokerToken(broker="schwab")
                    db.add(row)
                row.access_token = self._access_token
                row.refresh_token = self._refresh_token
                row.token_expiry = self._token_expiry
                db.commit()
        except Exception as exc:
            logger.error("[schwab] Failed to save tokens to DB: %s", exc)

    # ── Shared HTTP helper ───────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=SCHWAB_BASE_URL) as client:
            resp = await client.get(path, headers=self._auth_headers(), params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json: Optional[dict] = None) -> httpx.Response:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=SCHWAB_BASE_URL) as client:
            resp = await client.post(
                path, headers={**self._auth_headers(), "Content-Type": "application/json"},
                json=json, timeout=15,
            )
            resp.raise_for_status()
            return resp

    async def _delete(self, path: str) -> httpx.Response:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=SCHWAB_BASE_URL) as client:
            resp = await client.delete(path, headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            return resp

    # ── Account helpers ──────────────────────────────────────────────────────

    async def _get_account_hash(self) -> str:
        """
        Schwab requires the hashed account number for order endpoints.
        Cache it after the first successful fetch.
        ASSUMPTION: GET /accounts/accountNumbers returns [{accountNumber, hashValue}].
        """
        if self._account_hash:
            return self._account_hash
        data = await self._get("/accounts/accountNumbers")
        for entry in data:
            acct_num = entry.get("accountNumber", "")
            if (
                not self._settings.schwab_account_number
                or acct_num == self._settings.schwab_account_number
            ):
                self._account_hash = entry["hashValue"]
                return self._account_hash
        raise RuntimeError("Schwab account hash not found — check SCHWAB_ACCOUNT_NUMBER")

    # ── BrokerBase implementation ────────────────────────────────────────────

    async def get_accounts(self) -> List[AccountSummary]:
        data = await self._get("/accounts", params={"fields": "positions"})
        results = []
        for acct in data:
            sec = acct.get("securitiesAccount", {})
            balances = sec.get("currentBalances", {})
            results.append(
                AccountSummary(
                    broker="schwab",
                    account_id=sec.get("accountNumber", ""),
                    account_type=sec.get("type"),
                    buying_power=balances.get("buyingPower"),
                    cash=balances.get("cashBalance"),
                    equity=balances.get("liquidationValue"),
                    is_paper=False,
                )
            )
        return results

    async def get_positions(self, account_id: str) -> List[Position]:
        account_hash = await self._get_account_hash()
        data = await self._get(f"/accounts/{account_hash}", params={"fields": "positions"})
        raw_positions = data.get("securitiesAccount", {}).get("positions", [])
        return [
            Position(
                symbol=p["instrument"]["symbol"],
                quantity=p.get("longQuantity", 0) - p.get("shortQuantity", 0),
                average_cost=p.get("averagePrice"),
                current_price=p.get("marketValue") / p["longQuantity"] if p.get("longQuantity") else None,
                market_value=p.get("marketValue"),
                unrealized_pnl=p.get("unrealizedProfitOrLoss"),
            )
            for p in raw_positions
        ]

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        # ASSUMPTION: GET /marketdata/v1/quotes?symbols=AAPL,MSFT returns {AAPL: {...}, MSFT: {...}}
        data = await self._get(
            "/marketdata/v1/quotes",
            params={"symbols": ",".join(symbols), "fields": "quote"},
        )
        quotes: Dict[str, Quote] = {}
        for sym, info in data.items():
            q = info.get("quote", {})
            quotes[sym] = Quote(
                symbol=sym,
                bid=q.get("bidPrice"),
                ask=q.get("askPrice"),
                last=q.get("lastPrice"),
                volume=q.get("totalVolume"),
                timestamp=str(q.get("quoteTime", "")),
            )
        return quotes

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        """
        TODO: Schwab does not expose a dedicated order preview endpoint in the public
        Individual Trader API as of this writing. This method raises NotImplementedError.
        If Schwab adds a preview/dry-run endpoint in the future, implement it here.
        """
        raise NotImplementedError(
            "Schwab does not currently offer a public order preview endpoint. "
            "The execution service will skip the preview step for Schwab."
        )

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        account_hash = await self._get_account_hash()
        payload = self._build_order_payload(order)
        resp = await self._post(f"/accounts/{account_hash}/orders", json=payload)

        # Schwab returns 201 Created with a Location header containing the order ID
        location = resp.headers.get("Location", "")
        broker_order_id = location.rstrip("/").split("/")[-1] if location else None

        logger.info("[schwab] Order submitted: %s %s — broker_order_id=%s", order.side, order.symbol, broker_order_id)
        return OrderStatusResponse(
            broker_order_id=broker_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            status="submitted",
            raw={"location": location},
        )

    def _build_order_payload(self, order: OrderRequest) -> dict:
        """Map an OrderRequest to the Schwab REST API payload format."""
        leg = {
            "instruction": order.side,  # BUY | SELL
            "quantity": order.quantity,
            "instrument": {
                "symbol": order.symbol,
                "assetType": "EQUITY",
            },
        }
        payload: dict = {
            "orderType": order.order_type,
            "session": "NORMAL",
            "duration": order.time_in_force,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [leg],
        }
        if order.order_type in ("LIMIT", "STOP_LIMIT") and order.limit_price:
            payload["price"] = str(order.limit_price)
        if order.order_type in ("STOP", "STOP_LIMIT") and order.stop_price:
            payload["stopPrice"] = str(order.stop_price)
        return payload

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        account_hash = await self._get_account_hash()
        try:
            await self._delete(f"/accounts/{account_hash}/orders/{broker_order_id}")
            logger.info("[schwab] Order cancelled: %s", broker_order_id)
            return True
        except httpx.HTTPStatusError as exc:
            logger.error("[schwab] Cancel failed: %s", exc)
            return False

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        account_hash = await self._get_account_hash()
        data = await self._get(f"/accounts/{account_hash}/orders/{broker_order_id}")
        return self._parse_order_response(data)

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        account_hash = await self._get_account_hash()
        params: dict = {}
        if status:
            params["status"] = status
        data = await self._get(f"/accounts/{account_hash}/orders", params=params)
        return [self._parse_order_response(o) for o in data]

    def _parse_order_response(self, data: dict) -> OrderStatusResponse:
        leg = (data.get("orderLegCollection") or [{}])[0]
        activities = data.get("orderActivityCollection") or []
        fill_price = None
        filled_qty = None
        if activities:
            act = activities[-1]
            execs = act.get("executionLegs") or []
            if execs:
                fill_price = execs[-1].get("price")
                filled_qty = execs[-1].get("quantity")

        return OrderStatusResponse(
            broker_order_id=str(data.get("orderId", "")),
            symbol=leg.get("instrument", {}).get("symbol", ""),
            side=leg.get("instruction", ""),
            order_type=data.get("orderType", ""),
            quantity=data.get("quantity", 0),
            status=data.get("status", "").lower(),
            fill_price=fill_price,
            filled_quantity=filled_qty,
            raw=data,
        )
