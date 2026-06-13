"""Webull OpenAPI trading adapter (US region).

Uses HMAC-SHA1 request signing per the official webull-inc/openapi-python-sdk
recipe. Signing helpers live in :mod:`app.services.brokers.webull_signing` and
are shared with the market-data path.

Auth model: app_key + app_secret signed per-request. No OAuth flow — each
request stands alone, signed against the current UTC timestamp + a nonce.
Token storage methods are present for BrokerBase symmetry with Schwab but
are no-ops on Webull.

US v1 endpoints (confirmed against webullsdktrade==* SDK source):
  GET  /app/subscriptions/list             — list brokerage subscriptions (US accounts)
  GET  /account/profile                    — static account profile
  GET  /account/balance                    — balance + buying power (USD)
  GET  /account/positions                  — open positions (paged)
  POST /trade/order/place                  — submit order (US v1 schema)
  POST /trade/order/cancel                 — cancel by client_order_id
  GET  /trade/order/detail                 — single order status
  GET  /trade/orders/list-today            — today's order history
  GET  /trade/security                     — symbol → instrument_id lookup

The v2 endpoints under /openapi/account/* are NOT available for US accounts
(HK/JP only per the SDK docstrings). Do not switch to them.

Quotes use the market-data signed endpoint at /openapi/quote/v1/quote/last-price
(separate host quotes-api.webull.com — handled by webull_md.py).

If any endpoint returns a 4xx the adapter raises so the execution layer can
log and surface the failure. We never silently succeed on a failed order.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase
from app.services.brokers.webull_signing import build_signed_headers, json_body_bytes

logger = logging.getLogger(__name__)

WEBULL_HOST = "api.webull.com"
WEBULL_BASE_URL = f"https://{WEBULL_HOST}"

# Order-side enum used by Webull's /trade/order/place stock_order.side.
_SIDE_MAP = {"BUY": "BUY", "SELL": "SELL"}
# OrderType maps our enum to Webull's stock_order.order_type strings.
# (Confirmed against webullsdktrade.common.order_type.OrderType.)
_ORDER_TYPE_MAP = {
    "MARKET":        "MARKET",
    "LIMIT":         "LIMIT",
    "STOP":          "STOP_LOSS",
    "STOP_LIMIT":    "STOP_LOSS_LIMIT",
    "TRAILING_STOP": "TRAILING_STOP_LOSS",
}
# Our trail_type → Webull trailing_type (webullsdktrade.common.trailing_type).
_TRAILING_TYPE_MAP = {
    "PERCENT": "PERCENTAGE",
    "DOLLAR":  "AMOUNT",
}
# Reverse: normalize Webull's order_type strings back to OUR enums on read-back
# so the scheduler's "find resting SELL stops" and the PnL trail audit (which
# filter on order_type in STOP/TRAILING_STOP) recognize Webull orders.
_ORDER_TYPE_REVERSE = {
    "MARKET":             "MARKET",
    "LIMIT":              "LIMIT",
    "STOP_LOSS":          "STOP",
    "STOP_LOSS_LIMIT":    "STOP_LIMIT",
    "TRAILING_STOP_LOSS": "TRAILING_STOP",
}
# Time-in-force passthrough — Webull names align with ours.
_TIF_MAP = {"DAY": "DAY", "GTC": "GTC", "IOC": "IOC", "FOK": "FOK"}


class WebullBroker(BrokerBase):
    """Webull OpenAPI broker adapter (signed HMAC-SHA1, US region)."""

    name = "webull"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._app_key = self._settings.webull_app_key
        self._app_secret = self._settings.webull_app_secret
        # Webull's request signature is per-call. No access token to cache.
        self._account_id: str = self._settings.webull_account_id
        # symbol → instrument_id lookups are stable; cache for the process lifetime.
        self._instrument_cache: Dict[str, str] = {}

    # ── Auth (signing model — no token exchange) ────────────────────────────

    async def authenticate(self) -> None:
        """Validate credentials are present. Signing happens per-request."""
        if not self._app_key or not self._app_secret:
            logger.warning(
                "[webull] No app_key/app_secret configured. "
                "Set WEBULL_APP_KEY + WEBULL_APP_SECRET in .env."
            )
            return
        logger.info("[webull] Credentials loaded (signing model — no token exchange).")

    async def refresh_token_if_needed(self) -> None:
        """No-op — Webull signs each request, there is no bearer token."""
        return

    # ── Shared HTTP helpers ─────────────────────────────────────────────────

    def _require_creds(self) -> None:
        if not self._app_key or not self._app_secret:
            raise RuntimeError(
                "Webull credentials missing. Configure WEBULL_APP_KEY + "
                "WEBULL_APP_SECRET in .env."
            )

    async def _get(self, uri: str, params: Optional[dict] = None) -> Any:
        self._require_creds()
        q = {k: str(v) for k, v in (params or {}).items() if v is not None}
        headers = build_signed_headers(
            app_key=self._app_key, app_secret=self._app_secret,
            host=WEBULL_HOST, uri=uri, query=q,
        )
        async with httpx.AsyncClient(base_url=WEBULL_BASE_URL, timeout=15) as client:
            resp = await client.get(uri, headers=headers, params=q)
        self._raise_for_status(resp, uri)
        return self._unwrap(resp.json())

    async def _post(self, uri: str, body: dict, query: Optional[dict] = None) -> Any:
        self._require_creds()
        q = {k: str(v) for k, v in (query or {}).items() if v is not None}
        headers = build_signed_headers(
            app_key=self._app_key, app_secret=self._app_secret,
            host=WEBULL_HOST, uri=uri, query=q, body=body,
        )
        headers["Content-Type"] = "application/json"
        content = json_body_bytes(body)
        async with httpx.AsyncClient(base_url=WEBULL_BASE_URL, timeout=15) as client:
            resp = await client.post(uri, headers=headers, params=q, content=content)
        self._raise_for_status(resp, uri)
        return self._unwrap(resp.json())

    @staticmethod
    def _raise_for_status(resp: httpx.Response, uri: str) -> None:
        if resp.status_code >= 400:
            body = resp.text[:500]
            logger.error("[webull] HTTP %s on %s — %s", resp.status_code, uri, body)
            resp.raise_for_status()

    @staticmethod
    def _unwrap(payload: Any) -> Any:
        """Webull error bodies are typically {"code":"...", "msg":"..."}.

        Successful responses are returned directly (not always wrapped). If a
        non-success code arrives with HTTP 200 (rare), surface as a RuntimeError
        so callers don't silently treat it as success.
        """
        if isinstance(payload, dict) and "code" in payload and "data" not in payload:
            code = str(payload.get("code"))
            if code not in ("200", "0", "success"):
                raise RuntimeError(
                    f"Webull API error code={code} msg={payload.get('msg') or payload}"
                )
        if isinstance(payload, dict) and "data" in payload and "code" in payload:
            code = str(payload.get("code"))
            if code not in ("200", "0", "success"):
                raise RuntimeError(
                    f"Webull API error code={code} msg={payload.get('msg') or payload}"
                )
            return payload.get("data")
        return payload

    # ── Accounts / Positions / Quotes ───────────────────────────────────────

    async def get_accounts(self) -> List[AccountSummary]:
        """List US brokerage accounts via /app/subscriptions/list.

        Webull's US OpenAPI returns "subscriptions" (broker links) rather than
        a flat account list — each subscription carries an account_id and
        brokerage info. We enrich with /account/balance per subscription so
        the dashboard sees buying_power and cash.
        """
        subs = await self._get("/app/subscriptions/list")
        # Response is a list of subscriptions, each with:
        #   account_id      — Webull's internal id (used for all subsequent calls)
        #   account_number  — the human-facing brokerage account number (e.g. 5MS15206)
        #   subscription_id, user_id
        rows = subs if isinstance(subs, list) else (subs or {}).get("subscriptions") or []
        out: List[AccountSummary] = []
        for s in rows:
            acct_id = str(s.get("account_id") or s.get("accountId") or "")
            if not acct_id:
                continue
            bp = cash = equity = None
            try:
                bal = await self._get(
                    "/account/balance",
                    params={"account_id": acct_id, "total_asset_currency": "USD"},
                )
                # Webull's US balance puts the headline numbers at top level
                # plus a nested per-currency block. Prefer the USD nested block
                # since it carries net_liquidation_value and cash_power.
                usd_block: dict = {}
                for blk in (bal.get("account_currency_assets") or []):
                    if str(blk.get("currency", "")).upper() == "USD":
                        usd_block = blk
                        break
                cash   = _to_float(usd_block.get("cash_balance") or bal.get("total_cash_balance"))
                equity = _to_float(usd_block.get("net_liquidation_value"))
                # Webull doesn't ship a single "buying_power" field; cash_power
                # is the dollar amount available to spend on new orders today.
                bp     = _to_float(usd_block.get("cash_power") or usd_block.get("margin_power"))
            except Exception as exc:
                logger.warning("[webull] balance fetch failed for %s: %s", acct_id, exc)
            out.append(AccountSummary(
                broker="webull",
                account_id=acct_id,
                account_type=s.get("account_type") or s.get("accountType"),
                buying_power=bp, cash=cash, equity=equity,
                is_paper=False,
            ))
        return out

    async def get_positions(self, account_id: str) -> List[Position]:
        acct = account_id or self._account_id
        if not acct:
            raise RuntimeError("Webull get_positions requires an account_id")
        data = await self._get(
            "/account/positions",
            params={"account_id": acct, "page_size": 100},
        )
        # Webull's US response wraps positions in "holdings".
        rows = (
            data if isinstance(data, list)
            else (data or {}).get("holdings")
                 or (data or {}).get("positions")
                 or (data or {}).get("items") or []
        )
        out: List[Position] = []
        for p in rows:
            qty = _to_float(p.get("qty") or p.get("quantity") or p.get("position")) or 0.0
            avg = _to_float(p.get("unit_cost") or p.get("avg_price") or p.get("avgPrice") or p.get("cost_price"))
            last = _to_float(p.get("last_price") or p.get("lastPrice") or p.get("marketPrice") or p.get("current_price"))
            mv = _to_float(p.get("market_value") or p.get("marketValue"))
            if mv is None and qty and last is not None:
                mv = qty * last
            upl = _to_float(
                p.get("unrealized_profit_loss") or p.get("unrealizedProfitLoss")
                or p.get("unrealized_pnl") or p.get("unrealizedPnl")
            )
            sym = p.get("symbol") or p.get("ticker") or ""
            if not sym:
                continue
            out.append(Position(
                symbol=str(sym).upper(),
                quantity=qty,
                average_cost=avg,
                current_price=last,
                market_value=mv,
                unrealized_pnl=upl,
                broker="webull",
                account_id=acct,
            ))
        return out

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        # The signed quote endpoint lives on the market-data host; the scanner
        # uses webull_md.py for that path. The trading adapter doesn't need
        # to duplicate that — return empty so callers fall back to Schwab/MD.
        logger.debug("[webull] get_quotes: defer to market-data path (no-op here)")
        return {}

    # ── Symbol → instrument_id lookup ───────────────────────────────────────

    async def _resolve_instrument_id(self, symbol: str) -> str:
        """Resolve a US equity symbol to Webull's internal instrument_id.

        Webull's signed /trade/security endpoint is not enabled for the basic
        US OpenAPI tier (returns 404). The public ticker-search endpoint at
        quotes-gw.webullbroker.com returns the same tickerId Webull uses as
        instrument_id internally (verified against the holdings response —
        DRD → 913323252 matches in both places). It's also what webull_md.py
        already relies on for market data.
        """
        sym = symbol.upper()
        cached = self._instrument_cache.get(sym)
        if cached:
            return cached
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://quotes-gw.webullbroker.com/api/search/pc/tickers",
                params={"keyword": sym, "pageIndex": 1, "pageSize": 5, "regionId": 6},
            )
        resp.raise_for_status()
        rows = (resp.json() or {}).get("data") or []
        for row in rows:
            if str(row.get("symbol", "")).upper() == sym:
                inst_id = str(row.get("tickerId") or "")
                if inst_id:
                    self._instrument_cache[sym] = inst_id
                    return inst_id
        raise RuntimeError(f"Webull ticker search returned no instrument_id for {sym}")

    # ── Orders ──────────────────────────────────────────────────────────────

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        # Webull's US OpenAPI does not expose a public order-preview endpoint.
        # ExecutionService treats NotImplementedError as "skip preview" — same
        # contract as Schwab.
        raise NotImplementedError(
            "Webull OpenAPI does not currently expose a public order preview endpoint for US."
        )

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        acct = account_id or self._account_id
        if not acct:
            raise RuntimeError(
                "Webull place_order requires an account_id. Set WEBULL_ACCOUNT_ID in .env "
                "or pass account_id explicitly."
            )
        instrument_id = await self._resolve_instrument_id(order.symbol)
        payload = self._build_order_payload(order, acct, instrument_id)
        data = await self._post("/trade/order/place", body=payload)
        order_id = (
            (data or {}).get("client_order_id")
            or (data or {}).get("order_id")
            or (data or {}).get("orderId")
            or (data or {}).get("id")
        )
        logger.info(
            "[webull] Order submitted: %s %s qty=%s type=%s id=%s",
            order.side, order.symbol, order.quantity, order.order_type, order_id,
        )
        return OrderStatusResponse(
            broker_order_id=str(order_id) if order_id is not None else None,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            status="submitted",
            raw=data if isinstance(data, dict) else {"data": data},
        )

    def _build_order_payload(self, order: OrderRequest, account_id: str, instrument_id: str) -> dict:
        side = _SIDE_MAP.get(order.side)
        otype = _ORDER_TYPE_MAP.get(order.order_type)
        tif = _TIF_MAP.get(order.time_in_force, "DAY")
        if side is None or otype is None:
            raise ValueError(
                f"Webull does not support side={order.side!r} order_type={order.order_type!r}"
            )
        # MARKET orders must have extended_hours_trading=false per the SDK docstring.
        ext_hours = False if otype == "MARKET" else False
        stock_order: dict = {
            "client_order_id": order.idempotency_key or "",
            "instrument_id":   instrument_id,
            "qty":             str(int(order.quantity)) if order.quantity == int(order.quantity) else str(order.quantity),
            "side":            side,
            "tif":             tif,
            "extended_hours_trading": ext_hours,
            "order_type":      otype,
        }
        if order.order_type in ("LIMIT", "STOP_LIMIT") and order.limit_price is not None:
            stock_order["limit_price"] = str(order.limit_price)
        if order.order_type in ("STOP", "STOP_LIMIT") and order.stop_price is not None:
            stock_order["stop_price"] = str(order.stop_price)
        if order.order_type == "TRAILING_STOP":
            # Webull TRAILING_STOP_LOSS needs trailing_type + trailing_stop_step
            # (the trail distance: a % when PERCENTAGE, a $ amount when AMOUNT).
            t_type = _TRAILING_TYPE_MAP.get(order.trail_type or "PERCENT", "PERCENTAGE")
            stock_order["trailing_type"] = t_type
            if order.trail_value is not None:
                stock_order["trailing_stop_step"] = str(order.trail_value)
        return {
            "account_id":  account_id,
            "category":    "US_STOCK",
            "stock_order": stock_order,
        }

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        acct = account_id or self._account_id
        try:
            await self._post(
                "/trade/order/cancel",
                body={"account_id": acct, "client_order_id": broker_order_id},
            )
            logger.info("[webull] Order cancelled: %s", broker_order_id)
            return True
        except Exception as exc:
            logger.error("[webull] Cancel failed for %s: %s", broker_order_id, exc)
            return False

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        acct = account_id or self._account_id
        data = await self._get(
            "/trade/order/detail",
            params={"account_id": acct, "client_order_id": broker_order_id},
        )
        row = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
        return self._parse_order_response(self._flatten_order(row))

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        acct = account_id or self._account_id
        params: dict = {"account_id": acct, "page_size": 100}
        data = await self._get("/trade/orders/list-today", params=params)
        rows = data if isinstance(data, list) else (data or {}).get("orders") or (data or {}).get("items") or []
        parsed = [self._parse_order_response(self._flatten_order(r)) for r in rows]
        if status:
            parsed = [p for p in parsed if p.status == status.lower()]
        return parsed

    @staticmethod
    def _flatten_order(row: dict) -> dict:
        """Merge Webull's nested order wrapper with its executed leg.

        The US list-today / order-detail response wraps each order as::

            {"client_order_id": ..., "order_id": ..., "tif": ...,
             "items": [ {"symbol": ..., "filled_price": ..., "filled_qty": ...,
                         "order_status": "FILLED", "side": ..., "qty": ...} ]}

        The order-level identity fields live on the wrapper; the executed
        symbol/fill/status fields live inside items[]. We merge the first item
        over the wrapper so a single flat dict carries everything the parser
        needs. (Combo/multi-leg orders are rare for plain stock trades; we take
        the first leg, which is the equity fill.)
        """
        if not isinstance(row, dict):
            return {}
        items = row.get("items")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            merged = dict(row)
            merged.update(items[0])  # item fields win over the wrapper
            return merged
        return row

    def _parse_order_response(self, data: dict) -> OrderStatusResponse:
        return OrderStatusResponse(
            broker_order_id=str(
                # Prefer Webull's server-side order_id over the echoed
                # client_order_id so reconciliation stores a stable id.
                data.get("order_id")
                or data.get("orderId")
                or data.get("client_order_id")
                or data.get("id")
                or ""
            ),
            symbol=str(data.get("symbol") or "").upper(),
            side=str(data.get("side") or ""),
            order_type=_ORDER_TYPE_REVERSE.get(
                str(data.get("order_type") or data.get("orderType") or "").upper(),
                str(data.get("order_type") or data.get("orderType") or ""),
            ),
            quantity=_to_float(
                data.get("qty") or data.get("quantity") or data.get("totalQuantity")
            ) or 0.0,
            status=str(
                data.get("order_status")
                or data.get("status")
                or data.get("orderStatus")
                or ""
            ).lower(),
            fill_price=_to_float(
                data.get("filled_price")
                or data.get("avg_filled_price")
                or data.get("avgFilledPrice")
                or data.get("filledPrice")
            ),
            filled_quantity=_to_float(
                data.get("filled_qty") or data.get("filledQuantity")
            ),
            raw=data,
        )


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None
