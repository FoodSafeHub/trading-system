from __future__ import annotations

"""
Zerodha Kite Connect adapter (India — NSE/BSE).

Reference: https://kite.trade/docs/connect/v3/

Auth is NOT OAuth2. It is a 3-step login flow:
  1. Send the user to https://kite.zerodha.com/connect/login?api_key=...&v=3
  2. Kite redirects back to our redirect_uri with ?request_token=...&status=success
  3. We exchange it at POST /session/token with
       checksum = sha256(api_key + request_token + api_secret)
     and receive an access_token.

The access_token is single-day: it expires at ~07:30 IST every morning and
there is NO refresh endpoint (a SEBI requirement). refresh_token_if_needed()
is therefore a no-op that only logs when the token looks stale — the user must
re-run the login flow daily. We persist the token in the shared broker_tokens
table (refresh_token left null, token_expiry = next 07:30 IST).

Every authenticated request carries:
  Authorization: token <api_key>:<access_token>
  X-Kite-Version: 3

Symbols stay ticker-only across the rest of the system (e.g. "RELIANCE").
Translation to Kite's exchange + tradingsymbol happens HERE and nowhere else.
Default exchange is NSE; pass "BSE:RELIANCE" style only if a caller needs it.
"""

import hashlib
import logging
import urllib.parse
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)

KITE_BASE_URL = "https://api.kite.trade"
KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login"
KITE_VERSION = "3"
DEFAULT_EXCHANGE = "NSE"

# Kite Connect product types:
#   CNC = delivery (no leverage, no forced square-off) — our default for swing
#   MIS = intraday (leverage, auto square-off ~15:20 IST)
DEFAULT_PRODUCT = "CNC"


class ZerodhaBroker(BrokerBase):
    name = "zerodha"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._access_token: str = ""
        self._token_expiry: Optional[datetime] = None

    # ── Auth ────────────────────────────────────────────────────────────────

    def get_login_url(self) -> str:
        """Step 1: the URL the user opens to log in to Zerodha and authorize."""
        params = {"api_key": self._settings.zerodha_api_key, "v": KITE_VERSION}
        return f"{KITE_LOGIN_URL}?{urllib.parse.urlencode(params)}"

    def _checksum(self, request_token: str) -> str:
        raw = f"{self._settings.zerodha_api_key}{request_token}{self._settings.zerodha_api_secret}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def exchange_request_token(self, request_token: str) -> None:
        """Step 3: trade the request_token for a same-day access_token."""
        async with httpx.AsyncClient(base_url=KITE_BASE_URL) as client:
            resp = await client.post(
                "/session/token",
                headers={"X-Kite-Version": KITE_VERSION},
                data={
                    "api_key": self._settings.zerodha_api_key,
                    "request_token": request_token,
                    "checksum": self._checksum(request_token),
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            self._access_token = data["access_token"]
            self._token_expiry = self._next_expiry()
            await self._save_token_to_db()
            logger.info("[zerodha] Session established; token valid until %s", self._token_expiry)

    @staticmethod
    def _next_expiry() -> datetime:
        """Kite tokens die at ~07:30 IST. Compute the next such instant in UTC.

        Stored as UTC to match the rest of the system's storage convention.
        """
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(tz=ist)
        expiry = now.replace(hour=7, minute=30, second=0, microsecond=0)
        if now >= expiry:
            expiry = expiry + timedelta(days=1)
        return expiry.astimezone(ZoneInfo("UTC"))

    async def authenticate(self) -> None:
        """Load today's token from DB, or fall back to the .env bootstrap value."""
        if not await self._load_token_from_db():
            self._access_token = self._settings.zerodha_access_token

        if not self._access_token:
            logger.warning(
                "[zerodha] No access token. Run the login flow: GET /zerodha/login, "
                "open the URL, and Kite will redirect to /zerodha/callback."
            )
            return
        await self.refresh_token_if_needed()

    async def refresh_token_if_needed(self) -> None:
        """No-op by design: Kite has no refresh API. Just warn if stale.

        Kept to satisfy BrokerBase. When the token is past its 07:30 IST expiry,
        the next API call will 403 (TokenException) and the execution layer
        surfaces that as a re-login prompt.
        """
        from datetime import timezone

        if self._token_expiry is None:
            return
        expiry = self._token_expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if datetime.now(tz=timezone.utc) >= expiry:
            logger.warning(
                "[zerodha] Access token expired at %s IST cutoff. "
                "Re-login required: GET /zerodha/login (no auto-refresh — SEBI rule).",
                expiry,
            )

    async def _load_token_from_db(self) -> bool:
        try:
            from app.db import SessionLocal
            from app.models.broker_tokens import BrokerToken

            with SessionLocal() as db:
                row = db.query(BrokerToken).filter_by(broker="zerodha").first()
                if row:
                    self._access_token = row.access_token
                    self._token_expiry = row.token_expiry
                    return True
        except Exception as exc:
            logger.warning("[zerodha] Could not load token from DB: %s", exc)
        return False

    async def _save_token_to_db(self) -> None:
        try:
            from app.db import SessionLocal
            from app.models.broker_tokens import BrokerToken

            with SessionLocal() as db:
                row = db.query(BrokerToken).filter_by(broker="zerodha").first()
                if not row:
                    row = BrokerToken(broker="zerodha")
                    db.add(row)
                row.access_token = self._access_token
                row.refresh_token = None  # Kite has none
                row.token_expiry = self._token_expiry
                db.commit()
        except Exception as exc:
            logger.error("[zerodha] Failed to save token to DB: %s", exc)

    # ── Symbol translation ───────────────────────────────────────────────────

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        """'RELIANCE' -> ('NSE', 'RELIANCE'); 'BSE:TCS' -> ('BSE', 'TCS')."""
        sym = symbol.upper().strip()
        if ":" in sym:
            exch, ts = sym.split(":", 1)
            return exch, ts
        return DEFAULT_EXCHANGE, sym

    @classmethod
    def _kite_instrument(cls, symbol: str) -> str:
        exch, ts = cls._split_symbol(symbol)
        return f"{exch}:{ts}"

    # ── Shared HTTP helpers ────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self._settings.zerodha_api_key}:{self._access_token}",
            "X-Kite-Version": KITE_VERSION,
        }

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=KITE_BASE_URL) as client:
            resp = await client.get(path, headers=self._auth_headers(), params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})

    async def _post_form(self, path: str, data: dict) -> dict:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=KITE_BASE_URL) as client:
            resp = await client.post(path, headers=self._auth_headers(), data=data, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})

    async def _delete(self, path: str) -> dict:
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=KITE_BASE_URL) as client:
            resp = await client.delete(path, headers=self._auth_headers(), timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})

    # ── BrokerBase implementation ────────────────────────────────────────────

    async def get_accounts(self) -> List[AccountSummary]:
        # Kite returns one user profile + a separate margins call for funds.
        profile = await self._get("/user/profile")
        margins = await self._get("/user/margins")
        equity = (margins or {}).get("equity", {})
        available = (equity.get("available") or {})
        return [
            AccountSummary(
                broker="zerodha",
                account_id=profile.get("user_id", ""),
                account_type=profile.get("user_type"),
                buying_power=available.get("live_balance"),
                cash=available.get("cash"),
                equity=(equity.get("net")),
                is_paper=False,
            )
        ]

    async def get_positions(self, account_id: str) -> List[Position]:
        # Holdings = delivery (CNC) overnight; positions = intraday/net.
        holdings = await self._get("/portfolio/holdings") or []
        out: List[Position] = []
        for h in holdings:
            qty = h.get("quantity", 0) or 0
            # Skip flat lots — a fully-sold holding can remain in the array with
            # quantity 0 and would otherwise show as an "open" position.
            if qty == 0:
                continue
            out.append(
                Position(
                    symbol=h.get("tradingsymbol", ""),
                    quantity=qty,
                    average_cost=h.get("average_price"),
                    current_price=h.get("last_price"),
                    market_value=(h.get("last_price") or 0) * qty,
                    unrealized_pnl=h.get("pnl"),
                    broker="zerodha",
                    account_id=account_id,
                )
            )
        return out

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        # Prefer the Upstox data feed when configured — Zerodha's own quote API
        # needs a paid data add-on, while Upstox data is free. Orders still run
        # through Zerodha; only the price source is cross-broker.
        try:
            from app.services.marketdata import upstox_data
            if upstox_data.is_configured():
                ltp = upstox_data.get_ltp(symbols)
                if any(v is not None for v in ltp.values()):
                    return {
                        s.upper().strip(): Quote(symbol=s.upper().strip(), last=ltp.get(s.upper().strip()))
                        for s in symbols
                    }
        except Exception as exc:
            logger.debug("[zerodha] Upstox quote delegation failed, falling back: %s", exc)

        instruments = [self._kite_instrument(s) for s in symbols]
        params = [("i", inst) for inst in instruments]
        await self.refresh_token_if_needed()
        async with httpx.AsyncClient(base_url=KITE_BASE_URL) as client:
            resp = await client.get(
                "/quote", headers=self._auth_headers(), params=params, timeout=15
            )
            # The live-quote feed is a separate paid Kite subscription. Without it
            # /quote returns 403 PermissionException. Don't crash the caller —
            # return empty quotes so order placement (which doesn't need this
            # feed) and the rest of the UI keep working.
            if resp.status_code == 403:
                logger.warning(
                    "[zerodha] Quote feed returned 403 (PermissionException). "
                    "Live market-data subscription is not active on this Kite app. "
                    "Orders/portfolio still work; quotes are unavailable."
                )
                return {s.upper().strip(): Quote(symbol=s.upper().strip()) for s in symbols}
            resp.raise_for_status()
            data = resp.json().get("data", {})
        # Kite keys responses by "EXCH:TRADINGSYMBOL"; map back to the bare ticker.
        quotes: Dict[str, Quote] = {}
        for orig in symbols:
            inst = self._kite_instrument(orig)
            info = data.get(inst, {})
            depth = info.get("depth", {})
            bid = (depth.get("buy") or [{}])[0].get("price")
            ask = (depth.get("sell") or [{}])[0].get("price")
            quotes[orig.upper().strip()] = Quote(
                symbol=orig.upper().strip(),
                bid=bid,
                ask=ask,
                last=info.get("last_price"),
                volume=info.get("volume"),
                timestamp=str(info.get("timestamp", "")),
            )
        return quotes

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        raise NotImplementedError(
            "Zerodha has no order-preview endpoint. The execution service skips preview "
            "for Zerodha (brokerage can be estimated via /charges, not wired yet)."
        )

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        # Kite Connect has NO native trailing-stop order type (Zerodha removed
        # trailing SL years ago). Reject it loudly rather than letting the
        # order-type map silently fall through to MARKET — a MARKET SELL here
        # would instantly liquidate the position a protective stop was meant to
        # guard. The execution layer checks supports_native_trailing_stop
        # (False for this broker) and places a static STOP instead, so this
        # guard should never fire in normal flow; it's the last-line safety net.
        if order.order_type == "TRAILING_STOP":
            raise ValueError(
                "Zerodha/Kite has no native TRAILING_STOP order type. "
                "Place a static STOP (SL-M) instead; the Chandelier job ratchets it. "
                f"(symbol={order.symbol}, side={order.side})"
            )

        exch, ts = self._split_symbol(order.symbol)
        body = {
            "exchange": exch,
            "tradingsymbol": ts,
            "transaction_type": order.side,            # BUY | SELL
            "order_type": self._kite_order_type(order.order_type),
            "quantity": int(order.quantity),
            "product": DEFAULT_PRODUCT,
            "validity": "DAY",
        }
        if order.order_type in ("LIMIT", "STOP_LIMIT") and order.limit_price:
            body["price"] = order.limit_price
        if order.order_type in ("STOP", "STOP_LIMIT") and order.stop_price:
            body["trigger_price"] = order.stop_price

        data = await self._post_form("/orders/regular", body)
        broker_order_id = data.get("order_id")
        logger.info(
            "[zerodha] Order submitted: %s %s x%s — order_id=%s",
            order.side, ts, body["quantity"], broker_order_id,
        )
        return OrderStatusResponse(
            broker_order_id=str(broker_order_id) if broker_order_id else None,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            status="submitted",
            raw=data,
        )

    @staticmethod
    def _kite_order_type(order_type: str) -> str:
        """Map our order_type vocabulary to Kite's.

        Unknown types raise rather than silently defaulting to MARKET — a wrong
        default here can turn a protective order into an instant liquidation.
        """
        mapping = {
            "MARKET": "MARKET",
            "LIMIT": "LIMIT",
            "STOP": "SL-M",        # stop-loss market
            "STOP_LIMIT": "SL",    # stop-loss limit
        }
        try:
            return mapping[order_type]
        except KeyError:
            raise ValueError(
                f"Zerodha/Kite does not support order_type={order_type!r}. "
                f"Supported: {sorted(mapping)}"
            )

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        try:
            await self._delete(f"/orders/regular/{broker_order_id}")
            logger.info("[zerodha] Order cancelled: %s", broker_order_id)
            return True
        except httpx.HTTPStatusError as exc:
            logger.error("[zerodha] Cancel failed: %s", exc)
            return False

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        # Kite returns the full status history; the last entry is the latest state.
        history = await self._get(f"/orders/{broker_order_id}") or []
        latest = history[-1] if history else {}
        return self._parse_order(latest)

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        orders = await self._get("/orders") or []
        parsed = [self._parse_order(o) for o in orders]
        if status:
            parsed = [o for o in parsed if o.status == status.lower()]
        return parsed

    def _parse_order(self, data: dict) -> OrderStatusResponse:
        return OrderStatusResponse(
            broker_order_id=str(data.get("order_id", "")),
            symbol=data.get("tradingsymbol", ""),
            side=data.get("transaction_type", ""),
            order_type=data.get("order_type", ""),
            quantity=data.get("quantity", 0),
            status=str(data.get("status", "")).lower(),
            fill_price=data.get("average_price") or None,
            filled_quantity=data.get("filled_quantity"),
            raw=data,
        )
