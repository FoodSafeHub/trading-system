"""
AlpacaBroker — Alpaca Markets paper and live trading via their REST API.

Requires alpaca-trade-api or alpaca-py installed:
  pip install alpaca-py

Set env vars:
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_PAPER=true   (default true — use paper endpoint)

This module is optional — if alpaca-py is not installed, importing it
raises ImportError which PaperBroker handles gracefully.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Callable

from app.services.broker.base_broker import BaseBroker
from app.services.broker.models import Account, Order, OrderSide, OrderType, Position
from app.services.broker.order_manager import OrderManager

logger = logging.getLogger(__name__)

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_BASE_URL  = "https://api.alpaca.markets"


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker. Paper trading by default.
    Switch to live by setting ALPACA_PAPER=false.
    """

    def __init__(self):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
        except ImportError as e:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py\n"
                "Or use PaperBroker for offline paper trading."
            ) from e

        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._is_paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required."
            )

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._is_paper,
        )
        self._order_mgr = OrderManager()
        self._callbacks: list[Callable[[Order], None]] = []
        self._MarketOrderRequest = MarketOrderRequest
        self._LimitOrderRequest = LimitOrderRequest
        self._AlpacaSide = AlpacaSide
        self._TimeInForce = TimeInForce
        logger.info(
            "AlpacaBroker initialised (%s)",
            "PAPER" if self._is_paper else "LIVE",
        )

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @property
    def name(self) -> str:
        return f"Alpaca ({'Paper' if self._is_paper else 'Live'})"

    def get_account(self) -> Account:
        acc = self._client.get_account()
        return Account(
            account_id=str(acc.id),
            equity=float(acc.equity),
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            day_trades_used=int(getattr(acc, "daytrade_count", 0)),
            broker="alpaca_paper" if self._is_paper else "alpaca_live",
        )

    def get_positions(self) -> list[Position]:
        positions = self._client.get_all_positions()
        return [self._map_position(p) for p in positions]

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self._client.get_open_position(symbol.upper())
            return self._map_position(p)
        except Exception:
            return None

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
        order_id = f"ALP-{uuid.uuid4().hex[:8].upper()}"
        alpaca_side = (
            self._AlpacaSide.BUY if side == "BUY" else self._AlpacaSide.SELL
        )

        try:
            if order_type == "market":
                req = self._MarketOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=self._TimeInForce.DAY,
                )
            elif order_type == "limit" and limit_price:
                from alpaca.trading.requests import LimitOrderRequest
                req = LimitOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=self._TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                raise ValueError(f"Unsupported order_type: {order_type}")

            resp = self._client.submit_order(req)
            broker_oid = str(resp.id)

            order = Order(
                order_id=order_id,
                broker_order_id=broker_oid,
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price,
                strategy=strategy,
                reason=reason,
                status="PENDING",
                broker_response={"alpaca_id": broker_oid, "status": str(resp.status)},
            )
            logger.info("Alpaca order placed: %s → broker_id=%s", order_id, broker_oid)
            return order

        except Exception as e:
            order = Order(
                order_id=order_id,
                broker_order_id="",
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price,
                strategy=strategy,
                reason=reason,
                status="REJECTED",
                rejection_reason=str(e),
            )
            logger.error("Alpaca order rejected: %s — %s", order_id, e)
            return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order. order_id can be our internal ID or the Alpaca broker_order_id.
        Returns True if cancelled, False if already filled or not found.
        """
        try:
            # Try order_id as Alpaca UUID directly
            self._client.cancel_order_by_id(order_id)
            logger.info("Alpaca order cancelled: %s", order_id)
            return True
        except Exception as e:
            err = str(e).lower()
            if "not found" in err or "422" in err or "404" in err:
                logger.warning("Cancel failed — order not found or already terminal: %s (%s)", order_id, e)
                return False
            if "filled" in err or "complete" in err:
                logger.warning("Cancel failed — order already filled: %s", order_id)
                return False
            logger.error("Alpaca cancel_order error for %s: %s", order_id, e)
            return False

    def modify_order(self, order_id: str, new_qty: float | None = None, new_limit_price: float | None = None) -> Order:
        """
        Modify an existing order's quantity or limit price via Alpaca replace_order.
        Returns updated Order. Raises ValueError if order is not in a modifiable state.
        """
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            replace_req = ReplaceOrderRequest(
                qty=int(new_qty) if new_qty is not None else None,
                limit_price=new_limit_price,
            )
            resp = self._client.replace_order_by_id(order_id, replace_req)
            updated_order = Order(
                order_id=order_id,
                broker_order_id=str(resp.id),
                symbol=str(resp.symbol),
                side="BUY" if str(resp.side).lower() == "buy" else "SELL",
                order_type=str(resp.type).lower(),
                qty=float(resp.qty or new_qty or 0),
                limit_price=new_limit_price,
                status="PENDING",
                broker_response={"alpaca_id": str(resp.id), "status": str(resp.status)},
            )
            logger.info("Alpaca order modified: %s → new_qty=%s new_limit=%s", order_id, new_qty, new_limit_price)
            return updated_order
        except Exception as e:
            err = str(e).lower()
            if "not found" in err or "404" in err:
                raise ValueError(f"Order {order_id} not found on Alpaca") from e
            if "filled" in err or "complete" in err:
                raise ValueError(f"Order {order_id} already filled — cannot modify") from e
            if "422" in err:
                raise ValueError(f"Order {order_id} cannot be modified in its current state: {e}") from e
            raise

    def get_order(self, order_id: str) -> Order | None:
        return None  # would need to map broker_order_id → our order_id

    def get_orders(self, status: str | None = None) -> list[Order]:
        return []

    def close_position(self, symbol: str, reason: str = "") -> Order | None:
        try:
            resp = self._client.close_position(symbol.upper())
            return Order(
                order_id=f"CLOSE-{uuid.uuid4().hex[:6]}",
                broker_order_id=str(resp.id),
                symbol=symbol.upper(),
                side="SELL",
                order_type="market",
                qty=float(resp.qty or 0),
                limit_price=None,
                stop_price=None,
                status="PENDING",
                reason=reason,
            )
        except Exception as e:
            logger.error("Failed to close %s: %s", symbol, e)
            return None

    def close_all_positions(self, reason: str = "EOD") -> list[Order]:
        try:
            responses = self._client.close_all_positions(cancel_orders=True)
            return []  # detailed order tracking not wired yet
        except Exception as e:
            logger.error("close_all_positions failed: %s", e)
            return []

    def register_fill_callback(self, callback: Callable[[Order], None]) -> None:
        self._callbacks.append(callback)

    # ── Mappers ───────────────────────────────────────────────────────────────

    def _map_position(self, p) -> Position:
        return Position(
            symbol=str(p.symbol),
            qty=float(p.qty),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price or p.avg_entry_price),
            unrealized_pnl=float(p.unrealized_pl or 0),
            unrealized_pnl_pct=float(p.unrealized_plpc or 0) * 100,
        )
