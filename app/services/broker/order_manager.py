"""
OrderManager — tracks every order from placement to final state.

Maintains an in-process order book. Works with any BaseBroker implementation.
The brain emits events here; the UI reads state from here.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from datetime import datetime, date
from typing import Any, Callable

from app.services.broker.models import Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Thread-safe order registry. Tracks pending → partial → filled lifecycle
    and handles rejection/cancellation with explicit reasons.
    """

    def __init__(self):
        self._orders: dict[str, Order] = {}      # order_id → Order
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[Order], None]] = []
        # PDT tracking: date → set of symbols that completed round-trips
        self._day_trades: dict[date, set[str]] = defaultdict(set)

    # ── Create ────────────────────────────────────────────────────────────────

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        strategy: str = "",
        reason: str = "",
    ) -> Order:
        """Create a new order record in PENDING state."""
        order_id = f"OM-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            order_id=order_id,
            broker_order_id="",          # filled when broker accepts
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy=strategy,
            reason=reason,
        )
        with self._lock:
            self._orders[order_id] = order
        logger.info("Order created: %s %s %s x%.0f @ %s", order_id, side, symbol, qty, order_type)
        return order

    # ── State transitions ─────────────────────────────────────────────────────

    def on_broker_accept(self, order_id: str, broker_order_id: str) -> None:
        """Called when broker acknowledges the order."""
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o.broker_order_id = broker_order_id
        logger.debug("Broker accepted %s → broker_id=%s", order_id, broker_order_id)

    def on_fill(
        self,
        order_id: str,
        filled_qty: float,
        fill_price: float,
        filled_at: datetime | None = None,
    ) -> Order | None:
        """
        Called on every fill event (full or partial).
        Accumulates fills — safe to call multiple times for the same order.
        """
        with self._lock:
            o = self._orders.get(order_id)
            if o is None:
                logger.warning("on_fill: unknown order_id %s", order_id)
                return None

            # Weighted average fill price
            prev_value = o.filled_qty * o.avg_fill_price
            new_value = filled_qty * fill_price
            o.filled_qty += filled_qty
            o.remaining_qty = max(0.0, o.qty - o.filled_qty)
            o.avg_fill_price = (prev_value + new_value) / o.filled_qty if o.filled_qty > 0 else fill_price

            if o.filled_qty >= o.qty - 1e-6:
                o.status = "FILLED"
                o.filled_at = filled_at or datetime.utcnow()
                logger.info(
                    "Order FILLED: %s %s x%.0f @ %.4f",
                    order_id, o.symbol, o.filled_qty, o.avg_fill_price,
                )
                # Track as day trade (round-trip on same symbol same day)
                today = datetime.utcnow().date()
                self._day_trades[today].add(o.symbol)
            else:
                o.status = "PARTIAL"
                logger.warning(
                    "PARTIAL FILL: %s — %.0f of %.0f shares filled @ %.4f. "
                    "Remaining: %.0f",
                    order_id, o.filled_qty, o.qty, fill_price, o.remaining_qty,
                )

        self._emit(o)
        return o

    def on_reject(self, order_id: str, reason: str) -> Order | None:
        """Called when broker rejects the order."""
        rejection_map = {
            "insufficient_buying_power": "Account does not have enough buying power.",
            "market_closed": "Market is closed — cannot place order.",
            "invalid_symbol": "Symbol not found or not tradeable.",
            "pdt_restricted": "PDT rule: account < $25k, day trades exhausted.",
            "risk_check_failed": "Internal risk check rejected this order.",
        }
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o.status = "REJECTED"
                o.rejection_reason = rejection_map.get(reason, reason)
        if o:
            logger.error(
                "Order REJECTED: %s %s — %s",
                order_id, o.symbol, o.rejection_reason,
            )
            self._emit(o)
        return o

    def on_cancel(self, order_id: str) -> Order | None:
        """Called when an order is cancelled."""
        with self._lock:
            o = self._orders.get(order_id)
            if o:
                o.status = "CANCELLED"
                o.cancelled_at = datetime.utcnow()
        if o:
            logger.info("Order CANCELLED: %s %s", order_id, o.symbol)
            self._emit(o)
        return o

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_orders(self, status: str | None = None) -> list[Order]:
        with self._lock:
            orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return sorted(orders, key=lambda o: o.placed_at, reverse=True)

    def get_pending(self) -> list[Order]:
        return self.get_orders("PENDING")

    def get_open(self) -> list[Order]:
        """Pending + partial fills."""
        return [o for o in self.get_orders() if o.status in ("PENDING", "PARTIAL")]

    def get_today_fills(self) -> list[Order]:
        today = datetime.utcnow().date()
        return [
            o for o in self.get_orders("FILLED")
            if o.filled_at and o.filled_at.date() == today
        ]

    def day_trades_today(self) -> int:
        """Count completed day trades (round-trips) today."""
        today = datetime.utcnow().date()
        return len(self._day_trades.get(today, set()))

    def get_summary(self) -> dict[str, Any]:
        orders = self.get_orders()
        status_counts: dict[str, int] = defaultdict(int)
        for o in orders:
            status_counts[o.status] += 1
        today_fills = self.get_today_fills()
        total_pnl = 0.0  # populated if fill prices are tracked
        return {
            "total_orders": len(orders),
            "status_breakdown": dict(status_counts),
            "today_fills": len(today_fills),
            "day_trades_today": self.day_trades_today(),
            "open_orders": len(self.get_open()),
        }

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable[[Order], None]) -> None:
        """Register a function called on every order state change."""
        self._callbacks.append(fn)

    def _emit(self, order: Order) -> None:
        for fn in self._callbacks:
            try:
                fn(order)
            except Exception as e:
                logger.warning("Order callback error: %s", e)


# Module-level singleton
_order_manager = OrderManager()


def get_order_manager() -> OrderManager:
    return _order_manager
