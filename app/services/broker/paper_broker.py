"""
PaperBroker — in-process paper trading implementation of BaseBroker.

Simulates fills using the last known price from yfinance.
No network calls to a brokerage — fully self-contained for testing.
Can be replaced with AlpacaBroker or IBKRBroker without changing any
calling code (same BaseBroker interface).
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from typing import Callable

import yfinance as yf

from app.services.broker.base_broker import BaseBroker
from app.services.broker.models import Account, Order, OrderSide, OrderType, Position
from app.services.broker.order_manager import OrderManager
from app.services.strategy.daytrading.execution.fill_simulator import FillConfig, FillSimulator

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """
    Simulates a broker in memory.
    Fills are simulated at last bid/ask midpoint + slippage model.
    State resets on process restart (no persistence).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fill_config: FillConfig | None = None,
    ):
        self._capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._lock = threading.Lock()
        self._order_mgr = OrderManager()
        self._fill_sim = FillSimulator(fill_config or FillConfig())
        self._callbacks: list[Callable[[Order], None]] = []
        self._realized_pnl_today = 0.0
        self._day_trades_used = 0

    # ── BaseBroker implementation ─────────────────────────────────────────────

    @property
    def is_paper(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "PaperBroker"

    def get_account(self) -> Account:
        with self._lock:
            unrealized = sum(p.unrealized_pnl for p in self._positions.values())
            equity = self._cash + sum(p.market_value for p in self._positions.values())
        return Account(
            account_id="PAPER-001",
            equity=round(equity, 2),
            cash=round(self._cash, 2),
            buying_power=round(self._cash * 4, 2),   # 4× intraday margin
            day_trades_used=self._day_trades_used,
            initial_capital=self._capital,
            unrealized_pnl=round(unrealized, 2),
            realized_pnl_today=round(self._realized_pnl_today, 2),
            broker="paper",
        )

    def get_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol.upper())

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
        sym = symbol.upper()
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        order = Order(
            order_id=order_id,
            broker_order_id=order_id,
            symbol=sym,
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

        # Validate buying power before filling
        price = self._get_last_price(sym)
        order_value = qty * price
        account = self.get_account()

        if side == "BUY" and order_value > account.buying_power:
            self._order_mgr.on_reject(order_id, "insufficient_buying_power")
            order.status = "REJECTED"
            order.rejection_reason = f"Need ${order_value:,.0f}, have ${account.buying_power:,.0f} buying power."
            logger.error("Order rejected: %s — insufficient buying power", order_id)
            self._emit(order)
            return order

        if account.is_pdt_restricted:
            self._order_mgr.on_reject(order_id, "pdt_restricted")
            order.status = "REJECTED"
            order.rejection_reason = account.pdt_warning or "PDT restricted."
            self._emit(order)
            return order

        # Simulate immediate fill (paper trading = instant at market)
        self._simulate_fill(order, price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            o = self._orders.get(order_id)
            if o and o.status in ("PENDING", "PARTIAL"):
                o.status = "CANCELLED"
                o.cancelled_at = datetime.utcnow()
                logger.info("Paper order cancelled: %s", order_id)
                self._emit(o)
                return True
        return False

    def modify_order(
        self,
        order_id: str,
        new_qty: float | None = None,
        new_limit_price: float | None = None,
    ) -> Order:
        with self._lock:
            o = self._orders.get(order_id)
            if o and o.status in ("PENDING", "PARTIAL"):
                if new_qty is not None:
                    o.qty = new_qty
                    o.remaining_qty = new_qty - o.filled_qty
                if new_limit_price is not None:
                    o.limit_price = new_limit_price
        return o or Order(
            order_id=order_id, broker_order_id="", symbol="", side="BUY",
            order_type="market", qty=0, limit_price=None, stop_price=None,
            status="REJECTED", rejection_reason="Order not found.",
        )

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_orders(self, status: str | None = None) -> list[Order]:
        with self._lock:
            orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return sorted(orders, key=lambda o: o.placed_at, reverse=True)

    def close_position(self, symbol: str, reason: str = "") -> Order | None:
        sym = symbol.upper()
        pos = self.get_position(sym)
        if not pos:
            return None
        side: OrderSide = "SELL" if pos.qty > 0 else "BUY"
        return self.place_order(sym, abs(pos.qty), side, reason=reason or "close_position")

    def close_all_positions(self, reason: str = "EOD") -> list[Order]:
        orders = []
        for sym in list(self._positions.keys()):
            o = self.close_position(sym, reason=reason)
            if o:
                orders.append(o)
        return orders

    def register_fill_callback(self, callback: Callable[[Order], None]) -> None:
        self._callbacks.append(callback)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _simulate_fill(self, order: Order, price: float) -> None:
        """Simulate immediate market fill with slippage."""
        fill = self._fill_sim.fill_entry(order.side, price, order.qty)
        fill_price = fill.fill_price

        order.avg_fill_price = fill_price
        order.filled_qty = order.qty
        order.remaining_qty = 0.0
        order.status = "FILLED"
        order.filled_at = datetime.utcnow()

        sym = order.symbol
        with self._lock:
            if order.side == "BUY":
                cost = fill_price * order.qty + fill.total_cost
                self._cash -= cost
                existing = self._positions.get(sym)
                if existing:
                    # Average up
                    total_qty = existing.qty + order.qty
                    existing.avg_entry_price = (
                        (existing.avg_entry_price * existing.qty + fill_price * order.qty)
                        / total_qty
                    )
                    existing.qty = total_qty
                else:
                    self._positions[sym] = Position(
                        symbol=sym,
                        qty=order.qty,
                        avg_entry_price=fill_price,
                        current_price=fill_price,
                        strategy=order.strategy,
                    )
            else:  # SELL (close long or open short)
                pos = self._positions.get(sym)
                if pos and pos.qty > 0:
                    proceeds = fill_price * order.qty - fill.total_cost
                    realized = (fill_price - pos.avg_entry_price) * order.qty - fill.total_cost
                    self._realized_pnl_today += realized
                    self._cash += proceeds
                    pos.qty -= order.qty
                    if pos.qty <= 1e-6:
                        del self._positions[sym]
                        self._day_trades_used += 1

        logger.info(
            "Paper FILL: %s %s x%.0f @ %.4f (fill_price=%.4f slip=%.4f)",
            order.side, sym, order.qty, price, fill_price,
            fill.slippage_cost,
        )
        self._emit(order)

    def _get_last_price(self, symbol: str) -> float:
        """Fetch latest price from Schwab real-time quotes, fall back to yfinance."""
        try:
            import asyncio
            from app.services.brokers.schwab import SchwabBroker
            broker = SchwabBroker()
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(broker.authenticate())
                quotes = loop.run_until_complete(broker.get_quotes([symbol]))
                if quotes and symbol in quotes:
                    price = quotes[symbol].last or quotes[symbol].ask or quotes[symbol].bid
                    if price and price > 0:
                        return float(price)
            finally:
                loop.close()
        except Exception:
            pass
        try:
            t = yf.Ticker(symbol)
            info = t.fast_info
            price = float(info.last_price or info.previous_close or 0)
            if price > 0:
                return price
        except Exception:
            pass
        raise RuntimeError(f"Could not fetch price for {symbol} — market may be closed or symbol invalid")

    def _emit(self, order: Order) -> None:
        for fn in self._callbacks:
            try:
                fn(order)
            except Exception as e:
                logger.warning("Fill callback error: %s", e)
