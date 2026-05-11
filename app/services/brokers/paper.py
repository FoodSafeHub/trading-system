from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.schemas.account import AccountSummary, Position, Quote
from app.schemas.orders import OrderPreviewResponse, OrderRequest, OrderStatusResponse
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)

# Simulated paper account state (in-memory, resets on restart)
_paper_cash: float = 100_000.0
_paper_positions: Dict[str, Dict] = {}
_paper_orders: Dict[str, Dict] = {}


class PaperBroker(BrokerBase):
    """
    Simulated paper trading broker.
    All orders are filled immediately at a slightly randomised price.
    State is in-memory and resets on process restart.
    """

    name = "paper"

    async def authenticate(self) -> None:
        logger.info("[paper] Paper broker authenticated (no credentials needed)")

    async def refresh_token_if_needed(self) -> None:
        pass  # No-op for paper

    async def get_accounts(self) -> List[AccountSummary]:
        global _paper_cash
        equity = _paper_cash + sum(
            p["quantity"] * p.get("current_price", p["average_cost"])
            for p in _paper_positions.values()
        )
        return [
            AccountSummary(
                broker="paper",
                account_id="PAPER-001",
                account_type="MARGIN",
                buying_power=_paper_cash,
                cash=_paper_cash,
                equity=equity,
                is_paper=True,
            )
        ]

    async def get_positions(self, account_id: str) -> List[Position]:
        return [
            Position(
                symbol=sym,
                quantity=p["quantity"],
                average_cost=p["average_cost"],
                current_price=p.get("current_price"),
                market_value=p["quantity"] * p.get("current_price", p["average_cost"]),
                unrealized_pnl=(
                    (p.get("current_price", p["average_cost"]) - p["average_cost"]) * p["quantity"]
                    if p.get("current_price") else None
                ),
            )
            for sym, p in _paper_positions.items()
            if p["quantity"] > 0
        ]

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        # Simulate a plausible last price (pure simulation — not real data)
        quotes = {}
        for sym in symbols:
            base = _paper_positions.get(sym, {}).get("average_cost", 100.0)
            last = round(base * (1 + random.uniform(-0.005, 0.005)), 2)
            quotes[sym] = Quote(
                symbol=sym,
                bid=round(last - 0.01, 2),
                ask=round(last + 0.01, 2),
                last=last,
                timestamp=datetime.utcnow().isoformat(),
            )
        return quotes

    async def preview_order(self, order: OrderRequest, account_id: str) -> OrderPreviewResponse:
        estimated_cost = order.quantity * (order.limit_price or 100.0)
        return OrderPreviewResponse(
            broker="paper",
            estimated_cost=estimated_cost,
            estimated_commission=0.0,
            buying_power_effect=-estimated_cost if order.side == "BUY" else estimated_cost,
        )

    async def place_order(self, order: OrderRequest, account_id: str) -> OrderStatusResponse:
        global _paper_cash

        broker_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        fill_price = round((order.limit_price or 100.0) * (1 + random.uniform(-0.002, 0.002)), 4)
        cost = fill_price * order.quantity

        if order.side == "BUY":
            if _paper_cash < cost:
                logger.warning("[paper] Insufficient paper cash for BUY order")
                return OrderStatusResponse(
                    broker_order_id=broker_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    status="rejected",
                    raw={"reason": "insufficient_paper_cash"},
                )
            _paper_cash -= cost
            pos = _paper_positions.get(order.symbol, {"quantity": 0.0, "average_cost": 0.0})
            total_qty = pos["quantity"] + order.quantity
            avg_cost = (pos["quantity"] * pos["average_cost"] + cost) / total_qty
            _paper_positions[order.symbol] = {
                "quantity": total_qty,
                "average_cost": avg_cost,
                "current_price": fill_price,
            }
        else:  # SELL
            pos = _paper_positions.get(order.symbol, {"quantity": 0.0, "average_cost": 0.0})
            qty_to_sell = min(order.quantity, pos["quantity"])
            _paper_cash += fill_price * qty_to_sell
            pos["quantity"] -= qty_to_sell
            if pos["quantity"] <= 0:
                _paper_positions.pop(order.symbol, None)
            else:
                _paper_positions[order.symbol] = pos

        _paper_orders[broker_order_id] = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "fill_price": fill_price,
            "status": "filled",
        }

        logger.info(
            f"[paper] Order filled: {order.side} {order.symbol} x{order.quantity:.2f} "
            f"@ {fill_price:.4f} (id={broker_order_id})"
        )

        return OrderStatusResponse(
            broker_order_id=broker_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            status="filled",
            fill_price=fill_price,
            filled_quantity=order.quantity,
            raw={"paper": True},
        )

    async def cancel_order(self, broker_order_id: str, account_id: str) -> bool:
        if broker_order_id in _paper_orders:
            _paper_orders[broker_order_id]["status"] = "cancelled"
            return True
        return False

    async def get_order(self, broker_order_id: str, account_id: str) -> OrderStatusResponse:
        o = _paper_orders.get(broker_order_id)
        if not o:
            raise ValueError(f"Paper order not found: {broker_order_id}")
        return OrderStatusResponse(
            broker_order_id=broker_order_id,
            symbol=o["symbol"],
            side=o["side"],
            order_type="MARKET",
            quantity=o["quantity"],
            status=o["status"],
            fill_price=o.get("fill_price"),
            filled_quantity=o["quantity"],
            raw={"paper": True},
        )

    async def list_orders(self, account_id: str, status: Optional[str] = None) -> List[OrderStatusResponse]:
        result = []
        for oid, o in _paper_orders.items():
            if status and o["status"] != status:
                continue
            result.append(
                OrderStatusResponse(
                    broker_order_id=oid,
                    symbol=o["symbol"],
                    side=o["side"],
                    order_type="MARKET",
                    quantity=o["quantity"],
                    status=o["status"],
                    fill_price=o.get("fill_price"),
                    raw={"paper": True},
                )
            )
        return result
