from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.orders import Order
from app.models.signals import Signal
from app.schemas.orders import OrderOut, OrderRequest
from app.schemas._serializers import serialize_et
from app.services.brokers.factory import get_broker
from app.services.execution.service import ExecutionService

router = APIRouter(prefix="/orders", tags=["orders"])


@router.get("")
def list_orders(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[dict[str, Any]]:
    """Return recent orders with strategy_name joined in when signal_id is present.

    We bypass response_model=List[OrderOut] here because we want the joined
    strategy_name field on the wire so the dashboard can show a "Source" column
    without a second round-trip per order.
    """
    q = db.query(Order).order_by(Order.created_at.desc())
    if status:
        q = q.filter(Order.status == status)
    orders = q.limit(limit).all()

    # Single lookup for the strategy names — avoids N+1.
    signal_ids = {o.signal_id for o in orders if o.signal_id is not None}
    strategy_by_signal: dict[int, str] = {}
    if signal_ids:
        for sig in db.query(Signal).filter(Signal.id.in_(signal_ids)).all():
            strategy_by_signal[sig.id] = sig.strategy_name

    out: list[dict[str, Any]] = []
    for o in orders:
        out.append({
            "id": o.id,
            "broker": o.broker,
            "broker_order_id": o.broker_order_id,
            "symbol": o.symbol,
            "side": o.side,
            "order_type": o.order_type,
            "quantity": o.quantity,
            "limit_price": o.limit_price,
            "stop_price": o.stop_price,
            "status": o.status,
            "is_paper": o.is_paper,
            "signal_id": o.signal_id,
            "preview_json": o.preview_json,
            "source": getattr(o, "source", None) or "manual",
            "strategy_name": strategy_by_signal.get(o.signal_id) if o.signal_id else None,
            "created_at": serialize_et(o.created_at),
            "submitted_at": serialize_et(o.submitted_at),
            "filled_at": serialize_et(o.filled_at),
            "fill_price": o.fill_price,
            "error_message": o.error_message,
        })
    return out


@router.post("/manual", response_model=Optional[OrderOut])
async def manual_order(order_req: OrderRequest, account_id: str = ""):
    """Manually submit an order through the full execution pipeline (risk checks included)."""
    broker = get_broker()
    await broker.authenticate()

    if not account_id:
        accounts = await broker.get_accounts()
        if not accounts:
            raise HTTPException(status_code=400, detail="No accounts found")
        account_id = accounts[0].account_id

    # Force-stamp source=manual so the dashboard's Source column shows the
    # truth even if a caller passed a different value in the payload.
    order_req.source = "manual"

    svc = ExecutionService(broker)
    result = await svc.execute(order_req, account_id=account_id)
    if result is None:
        raise HTTPException(status_code=403, detail="Order blocked by risk engine. Check audit logs.")
    return result


@router.get("/broker")
async def list_broker_orders(account_id: str = ""):
    """Fetch orders directly from the broker (Schwab / paper) — not the local DB."""
    broker = get_broker()
    await broker.authenticate()

    if not account_id:
        accounts = await broker.get_accounts()
        if not accounts:
            raise HTTPException(status_code=400, detail="No accounts found")
        account_id = accounts[0].account_id

    orders = await broker.list_orders(account_id)
    return [
        {
            "broker_order_id": o.broker_order_id,
            "symbol":          o.symbol,
            "side":            o.side,
            "order_type":      o.order_type,
            "quantity":        o.quantity,
            "filled_quantity": o.filled_quantity,
            "fill_price":      o.fill_price,
            "status":          o.status,
        }
        for o in orders
    ]


@router.get("/{order_id}", response_model=OrderOut)
def get_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(Order).filter_by(id=order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@router.post("/{broker_order_id}/cancel")
async def cancel_order(broker_order_id: str, account_id: str = ""):
    broker = get_broker()
    await broker.authenticate()

    if not account_id:
        accounts = await broker.get_accounts()
        if not accounts:
            raise HTTPException(status_code=400, detail="No accounts found")
        account_id = accounts[0].account_id

    success = await broker.cancel_order(broker_order_id, account_id)
    return {"cancelled": success}
