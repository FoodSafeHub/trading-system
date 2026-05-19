from __future__ import annotations

"""
Execution Service — orchestrates the full order lifecycle:
  Signal → Risk Check → Preview → Submit → Track → Persist

Every step is logged and persisted to SQLite.
No live order is submitted without passing ALL risk checks.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.db import SessionLocal
from app.models.executions import Execution
from app.models.orders import Order, OrderPreview
from app.schemas.orders import OrderRequest, OrderStatusResponse
from app.services.audit.service import AuditService
from app.services.brokers.base import BrokerBase
from app.services.risk.engine import RiskEngine

logger = logging.getLogger(__name__)
_audit = AuditService()
_risk = RiskEngine()


class ExecutionService:

    def __init__(self, broker: BrokerBase) -> None:
        self.broker = broker

    async def execute(
        self,
        order_req: OrderRequest,
        account_id: str,
        signal_id: Optional[int] = None,
        estimated_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Full execution pipeline. Returns the persisted Order on success, None if blocked.
        """
        # ── Idempotency key ─────────────────────────────────────────────────
        if not order_req.idempotency_key:
            order_req.idempotency_key = str(uuid.uuid4())
        if signal_id:
            order_req.signal_id = signal_id

        logger.info(
            f"[exec] Starting execution: {order_req.side} {order_req.symbol} "
            f"x{order_req.quantity:.2f} (key={order_req.idempotency_key})"
        )

        # ── Step 1: Risk checks ─────────────────────────────────────────────
        risk_result = _risk.check(order_req, estimated_price=estimated_price)
        if not risk_result.passed:
            logger.warning("[exec] Risk BLOCKED: %s", risk_result.blocked_reason)
            _audit.log(
                event_type="RISK_BLOCKED",
                description=f"{order_req.side} {order_req.symbol} blocked: {risk_result.blocked_reason}",
                metadata={"order": order_req.model_dump(mode="json")},
            )
            return None

        if risk_result.warnings:
            for w in risk_result.warnings:
                logger.warning("[exec] Risk warning: %s", w)

        # ── Step 2: Persist order record (pending) ──────────────────────────
        db_order = self._persist_order(order_req, signal_id, status="pending")

        # ── Step 3: Preview (if broker supports it) ─────────────────────────
        try:
            preview = await self.broker.preview_order(order_req, account_id)
            self._persist_preview(db_order.id, preview)
            self._update_order_status(db_order.id, "previewed", preview_json=json.dumps(preview.model_dump()))
            logger.info(
                f"[exec] Preview: cost={preview.estimated_cost or 0:.2f} "
                f"commission={preview.estimated_commission or 0:.2f}"
            )
        except NotImplementedError:
            logger.debug("[exec] Broker does not support preview — skipping")
        except Exception as exc:
            logger.warning("[exec] Preview failed (non-fatal): %s", exc)

        # ── Step 4: Submit order ─────────────────────────────────────────────
        try:
            status_resp = await self.broker.place_order(order_req, account_id)
            self._update_order_status(
                db_order.id,
                status="submitted",
                broker_order_id=status_resp.broker_order_id,
                submitted_at=datetime.now(tz=timezone.utc),
            )
            _audit.log(
                event_type="ORDER_SUBMITTED",
                entity_type="order",
                entity_id=db_order.id,
                description=f"{order_req.side} {order_req.symbol} x{order_req.quantity} → broker_id={status_resp.broker_order_id}",
            )
            logger.info(
                "[exec] Order submitted: broker_id=%s status=%s",
                status_resp.broker_order_id, status_resp.status,
            )
        except Exception as exc:
            logger.error("[exec] Order submission failed: %s", exc)
            self._update_order_status(db_order.id, status="error", error_message=str(exc))
            self._persist_execution_event(db_order.id, "error", error=str(exc))
            _audit.log(
                event_type="ORDER_ERROR",
                entity_type="order",
                entity_id=db_order.id,
                description=str(exc),
            )
            return db_order

        # ── Step 5: Confirm actual broker state ─────────────────────────────
        # Schwab returns 201 from place_order before deciding to accept/reject the
        # order. Without a follow-up GET, our DB reports "submitted" for orders the
        # broker immediately rejected. Poll once to capture the real status.
        confirmed = await self._confirm_broker_status(
            status_resp.broker_order_id, account_id, db_order.id
        )
        final_status = confirmed.status if confirmed else status_resp.status

        # ── Step 6: Handle immediate fill ────────────────────────────────────
        if final_status in ("filled", "partial"):
            self._handle_fill(db_order.id, confirmed or status_resp)

        return db_order

    async def _confirm_broker_status(
        self, broker_order_id: Optional[str], account_id: str, order_id: int
    ) -> Optional[OrderStatusResponse]:
        """One-shot status poll after place_order. Updates DB to the real broker state."""
        if not broker_order_id:
            return None
        try:
            confirmed = await self.broker.get_order(broker_order_id, account_id)
        except Exception as exc:
            logger.warning("[exec] Status confirm failed for %s: %s", broker_order_id, exc)
            return None

        # Map broker status into our lifecycle. Anything that isn't a working state
        # (queued/working/pending_activation) overrides "submitted".
        broker_status = (confirmed.status or "").lower()
        terminal = {"filled", "partial", "rejected", "cancelled", "canceled", "expired", "replaced"}
        if broker_status in terminal:
            local_status = "partial" if broker_status == "partial" else broker_status
            if local_status == "canceled":
                local_status = "cancelled"
            reason = ""
            if isinstance(confirmed.raw, dict):
                reason = confirmed.raw.get("statusDescription") or ""
            self._update_order_status(
                order_id,
                status=local_status,
                error_message=reason or None,
            )
            _audit.log(
                event_type=f"ORDER_{local_status.upper()}",
                entity_type="order",
                entity_id=order_id,
                description=f"broker confirmed status={local_status} reason={reason or 'n/a'}",
            )
            logger.info(
                "[exec] Broker confirmed status=%s for order_id=%s (%s)",
                local_status, order_id, reason or "no reason",
            )
        return confirmed

    # ── Persistence helpers ──────────────────────────────────────────────────

    def _persist_order(self, req: OrderRequest, signal_id: Optional[int], status: str) -> Order:
        with SessionLocal() as db:
            order = Order(
                broker=self.broker.name,
                symbol=req.symbol,
                side=req.side,
                order_type=req.order_type,
                quantity=req.quantity,
                limit_price=req.limit_price,
                stop_price=req.stop_price,
                status=status,
                is_paper=(self.broker.name == "paper"),
                signal_id=signal_id,
                idempotency_key=req.idempotency_key,
                # source is required for downstream auditing; pydantic enforces
                # the literal values, but default to "manual" defensively.
                source=getattr(req, "source", "manual") or "manual",
                created_at=datetime.now(tz=timezone.utc),
            )
            db.add(order)
            db.commit()
            db.refresh(order)
            return order

    def _update_order_status(
        self,
        order_id: int,
        status: str,
        broker_order_id: Optional[str] = None,
        submitted_at: Optional[datetime] = None,
        filled_at: Optional[datetime] = None,
        fill_price: Optional[float] = None,
        error_message: Optional[str] = None,
        preview_json: Optional[str] = None,
    ) -> None:
        with SessionLocal() as db:
            order = db.query(Order).filter_by(id=order_id).first()
            if not order:
                return
            order.status = status
            if broker_order_id:
                order.broker_order_id = broker_order_id
            if submitted_at:
                order.submitted_at = submitted_at
            if filled_at:
                order.filled_at = filled_at
            if fill_price is not None:
                order.fill_price = fill_price
            if error_message:
                order.error_message = error_message
            if preview_json:
                order.preview_json = preview_json
            db.commit()

    def _persist_preview(self, order_id: int, preview) -> None:
        with SessionLocal() as db:
            prev = OrderPreview(
                order_id=order_id,
                broker=self.broker.name,
                estimated_cost=preview.estimated_cost,
                estimated_commission=preview.estimated_commission,
                buying_power_effect=preview.buying_power_effect,
                raw_response_json=json.dumps(preview.raw),
            )
            db.add(prev)
            db.commit()

    def _persist_execution_event(
        self,
        order_id: int,
        event_type: str,
        fill_price: Optional[float] = None,
        qty: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with SessionLocal() as db:
            exec_rec = Execution(
                order_id=order_id,
                broker=self.broker.name,
                event_type=event_type,
                fill_price=fill_price,
                quantity_filled=qty,
                occurred_at=datetime.now(tz=timezone.utc),
                raw_response_json=json.dumps({"error": error}) if error else None,
            )
            db.add(exec_rec)
            db.commit()

    def _handle_fill(self, order_id: int, status: OrderStatusResponse) -> None:
        fill_status = "filled" if status.status == "filled" else "partial"
        self._update_order_status(
            order_id,
            status=fill_status,
            fill_price=status.fill_price,
            filled_at=datetime.now(tz=timezone.utc),
        )
        self._persist_execution_event(
            order_id,
            event_type="fill" if fill_status == "filled" else "partial_fill",
            fill_price=status.fill_price,
            qty=status.filled_quantity,
        )
        logger.info(f"[exec] Fill recorded: order_id={order_id} price={status.fill_price or 0:.4f}")
