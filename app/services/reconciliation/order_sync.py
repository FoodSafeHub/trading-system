from __future__ import annotations

"""Broker → DB order reconciliation (the source of truth for fills).

The PnL page, Recent Fills, and realized-P&L all derive from `orders` rows with
status='filled' and a fill_price (FIFO in app/services/pnl/fifo.py reads ONLY
those). So a position that actually closed at the broker stays "open" in the app
until a filled SELL Order row exists for it.

The old `_run_gtc_fill_sync_job` only flipped our OWN status='submitted' rows on
the GLOBAL broker. That missed three real cases that left closes invisible:

  1. India (Zerodha) orders — never polled (only the global broker was checked).
  2. Fills on rows in other interim states (working / pending_activation / …),
     not just 'submitted'.
  3. ORPHAN fills with no matching DB row at all — a broker-native TRAILING_STOP
     that filled, or a position closed manually in the broker UI. These never had
     (or never kept) an Order row, so FIFO could never pair the close.

This module reconciles by pulling `broker.list_orders(account_id)` (ALL statuses)
for every relevant broker and UPSERTING each broker order into the DB by
broker_order_id:
  • match found + broker says filled  → flip our row to filled (+fill_price/at)
  • match found + broker says cancelled/rejected/expired → flip our row
  • NO match + broker says filled      → CREATE a filled Order row so FIFO sees it
Then it materializes realized trades so the PnL tables update immediately.

Read-only against the broker; only writes/updates orders + realized_trades.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.orders import Order
from app.services.brokers.factory import _build_one, get_broker

logger = logging.getLogger(__name__)

_FILLED = {"filled", "partial"}
_DEAD = {"canceled", "cancelled", "rejected", "expired"}


def _normalize_side(raw_side: str) -> str | None:
    """Map a broker instruction to BUY / SELL (what FIFO expects).

    Schwab uses BUY / SELL / SELL_SHORT / BUY_TO_COVER; Zerodha uses BUY / SELL.
    Returns None for anything we can't classify (so we skip it rather than
    miscount a leg).
    """
    s = (raw_side or "").upper()
    if s.startswith("BUY"):
        return "BUY"
    if s.startswith("SELL"):
        return "SELL"
    return None


def _coerce_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _broker_is_paper(broker) -> bool:
    name = (getattr(broker, "name", "") or "").lower()
    return "paper" in name and not any(b in name for b in ("schwab", "webull", "zerodha"))


def _reconcile_one_broker(broker, loop, lookback_days: int = 7) -> tuple[int, int]:
    """Reconcile a single broker's recent orders into the DB.

    lookback_days widens the broker order window (e.g. a one-time backfill of a
    symbol that closed more than the default window ago). Returns (updated,
    created) counts.
    """
    bname = getattr(broker, "name", "") or "unknown"
    try:
        loop.run_until_complete(broker.authenticate())
        accounts = loop.run_until_complete(broker.get_accounts())
        account_id = accounts[0].account_id if accounts else ""
        # Pass lookback_days when the adapter supports it; fall back gracefully
        # for brokers whose list_orders doesn't take the kwarg.
        try:
            broker_orders = loop.run_until_complete(
                broker.list_orders(account_id, lookback_days=lookback_days)
            ) or []
        except TypeError:
            broker_orders = loop.run_until_complete(broker.list_orders(account_id)) or []
    except Exception as exc:
        logger.warning("[order_sync] %s: could not list orders: %s", bname, exc)
        return 0, 0

    is_paper = _broker_is_paper(broker)
    updated = 0
    created = 0

    for bo in broker_orders:
        boid = getattr(bo, "broker_order_id", None)
        if not boid:
            continue
        status = (getattr(bo, "status", "") or "").lower()

        try:
            with SessionLocal() as db:
                row = (
                    db.query(Order)
                    .filter(Order.broker_order_id == str(boid))
                    .first()
                )

                # ── Existing row ─────────────────────────────────────────────
                if row is not None:
                    if status in _FILLED and row.status not in ("filled", "partial"):
                        row.status = status
                        if getattr(bo, "fill_price", None):
                            row.fill_price = float(bo.fill_price)
                        # filled_at: prefer broker close time when present.
                        close_t = (getattr(bo, "raw", {}) or {}).get("closeTime")
                        row.filled_at = _coerce_dt(close_t) if close_t else datetime.now(tz=timezone.utc)
                        db.commit()
                        updated += 1
                        logger.info(
                            "[order_sync] %s: %s %s filled @ %.4f (was %s)",
                            bname, row.symbol, boid, row.fill_price or 0.0, row.status,
                        )
                    elif status in _DEAD and row.status not in _DEAD:
                        row.status = "cancelled" if status.startswith("cancel") else status
                        db.commit()
                        updated += 1
                    continue

                # ── Orphan fill: no DB row, broker says filled → create one ──
                # This is the native-trailing-stop / manual-close case. Without
                # this, FIFO never pairs the close and PnL shows it still open.
                if status in _FILLED:
                    side = _normalize_side(getattr(bo, "side", ""))
                    fill_price = getattr(bo, "fill_price", None)
                    qty = float(getattr(bo, "quantity", 0) or 0)
                    if side is None or not fill_price or qty <= 0:
                        logger.debug(
                            "[order_sync] %s: orphan fill %s skipped "
                            "(side=%s price=%s qty=%s)",
                            bname, boid, side, fill_price, qty,
                        )
                        continue
                    close_t = (getattr(bo, "raw", {}) or {}).get("closeTime")
                    new_order = Order(
                        broker=bname,
                        broker_order_id=str(boid),
                        symbol=(getattr(bo, "symbol", "") or "").upper(),
                        side=side,
                        order_type=(getattr(bo, "order_type", "") or "MARKET").upper(),
                        quantity=qty,
                        status="filled",
                        is_paper=is_paper,
                        fill_price=float(fill_price),
                        filled_at=_coerce_dt(close_t) if close_t else datetime.now(tz=timezone.utc),
                        # Reconciled-in fills weren't placed through our pipeline.
                        source="broker_reconcile",
                        # Unique key so a second reconcile pass can't double-insert.
                        idempotency_key=f"reconcile-{bname}-{boid}",
                    )
                    db.add(new_order)
                    db.commit()
                    created += 1
                    logger.info(
                        "[order_sync] %s: INGESTED orphan fill %s %s x%.4f @ %.4f "
                        "(broker_id=%s) — was invisible to PnL.",
                        bname, side, new_order.symbol, qty, float(fill_price), boid,
                    )
        except Exception as exc:
            # A duplicate idempotency_key (race with another worker) lands here;
            # safe to ignore — the row exists.
            logger.debug("[order_sync] %s: order %s upsert skipped: %s", bname, boid, exc)

    return updated, created


def sync_broker_orders_once(lookback_days: int = 7) -> dict:
    """Reconcile every relevant broker's recent orders into the DB, then
    materialize realized trades so PnL/Recent Fills update automatically.

    lookback_days widens the broker order window — the default 7 covers routine
    syncing; pass a larger value for a one-time backfill of a position that
    closed further back.

    Brokers covered: the global active broker (Schwab/Webull/paper) plus Zerodha
    when any enabled assignment routes to India. Never raises — returns a summary.
    """
    from app.services.strategy.scheduler import _has_india_assignments

    loop = asyncio.new_event_loop()
    total_updated = 0
    total_created = 0
    try:
        brokers: list = []
        seen: set[str] = set()
        try:
            gb = get_broker()
            brokers.append(gb)
            seen.add(getattr(gb, "name", ""))
        except Exception as exc:
            logger.error("[order_sync] could not build global broker: %s", exc)

        try:
            if _has_india_assignments():
                zb = _build_one("zerodha")
                if getattr(zb, "name", "") not in seen:
                    brokers.append(zb)
        except Exception as exc:
            logger.error("[order_sync] could not build Zerodha broker: %s", exc)

        for broker in brokers:
            u, c = _reconcile_one_broker(broker, loop, lookback_days=lookback_days)
            total_updated += u
            total_created += c
    finally:
        loop.close()

    # Materialize realized round-trips now so the PnL page reflects closes
    # immediately rather than on the next /pnl read (it would catch up anyway,
    # but doing it here keeps everything consistent right after a fill lands).
    inserted = 0
    if total_updated or total_created:
        try:
            from app.services.pnl.store import sync_realized_trades
            with SessionLocal() as db:
                inserted, _ = sync_realized_trades(db)
        except Exception as exc:
            logger.warning("[order_sync] realized-trade sync failed: %s", exc)

    if total_updated or total_created or inserted:
        logger.info(
            "[order_sync] done — %d order(s) updated, %d orphan fill(s) ingested, "
            "%d realized round-trip(s) materialized.",
            total_updated, total_created, inserted,
        )
    return {
        "updated": total_updated,
        "created": total_created,
        "realized_inserted": inserted,
    }
