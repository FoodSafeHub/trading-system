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


def _attribute_fill_to_ledger(db, order: Order) -> None:
    """Update the per-strategy share ledger for a freshly-filled order.

    Attribution chain: Order.signal_id -> Signal.strategy_name, then match the
    enabled assignment to recover its broker route. Orders with no signal link
    (orphan native-trail/manual closes) or no matching assignment are skipped —
    the ledger only tracks lots our strategies opened, and the scheduler's
    min(ledger, broker_held) SELL clamp keeps us from overselling when an
    untracked lot exists.

    IMPORTANT: the Signal row stores the scheduler's *label*, which for scanner
    and perplexity systems is prefixed ("scanner:NAME", "perplexity:NAME"). The
    assignment row — and the scheduler's ledger reads (strategy_ledger.get_held)
    — use the BARE system + strategy name. We must split the label back into
    (system, bare_name) and key the ledger by those, or the keys won't line up
    and every BUY would see held=0 and re-buy the full cap each cycle.
    """
    try:
        from app.models.assignments import SymbolStrategyAssignment
        from app.models.signals import Signal
        from app.services.strategy import strategy_ledger
        from app.services.markets import is_india_symbol

        if not order.signal_id:
            return
        sig = db.query(Signal).filter_by(id=order.signal_id).first()
        if sig is None or not sig.strategy_name:
            return
        symbol = (order.symbol or "").upper()

        # Split a possible "system:NAME" label into bare (system, name). Bollinger
        # labels carry no prefix, so default the system to "bollinger".
        label = sig.strategy_name
        if ":" in label:
            prefix, bare_name = label.split(":", 1)
            sys_name = prefix
        else:
            bare_name, sys_name = label, None

        # Resolve the exact assignment. Prefer an exact (symbol, system, name)
        # match; fall back to (symbol, name) so a missing/odd prefix still maps
        # when the symbol+name pair is unambiguous.
        q = db.query(SymbolStrategyAssignment).filter_by(
            symbol=symbol, strategy_name=bare_name
        )
        asgn = (q.filter_by(system=sys_name).first() if sys_name else None) or q.first()
        if asgn is None:
            return  # consensus / non-assigned fill — not ledger-tracked.
        broker = asgn.broker or "default"
        if broker == "default" and is_india_symbol(symbol):
            broker = "zerodha"
        strategy_ledger.apply_fill(
            symbol=symbol,
            system=asgn.system,
            strategy_name=asgn.strategy_name,
            side=order.side,
            qty=float(order.quantity or 0.0),
            price=order.fill_price,
            broker=broker,
            db=db,
        )
    except Exception as exc:
        logger.debug("[order_sync] ledger attribution skipped for order %s: %s",
                     getattr(order, "id", "?"), exc)

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
                        # Attribute this fill to the firing strategy's share ledger
                        # so per-strategy SELL sizing stays accurate. Same session,
                        # committed together with the status flip.
                        _attribute_fill_to_ledger(db, row)
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


def sync_broker_orders_once(lookback_days: int | None = None) -> dict:
    """Reconcile every relevant broker's recent orders into the DB, then
    materialize realized trades so PnL/Recent Fills update automatically.

    lookback_days widens the broker order window. None (default) uses
    settings.order_sync_lookback_days; pass a larger value for a one-time
    backfill of a position that closed further back.

    Brokers covered: the global active broker (Schwab/Webull/paper) plus Zerodha
    when any enabled assignment routes to India. Never raises — returns a summary.
    """
    from app.services.strategy.scheduler import _has_india_assignments

    from app.config import get_settings

    if lookback_days is None:
        lookback_days = get_settings().order_sync_lookback_days

    loop = asyncio.new_event_loop()
    total_updated = 0
    total_created = 0
    try:
        brokers: list = []
        seen: set[str] = set()

        def _add(b) -> None:
            name = getattr(b, "name", "") or ""
            if name and name not in seen:
                seen.add(name)
                brokers.append(b)

        try:
            gb = get_broker()
            # A MultiBroker (trade_routing="both") FANS OUT placements to every
            # leg but its read methods (list_orders) delegate to the PRIMARY leg
            # only. Reconciling the wrapper therefore polls just one broker and
            # misses fills on the other leg — e.g. SO sold on Webull while the
            # primary is Schwab stayed "open" forever. Expand to the underlying
            # legs so EVERY broker the system trades through is reconciled. A
            # single (non-multi) broker is used as-is.
            legs = getattr(gb, "_brokers", None)
            if legs:
                for leg in legs:
                    _add(leg)
            else:
                _add(gb)
        except Exception as exc:
            logger.error("[order_sync] could not build global broker: %s", exc)

        try:
            if _has_india_assignments():
                _add(_build_one("zerodha"))
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
