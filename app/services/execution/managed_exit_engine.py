"""Bot-managed exit engine — no resting protective orders at the broker.

When settings.schwab_managed_exits_enabled is on, US/Schwab swing positions
keep NO protective order resting at the broker: a resting stop (native trail
or static STOP) is exactly what a market-fear sweep triggers, converting a
temporary drawdown into a realized loss. Instead:

  * the software trail level lives in managed_exit_state.target_stop and the
    trail-hit branch of ExecutionService.tighten_trail_on_sell is the only
    exit executor (an idempotent MARKET sell);
  * a SELL signal on a RED position (price < FIFO avg cost) is HELD, not
    executed — the user reviews it manually; the engine resumes the normal
    exit path automatically when the position turns green (red_hold mode);
  * the safety net is alert-only: escalating drawdown notifications at
    configurable levels vs FIFO cost. This module never places orders.

run_once() is called from the scheduler's 60-second fast-trail job right
after reconcile_trails_now(); the invariant watchdog treats a stale
heartbeat/last_eval_at during market hours as a violation.

India (Zerodha static STOP + Chandelier) and the day-trading autotrader are
out of scope and untouched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models.managed_exit_state import (
    MODE_EXITED,
    MODE_MONITORING,
    MODE_RED_HOLD,
    ManagedExitState,
)

logger = logging.getLogger(__name__)

HEARTBEAT_KEY = "managed_exits_last_run"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def managed_mode_active() -> bool:
    """Master switch for bot-managed (no-resting-order) exits."""
    return bool(get_settings().schwab_managed_exits_enabled)


def is_managed_symbol(symbol: str) -> bool:
    """True when this symbol's exits are bot-managed (US swing path only).

    India symbols always keep the Zerodha static-STOP + Chandelier path —
    Kite has no native trail and the resting-stop sweep problem is a
    Schwab/US complaint. Fail-closed to the legacy path on classification
    errors so a lookup blip can't strip protection semantics.
    """
    if not managed_mode_active():
        return False
    try:
        from app.services.markets import is_india_symbol
        return not is_india_symbol(symbol)
    except Exception as exc:
        logger.debug("[managed-exits] is_managed_symbol(%s) classify failed: %s",
                     symbol, exc)
        return False


def parse_alert_levels(raw: str) -> list[float]:
    """Parse the drawdown-alert config into negative pcts, shallow→deep.

    "-8,-12,-20" → [-8.0, -12.0, -20.0]. Invalid entries are dropped;
    positive values are negated (a "-8" vs "8" typo shouldn't kill alerts).
    """
    levels: list[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if v > 0:
            v = -v
        if v < 0:
            levels.append(v)
    return sorted(set(levels), reverse=True)  # shallow (e.g. -8) first


def fifo_avg_costs(db: Session, symbols: Optional[list[str]] = None) -> dict[str, float]:
    """Qty-weighted FIFO avg cost per symbol from the open lots ledger.

    The red/green line and drawdown-alert base. Empty dict on ledger failure
    — callers must fail OPEN (treat cost-unknown as green) so a ledger gap
    can't strand a position un-exitable.
    """
    try:
        from app.services.pnl.store import sync_realized_trades
        _ins, fifo = sync_realized_trades(db)
    except Exception as exc:
        logger.warning("[managed-exits] FIFO build failed: %s", exc)
        return {}
    want = {s.upper() for s in symbols} if symbols else None
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    for lot in fifo.open_lots:
        sym = lot.symbol.upper()
        if want is not None and sym not in want:
            continue
        qty[sym] = qty.get(sym, 0.0) + lot.quantity
        cost[sym] = cost.get(sym, 0.0) + lot.quantity * lot.buy_price
    return {s: cost[s] / qty[s] for s in qty if qty[s] > 0}


def get_state(db: Session, symbol: str) -> Optional[ManagedExitState]:
    return (
        db.query(ManagedExitState)
        .filter(ManagedExitState.symbol == symbol.upper())
        .one_or_none()
    )


def upsert_state(db: Session, symbol: str, **fields) -> ManagedExitState:
    """Get-or-create the per-symbol state row and apply field updates."""
    state = get_state(db, symbol)
    if state is None:
        state = ManagedExitState(symbol=symbol.upper(), mode=MODE_MONITORING)
        db.add(state)
    for k, v in fields.items():
        setattr(state, k, v)
    state.updated_at = _utcnow()
    return state


def red_hold_check(
    db: Session,
    symbol: str,
    current_price: float,
    *,
    signal_id: Optional[int] = None,
    signal_price: Optional[float] = None,
    source: str = "scheduler",
    avg_cost: Optional[float] = None,
) -> bool:
    """Gate an assigned/consensus SELL on the position being green.

    Returns True when the position is RED (price < FIFO avg cost) and the
    hold applies: the caller must NOT execute the exit and must NOT mark the
    signal acted_on (so reconcile keeps re-evaluating it every cycle — the
    green flip then resumes the normal exit path with no extra machinery).
    Notifies ONCE per hold episode (dedup via red_hold_notified_at).

    Fail-open: returns False (exit proceeds normally) when the flags are
    off, the symbol isn't bot-managed, or FIFO cost is unknown.
    """
    settings = get_settings()
    if not (settings.red_hold_enabled and is_managed_symbol(symbol)):
        return False
    if not current_price or current_price <= 0:
        return False
    if avg_cost is None:
        avg_cost = fifo_avg_costs(db, [symbol]).get(symbol.upper())
    if not avg_cost or avg_cost <= 0:
        return False  # fail-open: cost unknown → normal exit path
    if current_price >= avg_cost:
        return False  # green — exit proceeds

    now = _utcnow()
    state = get_state(db, symbol)
    fresh_episode = state is None or state.mode != MODE_RED_HOLD
    state = upsert_state(
        db, symbol,
        mode=MODE_RED_HOLD,
        signal_id=signal_id,
        signal_price=signal_price,
        avg_cost=avg_cost,
        last_eval_at=now,
    )
    if fresh_episode:
        state.red_hold_since = now
        state.red_hold_notified_at = None
    db.commit()

    if state.red_hold_notified_at is None:
        dd = (current_price / avg_cost - 1.0) * 100.0
        try:
            from app.services.notifications.bus import notify_suppression
            notify_suppression(
                symbol=symbol,
                reason="red_hold",
                detail=(
                    f"SELL signal fired but position is RED ({dd:+.1f}% vs cost "
                    f"${avg_cost:,.2f}, now ${current_price:,.2f}). Holding for "
                    f"manual review — will resume the exit automatically when green."
                ),
                source=source,
                direction="SELL",
                toast=True,
            )
        except Exception:
            pass  # notification must never block the hold decision
        state.red_hold_notified_at = now
        db.commit()
    return True


def drawdown_alert_pass(db: Session, state: ManagedExitState, current_price: float) -> Optional[float]:
    """Alert-only safety net: escalating notifications at drawdown levels.

    Fires the deepest configured level the drawdown has crossed and not yet
    alerted this episode. A fired level re-arms only after price recovers
    managed_exit_alert_rearm_pct points above it (hysteresis — oscillation
    around a threshold can't spam). Never places orders.

    Returns the level fired this pass (for tests/telemetry), else None.
    """
    settings = get_settings()
    if not state.avg_cost or state.avg_cost <= 0 or not current_price or current_price <= 0:
        return None
    dd = (current_price / state.avg_cost - 1.0) * 100.0
    levels = parse_alert_levels(settings.managed_exit_drawdown_alert_levels)
    if not levels:
        return None
    crossed = [lv for lv in levels if dd <= lv]
    deepest = min(crossed) if crossed else None
    last = state.last_alert_level

    # Re-arm with hysteresis: once price recovers rearm_pct above the last
    # fired level, relax the marker to the deepest level still crossed (or
    # clear it) so a re-plunge alerts again.
    if last is not None and dd >= last + settings.managed_exit_alert_rearm_pct:
        state.last_alert_level = deepest
        last = deepest

    if deepest is None or (last is not None and deepest >= last):
        db.commit()
        return None

    severe = deepest == min(levels)
    toast = deepest != max(levels)  # shallowest level stays quiet (Telegram only)
    try:
        from app.services.notifications.bus import notify_suppression
        notify_suppression(
            symbol=state.symbol,
            reason="drawdown_alert",
            detail=(
                ("SEVERE: " if severe else "")
                + f"{state.symbol} is {dd:+.1f}% vs cost ${state.avg_cost:,.2f} "
                f"(now ${current_price:,.2f}) — crossed the {deepest:g}% alert level. "
                f"No resting stop (bot-managed exits); review if action is needed."
            ),
            source="managed_exits",
            toast=toast,
        )
    except Exception:
        pass
    state.last_alert_level = deepest
    state.last_alert_at = _utcnow()
    db.commit()
    return deepest


def _write_heartbeat(db: Session) -> None:
    from app.models.settings import AppSetting
    row = db.get(AppSetting, HEARTBEAT_KEY)
    now_iso = _utcnow().isoformat()
    if row is None:
        db.add(AppSetting(key=HEARTBEAT_KEY, value=now_iso,
                          description="Managed-exit engine last run (UTC)"))
    else:
        row.value = now_iso
    db.commit()


def heartbeat_age_seconds(db: Session) -> Optional[float]:
    """Seconds since the engine last completed a pass; None if it never ran."""
    from app.models.settings import AppSetting
    row = db.get(AppSetting, HEARTBEAT_KEY)
    if row is None or not row.value:
        return None
    try:
        ts = datetime.fromisoformat(row.value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_utcnow() - ts).total_seconds()


def run_once(
    positions: Optional[dict[str, float]] = None,
    prices: Optional[dict[str, float]] = None,
) -> dict:
    """One monitoring pass over held bot-managed positions.

    Called from the 60s fast-trail job (which passes the positions/quotes it
    already fetched — no second broker round-trip). With no arguments it
    builds its own broker context, for on-demand/API use.

    Per held US position with its market open: upsert the state row, refresh
    FIFO avg cost, clear a red-hold whose position turned green (notify once
    — the still-unacted SELL signal makes the next reconcile pass arm the
    trail), and run the drawdown-alert pass. State rows whose position is
    gone are marked exited. Finally stamps the global heartbeat.
    """
    if not managed_mode_active():
        return {"status": "disabled"}

    if positions is None or prices is None:
        try:
            positions, prices = _fetch_positions_and_prices()
        except Exception as exc:
            logger.error("[managed-exits] position/quote fetch failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # Lazy import — scheduler imports this module, so the reverse must be lazy.
    from app.services.strategy.scheduler import _symbol_market_open

    held = {s.upper(): q for s, q in positions.items() if q and q >= 1.0}
    evaluated = 0
    alerts = 0
    greened: list[str] = []

    with SessionLocal() as db:
        avg_costs = fifo_avg_costs(db, list(held.keys()))
        now = _utcnow()

        for symbol, _qty in held.items():
            if not is_managed_symbol(symbol):
                continue
            if not _symbol_market_open(symbol):
                continue
            price = prices.get(symbol)
            state = upsert_state(
                db, symbol,
                avg_cost=avg_costs.get(symbol),
                last_eval_at=now,
            )
            db.commit()
            evaluated += 1
            if not price or price <= 0:
                continue

            if (state.mode == MODE_RED_HOLD and state.avg_cost
                    and price >= state.avg_cost):
                state.mode = MODE_MONITORING
                state.red_hold_since = None
                state.red_hold_notified_at = None
                db.commit()
                greened.append(symbol)
                try:
                    from app.services.notifications.bus import notify_suppression
                    notify_suppression(
                        symbol=symbol,
                        reason="red_hold_cleared",
                        detail=(
                            f"{symbol} turned GREEN (now ${price:,.2f} vs cost "
                            f"${state.avg_cost:,.2f}) — resuming the normal exit path."
                        ),
                        source="managed_exits",
                        direction="SELL",
                        toast=True,
                    )
                except Exception:
                    pass

            if state.avg_cost:
                if drawdown_alert_pass(db, state, price) is not None:
                    alerts += 1

        # Positions gone from the broker → exited (keeps the dashboard honest).
        stale = (
            db.query(ManagedExitState)
            .filter(ManagedExitState.mode != MODE_EXITED)
            .all()
        )
        for row in stale:
            if row.symbol not in held:
                row.mode = MODE_EXITED
        db.commit()

        _write_heartbeat(db)

    return {
        "status": "ok",
        "evaluated": evaluated,
        "alerts_fired": alerts,
        "red_holds_cleared": greened,
    }


_PROTECTIVE_ORDER_TYPES = ("STOP", "STOP_LIMIT", "TRAILING_STOP")
_BOT_IDEMPOTENCY_PREFIXES = ("trailstop-", "sell-trail-", "protstop-")
_CANCELLED_STATUSES = {"canceled", "cancelled", "expired", "replaced"}
_OPEN_ORDER_STATUSES = (
    "working", "awaiting_stop_condition", "pending_activation",
    "submitted", "queued", "accepted", "pending_acknowledgement",
)


def migrate_from_resting_stops(dry_run: bool = True) -> dict:
    """One-time rollout migration: cancel every BOT-placed resting protective
    SELL at the default (Schwab) broker and hand those positions to the
    managed-exit engine.

    Ordering per order is seed-state-FIRST, then cancel, then verify — the
    protection level (stop price → target_stop, implied peak → trail_peaks)
    must survive the switch, and only a VERIFIED cancel counts; failures are
    reported and the order left resting. Manual (non-bot) stops are listed
    for the user but never touched.

    dry_run=True (default) only returns the candidate list.
    """
    if not managed_mode_active():
        raise RuntimeError(
            "schwab_managed_exits_enabled is off — enable it before migrating."
        )
    import asyncio as _aio

    from app.services.brokers.factory import get_broker

    loop = _aio.new_event_loop()
    try:
        broker = get_broker()
        loop.run_until_complete(broker.authenticate())
        accts = loop.run_until_complete(broker.get_accounts())
        account_id = accts[0].account_id if accts else ""

        # Sweep all pending statuses (a resting STOP parks in
        # AWAITING_STOP_CONDITION on Schwab, not "working").
        open_orders: list = []
        seen: set = set()
        for st in _OPEN_ORDER_STATUSES:
            try:
                for o in loop.run_until_complete(
                    broker.list_orders(account_id, status=st)
                ) or []:
                    boid = getattr(o, "broker_order_id", None)
                    if not boid or boid in seen:
                        continue
                    seen.add(boid)
                    open_orders.append(o)
            except Exception as exc:
                logger.debug("[managed-exits] migrate list_orders(%s): %s", st, exc)

        protective = [
            o for o in open_orders
            if (getattr(o, "side", "") or "").upper() == "SELL"
            and (getattr(o, "order_type", "") or "").upper() in _PROTECTIVE_ORDER_TYPES
        ]

        # Bot-placed = our orders ledger knows the broker_order_id, or the
        # idempotency key carries one of our protective prefixes.
        from app.models.orders import Order
        candidates: list[dict] = []
        with SessionLocal() as db:
            for o in protective:
                boid = o.broker_order_id
                row = (
                    db.query(Order)
                    .filter(Order.broker_order_id == boid)
                    .one_or_none()
                )
                bot_placed = row is not None and (
                    (row.idempotency_key or "").startswith(_BOT_IDEMPOTENCY_PREFIXES)
                    or (row.source or "") in ("scheduler", "consensus", "strategy")
                )
                candidates.append({
                    "symbol": (getattr(o, "symbol", "") or "").upper(),
                    "broker_order_id": boid,
                    "order_type": (getattr(o, "order_type", "") or "").upper(),
                    "stop_price": getattr(o, "stop_price", None),
                    "quantity": getattr(o, "quantity", None),
                    "bot_placed": bot_placed,
                    "action": "cancel" if bot_placed else "manual_review",
                })

        if dry_run:
            return {"dry_run": True, "account_id": account_id,
                    "candidates": candidates}

        # Anchor SELL signals (same source of truth as the reconcile loop) so
        # the trail peak can be re-seeded under the right (symbol, signal_id).
        anchor_by_symbol: dict = {}
        try:
            from app.services.pnl.store import sync_realized_trades
            from app.services.strategy.trail_anchor import (
                assigned_strategy_map, first_assigned_sell_signals,
            )
            syms = sorted({c["symbol"] for c in candidates if c["bot_placed"]})
            with SessionLocal() as db:
                _ins, fifo = sync_realized_trades(db)
                earliest_buy: dict = {}
                for lot in fifo.open_lots:
                    s = lot.symbol.upper()
                    if s not in earliest_buy or lot.buy_at < earliest_buy[s]:
                        earliest_buy[s] = lot.buy_at
                strat_map = assigned_strategy_map(db, syms)
                anchor_by_symbol = first_assigned_sell_signals(
                    db, strat_map, earliest_buy
                )
        except Exception as exc:
            logger.warning("[managed-exits] migrate anchor lookup failed: %s", exc)

        from app.services.audit.service import AuditService
        from app.services.execution.service import ExecutionService
        _audit = AuditService()
        exec_svc = ExecutionService(broker)

        cancelled: list[str] = []
        failed: list[dict] = []
        with SessionLocal() as db:
            for c in candidates:
                if not c["bot_placed"]:
                    continue
                sym = c["symbol"]
                if not is_managed_symbol(sym):
                    continue
                sig = anchor_by_symbol.get(sym)
                stop_level = float(c["stop_price"] or 0.0) or None

                # 1. Seed durable state BEFORE touching the order.
                fields: dict = {"mode": MODE_MONITORING}
                if sig is not None:
                    fields.update(
                        mode="trail_armed",
                        signal_id=sig.id,
                        signal_price=float(sig.price_at_signal),
                        floor=round(float(sig.price_at_signal), 2),
                    )
                if stop_level:
                    fields["target_stop"] = stop_level
                upsert_state(db, sym, **fields)
                db.commit()
                if sig is not None and stop_level:
                    # Carry the resting stop's implied peak into trail_peaks so
                    # the software trail resumes at the same protection level.
                    try:
                        exec_svc._advance_trail_peak(
                            sym, sig.id, float(sig.price_at_signal),
                            candidate_peak=stop_level,
                        )
                    except Exception as exc:
                        logger.debug("[managed-exits] migrate peak seed %s: %s",
                                     sym, exc)

                # 2. Cancel, then 3. verify — only a confirmed cancel counts.
                boid = c["broker_order_id"]
                try:
                    loop.run_until_complete(broker.cancel_order(boid, account_id))
                    verified = False
                    try:
                        st = loop.run_until_complete(broker.get_order(boid, account_id))
                        verified = (getattr(st, "status", "") or "").lower() in _CANCELLED_STATUSES
                    except Exception:
                        pass
                    if verified:
                        cancelled.append(f"{sym}:{boid}")
                        _audit.log(
                            event_type="MANAGED_EXIT_MIGRATION",
                            entity_type="order",
                            description=(
                                f"Cancelled resting {c['order_type']} on {sym} "
                                f"(broker_id={boid}, level={stop_level}) — "
                                f"switched to bot-managed monitoring."
                            ),
                        )
                    else:
                        failed.append({**c, "error": "cancel not verified"})
                except Exception as exc:
                    failed.append({**c, "error": str(exc)})

        try:
            from app.services.notifications.bus import notify_suppression
            notify_suppression(
                symbol="", reason="managed_exit_migration",
                detail=(
                    f"Managed-exits migration: cancelled {len(cancelled)} "
                    f"resting protective order(s) "
                    f"({', '.join(cancelled) or 'none'}); "
                    f"{len(failed)} failed/unverified; "
                    f"{sum(1 for c in candidates if not c['bot_placed'])} "
                    f"manual stop(s) left for your review."
                ),
                source="managed_exits", toast=True,
            )
        except Exception:
            pass

        return {
            "dry_run": False,
            "account_id": account_id,
            "cancelled": cancelled,
            "failed": failed,
            "candidates": candidates,
        }
    finally:
        loop.close()


def _fetch_positions_and_prices() -> tuple[dict[str, float], dict[str, float]]:
    """Standalone broker context (on-demand runs only; the fast job passes its own)."""
    import asyncio as _aio

    from app.services.brokers.factory import get_position_brokers

    loop = _aio.new_event_loop()
    try:
        positions: dict[str, float] = {}
        prices: dict[str, float] = {}
        for b in get_position_brokers():
            try:
                loop.run_until_complete(b.authenticate())
                accts = loop.run_until_complete(b.get_accounts())
                acct = accts[0].account_id if accts else ""
                for p in loop.run_until_complete(b.get_positions(acct)):
                    sym = p.symbol.upper()
                    if sym not in positions and p.quantity:
                        positions[sym] = p.quantity
            except Exception as exc:
                logger.warning("[managed-exits] %s positions fetch failed: %s",
                               getattr(b, "name", "?"), exc)
        held = [s for s, q in positions.items() if q and q >= 1.0]
        for b in get_position_brokers():
            remaining = [s for s in held if s not in prices]
            if not remaining:
                break
            try:
                quotes = loop.run_until_complete(b.get_quotes(remaining))
            except Exception:
                continue
            for s, q in (quotes or {}).items():
                px = getattr(q, "last", None) or getattr(q, "bid", None) or getattr(q, "ask", None)
                if px:
                    prices[s.upper()] = float(px)
        return positions, prices
    finally:
        loop.close()
