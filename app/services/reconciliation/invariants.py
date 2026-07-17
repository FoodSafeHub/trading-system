"""Scheduled invariant check for the automated trading system.

The scheduler already *self-heals* a lot (ledger vs broker max(), orphan-fill
ingestion, re-arming naked positions on the next cycle). This module is the
independent watchdog on top of that: it verifies the invariants directly and
NOTIFIES on any violation, so silent drift becomes an alert instead of a
surprise you find days later in the PnL.

Invariants checked (per enabled-assignment symbol that is actually held):

    1. PROTECTION — every held position has an open protective SELL order
       (STOP / STOP_LIMIT / TRAILING_STOP). A held position with none is
       "naked" and gets a toast — this is the one that costs money.

    2. LEDGER DRIFT — the sum of the per-strategy ledger (StrategyPosition
       rows) for a symbol matches the broker's aggregate quantity within a
       tolerance. Drift means a fill wasn't attributed and future cap/exit
       sizing will be wrong.

This job only READS and NOTIFIES; it never places or cancels orders (the
reconcile/fast-trail jobs own remediation). Fails safe — any error is logged
and swallowed so the watchdog can't take the scheduler down.
"""
from __future__ import annotations

import asyncio
import logging

from app.db import SessionLocal
from app.models.assignments import SymbolStrategyAssignment
from app.models.strategy_positions import StrategyPosition
from app.services.brokers.factory import get_broker

logger = logging.getLogger(__name__)

# Terminal order statuses — an order in one of these is NOT active protection.
_TERMINAL_STATUSES = {
    "FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "REPLACED",
}
# Order types that constitute a protective SELL exit.
_PROTECTIVE_TYPES = {"STOP", "STOP_LIMIT", "TRAILING_STOP"}
# Held-quantity floor: sub-1-share dust isn't a real position to protect.
_HELD_FLOOR = 1.0
# Ledger-vs-broker tolerance (shares) before we call it drift.
_DRIFT_TOL = 0.01


def _brokers_in_use(assignments: list[dict]) -> set[str]:
    """Distinct broker keys across enabled assignments ("default" always in).

    An India symbol left on broker="default" actually trades on Zerodha (the
    order path resolves it there — see factory._assignment_broker_names), so
    the watchdog must scan Zerodha too or that position is invisible here and
    the naked-position check silently skips it.
    """
    brokers = {"default"}
    for a in assignments:
        b = (a.get("broker") or "default").lower()
        brokers.add(b)
        if b == "default":
            try:
                from app.services.markets import is_india_symbol
                if is_india_symbol(a.get("symbol") or ""):
                    brokers.add("zerodha")
            except Exception:
                pass
    return brokers


def _is_protective_open(order) -> bool:
    """True if an order is an active protective SELL (stop/trailing)."""
    side = (getattr(order, "side", "") or "").upper()
    otype = (getattr(order, "order_type", "") or "").upper()
    status = (getattr(order, "status", "") or "").upper()
    if side != "SELL":
        return False
    if otype not in _PROTECTIVE_TYPES:
        return False
    if status in _TERMINAL_STATUSES:
        return False
    return True


def check_invariants(notify: bool = True) -> dict:
    """Run the protection + ledger-drift checks across all held assigned symbols.

    Returns a summary dict:
        {"status": "ok"|"error",
         "checked": N, "naked": [...], "drift": [...], "violations": M}

    When notify=True (default) each violation emits a suppression notification
    (naked positions toast; drift is quiet).
    """
    loop = asyncio.new_event_loop()
    try:
        with SessionLocal() as db:
            assignments = [
                {"symbol": (a.symbol or "").upper(), "broker": (a.broker or "default")}
                for a in db.query(SymbolStrategyAssignment).filter_by(enabled=True).all()
            ]
        assigned_syms = {a["symbol"] for a in assignments if a["symbol"]}
        if not assigned_syms:
            return {"status": "ok", "checked": 0, "naked": [], "drift": [], "violations": 0}

        # Gather per-broker positions + open orders. One broker context per
        # distinct broker key so India (zerodha) and US (default) are both seen.
        # broker_positions: {broker: {symbol: qty}}
        # protected_syms:   set of symbols with an open protective SELL
        broker_positions: dict[str, dict[str, float]] = {}
        protected: set[str] = set()

        for bkey in _brokers_in_use(assignments):
            try:
                broker = get_broker() if bkey == "default" else _build_one(bkey)
                loop.run_until_complete(broker.authenticate())
                accts = loop.run_until_complete(broker.get_accounts())
                aid = accts[0].account_id if accts else ""
                positions = loop.run_until_complete(broker.get_positions(aid))
                broker_positions[bkey] = {
                    p.symbol.upper(): float(p.quantity) for p in positions
                }
                # Open orders — best-effort; some brokers may not support a
                # status filter, so fall back to all orders and filter locally.
                try:
                    orders = loop.run_until_complete(broker.list_orders(aid))
                except Exception as exc:
                    logger.warning("[invariants] list_orders(%s) failed: %s", bkey, exc)
                    orders = []
                for o in orders or []:
                    if _is_protective_open(o):
                        protected.add((getattr(o, "symbol", "") or "").upper())
            except Exception as exc:
                logger.warning("[invariants] broker %s scan failed: %s", bkey, exc)

        # Merge broker positions into one aggregate per symbol (a symbol trades
        # on exactly one broker in practice, so union is safe).
        agg: dict[str, float] = {}
        for _bkey, pmap in broker_positions.items():
            for sym, qty in pmap.items():
                agg[sym] = agg.get(sym, 0.0) + qty

        naked: list[str] = []
        drift: list[dict] = []
        stale_monitoring: list[str] = []
        unexpected_stops: list[str] = []
        engine_stale = False
        checked = 0

        # Bot-managed exits invert invariant 1 for US symbols: protection is
        # the 60s engine's fresh eval (a resting stop would be sweep risk).
        try:
            from app.services.execution.managed_exit_engine import (
                is_managed_symbol as _managed_sym,
                managed_mode_active as _managed_active,
            )
            managed_active = _managed_active()
        except Exception:
            managed_active = False
            _managed_sym = lambda _s: False  # noqa: E731

        with SessionLocal() as db:
            for sym in sorted(assigned_syms):
                held = agg.get(sym, 0.0)
                if held < _HELD_FLOOR:
                    continue  # not actually holding it — nothing to protect
                checked += 1

                # 1. Protection
                if managed_active and _managed_sym(sym):
                    # COVERAGE: the engine must have evaluated this position
                    # recently while its market is open — else it's effectively
                    # unmonitored (the managed-mode equivalent of naked).
                    if _market_open(sym) and not _monitoring_fresh(db, sym):
                        stale_monitoring.append(sym)
                    # INVERSION: any resting protective order IS the sweep risk
                    # managed mode exists to remove (pre-migration leftover or
                    # a manual stop) — surface it for cancellation.
                    if sym in protected:
                        unexpected_stops.append(sym)
                elif sym not in protected:
                    naked.append(sym)

                # 2. Ledger drift — sum StrategyPosition rows for this symbol.
                ledger_sum = sum(
                    float(r.held_qty or 0.0)
                    for r in db.query(StrategyPosition).filter_by(symbol=sym).all()
                )
                if abs(ledger_sum - held) > _DRIFT_TOL:
                    drift.append({"symbol": sym, "ledger": ledger_sum, "broker": held})

        # HEARTBEAT: the engine itself must be alive during market hours —
        # with no resting stops, a dead 60s loop means nothing protects any
        # position (the gap risk the user accepted, surfaced loudly).
        if managed_active and _market_open("SPY"):
            try:
                from app.config import get_settings
                from app.services.execution.managed_exit_engine import (
                    heartbeat_age_seconds,
                )
                with SessionLocal() as db:
                    age = heartbeat_age_seconds(db)
                stale_after = get_settings().managed_exit_stale_after_seconds
                engine_stale = age is None or age > stale_after
            except Exception as exc:
                logger.warning("[invariants] heartbeat check failed: %s", exc)

        if notify:
            _emit(naked, drift, stale_monitoring, unexpected_stops, engine_stale)

        summary = {
            "status": "ok",
            "checked": checked,
            "naked": naked,
            "drift": drift,
            "stale_monitoring": stale_monitoring,
            "unexpected_stops": unexpected_stops,
            "engine_stale": engine_stale,
            "violations": (
                len(naked) + len(drift) + len(stale_monitoring)
                + len(unexpected_stops) + (1 if engine_stale else 0)
            ),
        }
        logger.info(
            "[invariants] checked=%d naked=%s drift=%d stale_mon=%s "
            "unexpected_stops=%s engine_stale=%s",
            checked, naked, len(drift), stale_monitoring,
            unexpected_stops, engine_stale,
        )
        return summary
    except Exception as exc:
        logger.error("[invariants] check failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}
    finally:
        loop.close()


def _market_open(symbol: str) -> bool:
    """This symbol's own session open? Fail-open (True) so a lookup blip can't
    suppress a real staleness alert during trading hours."""
    try:
        from app.services.strategy.scheduler import _symbol_market_open
        return _symbol_market_open(symbol)
    except Exception:
        return True


def _monitoring_fresh(db, symbol: str) -> bool:
    """True when the managed-exit engine evaluated this symbol recently."""
    from datetime import datetime, timezone

    from app.config import get_settings
    from app.models.managed_exit_state import ManagedExitState

    st = (
        db.query(ManagedExitState)
        .filter(ManagedExitState.symbol == symbol.upper())
        .one_or_none()
    )
    if st is None or st.last_eval_at is None:
        return False
    last = st.last_eval_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age <= get_settings().managed_exit_stale_after_seconds


def _emit(
    naked: list[str],
    drift: list[dict],
    stale_monitoring: list[str] | None = None,
    unexpected_stops: list[str] | None = None,
    engine_stale: bool = False,
) -> None:
    """Best-effort notifications for the violations found."""
    try:
        from app.services.notifications.bus import notify_suppression
    except Exception:
        return
    for sym in stale_monitoring or []:
        try:
            notify_suppression(
                symbol=sym, direction="SELL", reason="monitoring_stale",
                detail=(f"{sym} is held under bot-managed exits but the 60s "
                        f"engine hasn't evaluated it recently — position is "
                        f"effectively unmonitored (no resting stop by design)."),
                source="invariants", toast=True,
            )
        except Exception:
            pass
    for sym in unexpected_stops or []:
        try:
            notify_suppression(
                symbol=sym, direction="SELL", reason="unexpected_resting_stop",
                detail=(f"{sym} has a resting protective order but bot-managed "
                        f"exits expect NONE (sweep risk). The next trail pass "
                        f"cancels bot-placed ones; cancel manual stops yourself."),
                source="invariants", toast=True,
            )
        except Exception:
            pass
    if engine_stale:
        try:
            notify_suppression(
                symbol="", direction=None, reason="managed_exits_engine_stale",
                detail=("Managed-exit engine heartbeat is STALE during market "
                        "hours — NO position is being monitored and no stops "
                        "rest at the broker. Check the scheduler/bot host."),
                source="invariants", toast=True,
            )
        except Exception:
            pass
    for sym in naked:
        try:
            notify_suppression(
                symbol=sym, direction="SELL", reason="naked_position",
                detail=(f"{sym} is held with NO protective stop/trail — position "
                        f"unprotected. Reconcile should re-arm; investigate if it persists."),
                source="invariants", toast=True,
            )
        except Exception:
            pass
    for d in drift:
        try:
            notify_suppression(
                symbol=d["symbol"], direction=None, reason="ledger_drift",
                detail=(f"{d['symbol']} ledger {d['ledger']:.4f} sh vs broker "
                        f"{d['broker']:.4f} sh — a fill wasn't attributed; "
                        f"cap/exit sizing may be off."),
                source="invariants", toast=False,
            )
        except Exception:
            pass


def _build_one(name: str):
    from app.services.brokers.factory import _build_one as _bo
    return _bo(name)
