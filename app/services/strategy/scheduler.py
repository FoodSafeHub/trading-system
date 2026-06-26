from __future__ import annotations

"""
Strategy scheduler — runs the strategy evaluation cycle periodically during market hours.
Uses APScheduler with an in-process background scheduler.
Overlap prevention: if a prior cycle is still running, the new trigger is skipped.
"""

import asyncio
import logging
import threading
from collections import defaultdict

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings
from app.db import SessionLocal
from app.models.signals import Signal
from app.services.brokers.factory import _build_one, get_broker
from app.services.execution.service import ExecutionService
from app.services.market_data.provider import get_ohlcv, get_price_series
from app.services.risk.engine import RiskEngine
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.engine import StrategyEngine, load_strategies_from_config
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES, run_perplexity_signal
from app.utils.time_utils import is_market_hours

logger = logging.getLogger(__name__)

_engine = StrategyEngine()
_risk = RiskEngine()
_lock = threading.Lock()
_running = False

# Runtime overrides — set via API without restarting server.
# None means "fall back to .env / config.py value".
_override_run_bollinger: bool | None = None
_override_run_perplexity: bool | None = None


def _persist_signal(
    symbol: str,
    direction: str,
    strategy_label: str,
    entry: float | None,
    *,
    acted_on: bool = True,
) -> int | None:
    """Write a Signal row so the resulting Order can join back to a strategy name.

    acted_on=False writes an observation-only row (HOLD or a signal that was
    evaluated but blocked before order placement). acted_on=True marks signals
    that reached the broker execution path.
    """
    try:
        with SessionLocal() as db:
            sig = Signal(
                strategy_name=strategy_label[:128],
                symbol=symbol.upper(),
                direction=direction,
                strength=1.0,
                price_at_signal=entry,
                acted_on=acted_on,
            )
            db.add(sig)
            db.commit()
            db.refresh(sig)
            return sig.id
    except Exception as exc:
        logger.warning("[scheduler] Could not persist Signal row for %s/%s: %s", symbol, strategy_label, exc)
        return None


def _pending_signal_id(symbol: str, direction: str, strategy_label: str) -> int | None:
    """Return the most recent un-acted matching signal row id WITHOUT marking it.

    Used to link a signal to the order it triggers before we know whether the
    order placement succeeded. Only flip acted_on=True (via
    _mark_signal_acted_on) once the action actually completes, so a failed
    placement leaves the signal eligible for the reconcile retry.
    """
    try:
        with SessionLocal() as db:
            sig = (
                db.query(Signal)
                .filter(
                    Signal.symbol == symbol.upper(),
                    Signal.direction == direction,
                    Signal.strategy_name == strategy_label[:128],
                    Signal.acted_on == False,  # noqa: E712
                )
                .order_by(Signal.id.desc())
                .first()
            )
            return sig.id if sig else None
    except Exception as exc:
        logger.warning("[scheduler] Could not resolve pending signal for %s/%s: %s", symbol, strategy_label, exc)
    return None


def _mark_signal_acted_on(symbol: str, direction: str, strategy_label: str) -> int | None:
    """Flip the most recent matching signal row to acted_on=True and return its id."""
    try:
        with SessionLocal() as db:
            sig = (
                db.query(Signal)
                .filter(
                    Signal.symbol == symbol.upper(),
                    Signal.direction == direction,
                    Signal.strategy_name == strategy_label[:128],
                    Signal.acted_on == False,  # noqa: E712
                )
                .order_by(Signal.id.desc())
                .first()
            )
            if sig:
                sig.acted_on = True
                db.commit()
                return sig.id
    except Exception as exc:
        logger.warning("[scheduler] Could not mark signal acted_on for %s/%s: %s", symbol, strategy_label, exc)
    return None


def _quantize_for_broker(shares: float) -> float:
    """Round shares to a quantity the active broker will accept.

    Schwab cash accounts reject fractional orders ("No trades are currently allowed")
    unless the user has explicitly enabled Stock Slices. Paper broker accepts any qty.
    To stay safe by default, we floor to whole shares for live brokers.
    """
    settings = get_settings()
    if settings.active_broker == "paper":
        return max(round(shares, 6), 0.001)
    whole = int(shares)
    return float(whole) if whole >= 1 else 0.0


def _reconcile_trail_stops(
    loop, broker, account_id: str,
    current_positions: dict[str, float],
    assignments: list[dict],
    live_prices: dict[str, float],
    svc_for,
) -> None:
    """Bring every held position into the correct tight-trail state.

    For each held position whose ASSIGNED strategy fired a SELL signal (the
    scanner-stream signal) at/after acquisition, call tighten_trail_on_sell —
    which implements Approach C end-to-end: it holds off until price clears the
    arm gate (signal + 0.25%), then places/ratchets a bot-managed STOP floored
    at signal + 0.25% so a reversal still exits in profit.

    Idempotency/ratchet lives INSIDE tighten_trail_on_sell now: it re-computes
    the floored target each cycle and only cancel/replaces when the new stop is
    materially higher than the resting one, so calling it every cycle is safe and
    keeps the stop climbing. We no longer reason about native TRAILING_STOP vs
    static STOP here — the bot manages a floored STOP on every broker so the
    hard floor is guaranteed (a native % trail could sit below it).

    Uses first_assigned_sell_signals so the anchor matches the PnL table exactly
    (first scanner-stream SELL from the assigned strategy, after acquisition).
    """
    from app.services.pnl.store import sync_realized_trades
    from app.services.strategy.trail_anchor import (
        assigned_strategy_map, assigned_trail_pct_map, first_assigned_sell_signals,
    )

    held = {s.upper(): q for s, q in current_positions.items() if q and q >= 1.0}
    if not held:
        return
    symbols = list(held.keys())

    with SessionLocal() as db:
        # FIFO earliest-buy per held symbol (anchor cutoff for the SELL signal).
        try:
            _ins, fifo = sync_realized_trades(db)
            earliest_buy = {}
            for lot in fifo.open_lots:
                sym = lot.symbol.upper()
                if sym not in earliest_buy or lot.buy_at < earliest_buy[sym]:
                    earliest_buy[sym] = lot.buy_at
        except Exception as exc:
            logger.warning("[scheduler] Trail reconcile: FIFO build failed: %s", exc)
            earliest_buy = {}

        assigned_strat = assigned_strategy_map(db, symbols)
        trail_pcts = assigned_trail_pct_map(db, symbols)
        anchor_sig = first_assigned_sell_signals(db, assigned_strat, earliest_buy)

        for symbol, qty in held.items():
            sig = anchor_sig.get(symbol)
            if sig is None:
                continue  # assigned strategy hasn't flagged an exit — leave it

            asgn = next((a for a in assignments if a["symbol"].upper() == symbol), None)
            trail_pct = trail_pcts.get(symbol, 2.0)
            sig_price = float(sig.price_at_signal)
            exec_svc, exec_acct = svc_for((asgn or {}).get("broker", "default") or "default")

            # tighten_trail_on_sell is the single source of Approach C: arm gate,
            # hard floor, and per-cycle ratchet all live inside it and are
            # idempotent, so we always call it (no force_replace needed). It
            # holds off below the arm gate and only ratchets the floored STOP up.
            logger.info(
                "[scheduler] Trail reconcile: %s held=%.4f assigned SELL @ $%.2f "
                "(%s) — evaluating %.1f%% floored trail.",
                symbol, qty, sig_price, sig.strategy_name, trail_pct,
            )
            try:
                ok = loop.run_until_complete(exec_svc.tighten_trail_on_sell(
                    symbol=symbol,
                    quantity=qty,
                    account_id=exec_acct,
                    signal_price=sig_price,
                    signal_at=sig.created_at,
                    trail_pct=trail_pct,
                    source="scheduler",
                    idempotency_suffix=f"reconcile-{symbol}-{int(sig_price * 100)}",
                    signal_id=sig.id,
                ))
                if ok and not sig.acted_on:
                    sig.acted_on = True
                    db.commit()
            except Exception as exc:
                logger.warning("[scheduler] Trail reconcile failed for %s: %s", symbol, exc)


def reconcile_trails_now() -> dict:
    """Standalone trail reconciliation — sets up its own broker context and runs
    _reconcile_trail_stops across every held position.

    Callable on demand (API endpoint / manual trigger) independent of the
    15-min scheduler cycle. Returns a summary dict for the caller.
    """
    import asyncio as _aio
    from app.models.assignments import SymbolStrategyAssignment

    loop = _aio.new_event_loop()
    try:
        broker = get_broker()
        loop.run_until_complete(broker.authenticate())
        accounts = loop.run_until_complete(broker.get_accounts())
        account_id = accounts[0].account_id if accounts else ""

        positions = loop.run_until_complete(broker.get_positions(account_id))
        current_positions = {p.symbol.upper(): p.quantity for p in positions}

        with SessionLocal() as db:
            assignments = [
                {"symbol": a.symbol, "broker": a.broker or "default",
                 "tight_trail_pct": a.tight_trail_pct}
                for a in db.query(SymbolStrategyAssignment).filter_by(enabled=True).all()
            ]

        # Live quotes for the held symbols (best-effort; reconcile reads them).
        live_prices: dict[str, float] = {}
        try:
            held_syms = [s for s, q in current_positions.items() if q and q >= 1.0]
            if held_syms:
                quotes = loop.run_until_complete(broker.get_quotes(held_syms))
                for s, q in (quotes or {}).items():
                    px = getattr(q, "last", None) or getattr(q, "bid", None) or getattr(q, "ask", None)
                    if px:
                        live_prices[s.upper()] = float(px)
        except Exception:
            pass

        # Per-assignment broker routing (mirror the cycle's _svc_for).
        svc_cache: dict[str, tuple] = {"default": (ExecutionService(broker), account_id)}

        def _svc_for(name: str):
            key = (name or "default").lower()
            if key in svc_cache:
                return svc_cache[key]
            try:
                b = _build_one(key)
                loop.run_until_complete(b.authenticate())
                accts = loop.run_until_complete(b.get_accounts())
                aid = accts[0].account_id if accts else ""
                pair = (ExecutionService(b), aid)
                svc_cache[key] = pair
                return pair
            except Exception as exc:
                logger.warning("[scheduler] reconcile _svc_for(%s) failed: %s — using default", name, exc)
                return svc_cache["default"]

        _reconcile_trail_stops(
            loop, broker, account_id,
            current_positions, assignments, live_prices, _svc_for,
        )
        n_held = sum(1 for q in current_positions.values() if q and q >= 1.0)
        return {"status": "ok", "held_positions": n_held}
    except Exception as exc:
        logger.error("[scheduler] reconcile_trails_now failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}
    finally:
        loop.close()


def _live_position_state(symbol: str, df, held_qty: float):
    """Build the PositionState the trailing-stop overlay needs for a LIVE bar.

    Unlike the backtest (which tracks entry/peak in-loop), the live path sees one
    cycle at a time, so we reconstruct from order history: entry_price = the most
    recent filled BUY's fill price; highest_close = max close from that fill date
    forward in the fetched df. Returns None when flat or when we can't establish
    an entry, so the overlay no-ops and the rule's fixed-band exit stands.
    """
    if held_qty <= 0 or df is None or df.empty:
        return None
    from app.models.orders import Order
    from app.services.strategy.rules import PositionState
    try:
        with SessionLocal() as db:
            last_buy = (
                db.query(Order)
                .filter(Order.symbol == symbol.upper(), Order.side == "BUY",
                        Order.status == "filled")
                .order_by(Order.created_at.desc())
                .first()
            )
        if last_buy is None or not last_buy.fill_price:
            return None
        entry_price = float(last_buy.fill_price)
        # Peak close since entry: slice df from the fill date forward.
        peak = entry_price
        fill_dt = last_buy.filled_at or last_buy.created_at
        try:
            since = df.loc[str(fill_dt)[:10]:]
            if not since.empty:
                peak = max(entry_price, float(since["Close"].max()))
            else:
                peak = max(entry_price, float(df["Close"].iloc[-1]))
        except Exception:
            peak = max(entry_price, float(df["Close"].iloc[-1]))
        return PositionState(entry_price=entry_price, highest_close=peak)
    except Exception as exc:
        logger.warning("[scheduler] could not build position state for %s: %s", symbol, exc)
        return None


def _compute_quantity(
    symbol: str,
    entry: float,
    stop: float | None,
    max_capital_usd: float | None = None,
    max_shares: float | None = None,
    held_qty: float = 0.0,
) -> float:
    """
    Return shares to BUY using fixed-fractional position sizing, honouring the
    user's per-assignment cap **inclusive of shares already held**.

    The cap (max_capital_usd or max_shares) defines the TOTAL allowed position,
    not the size of a single order. If you already hold some shares of the
    symbol, this function returns only the gap up to the cap; if you're already
    at or over the cap, it returns 0 (the caller will then skip the BUY).

    max_capital_usd: per-symbol DOLLAR cap. When set, total notional
                     (held + new) is capped at this amount.
    max_shares:      per-symbol SHARES cap. Used only when max_capital_usd is
                     empty — dollar cap wins whenever both are set.
    held_qty:        current holding for this symbol (from the broker). The
                     scheduler passes this through so caps are honoured across
                     multiple BUY signals on the same symbol; previously a
                     top-up could pyramid past the user's cap.

    Returns 0 when adding any shares would exceed the cap.
    """
    # Resolve TOTAL shares this assignment is allowed to hold (incl. existing).
    total_allowed: float | None = None
    if max_capital_usd and entry > 0:
        total_allowed = max_capital_usd / entry
    elif max_shares and max_shares > 0:
        total_allowed = max_shares

    if stop is None or stop <= 0 or stop >= entry:
        # No usable stop — size by the cap if there is one; otherwise 1 share.
        if total_allowed is not None:
            gap = max(0.0, total_allowed - max(0.0, held_qty))
            return _quantize_for_broker(gap) if gap > 0 else 0.0
        # No cap and no stop → default 1 share, still gap-aware.
        gap = max(0.0, 1.0 - max(0.0, held_qty))
        return _quantize_for_broker(gap) if gap > 0 else 0.0

    settings = get_settings()
    # Dollar cap wins. When it's absent and a shares cap is set, derive a
    # dollar-equivalent cap from max_shares * entry so the risk sizer can still
    # honour the user's intent.
    if max_capital_usd:
        effective_max = max_capital_usd
    elif max_shares and max_shares > 0 and entry > 0:
        effective_max = max_shares * entry
    else:
        effective_max = settings.max_position_size_usd
    # Tighten effective_max by the dollar value of shares already held so the
    # risk sizer can never propose a quantity that would push us past the cap.
    held_value = max(0.0, held_qty) * entry if entry > 0 else 0.0
    effective_max_for_new = max(0.0, effective_max - held_value)
    if effective_max_for_new <= 0.0:
        logger.info(
            "[scheduler] Position size %s: already at/over cap "
            "(held=%.4f @ $%.2f = $%.0f vs cap $%.0f) — no top-up.",
            symbol, held_qty, entry, held_value, effective_max,
        )
        return 0.0
    try:
        sz = calculate_position_size(
            symbol=symbol,
            entry_price=entry,
            stop_price=stop,
            account_value=settings.account_value,
            risk_pct_per_trade=settings.risk_pct_per_trade,
            max_position_size_usd=effective_max_for_new,
            max_account_risk_pct=settings.max_account_risk_pct,
        )
        if sz.viable and sz.shares >= 0.001:
            # Also clamp by shares cap if one exists, accounting for held.
            qty = sz.shares
            if total_allowed is not None:
                gap = max(0.0, total_allowed - max(0.0, held_qty))
                qty = min(qty, gap)
            qty = _quantize_for_broker(qty)
            logger.info(
                f"[scheduler] Position size {symbol}: raw={sz.shares:.4f} -> qty={qty} "
                f"@ ${entry:.2f}, stop ${stop:.2f}, risk ${sz.risk_amount:.2f}, "
                f"cap=${effective_max:.0f}, held={held_qty:.4f}, gap_cap=${effective_max_for_new:.0f}"
            )
            return qty
    except Exception as exc:
        logger.warning(f"[scheduler] Position sizing failed for {symbol}: {exc}")
    # Sizer fallback: still honour held vs total_allowed if a cap exists.
    if total_allowed is not None:
        gap = max(0.0, total_allowed - max(0.0, held_qty))
        return _quantize_for_broker(gap) if gap > 0 else 0.0
    gap = max(0.0, 1.0 - max(0.0, held_qty))
    return _quantize_for_broker(gap) if gap > 0 else 0.0


def set_scheduler_system_flags(run_bollinger: bool | None, run_perplexity: bool | None) -> None:
    global _override_run_bollinger, _override_run_perplexity
    if run_bollinger is not None:
        _override_run_bollinger = run_bollinger
    if run_perplexity is not None:
        _override_run_perplexity = run_perplexity


def _bollinger_enabled() -> bool:
    if _override_run_bollinger is not None:
        return _override_run_bollinger
    return get_settings().scheduler_run_bollinger


def _perplexity_enabled() -> bool:
    if _override_run_perplexity is not None:
        return _override_run_perplexity
    return get_settings().scheduler_run_perplexity


def _run_cycle(*, force: bool = False, dry_run: bool = False) -> None:
    """Run one strategy evaluation + order cycle.

    force=True   — skips the market-hours gate so a manual run works any time.
    dry_run=True — evaluates all strategies and writes signal rows but skips
                   every broker call. Safe to run while the live scheduler is
                   also running; does NOT touch the kill switch so there is no
                   race with concurrent cycles.
    """
    global _running
    if not _lock.acquire(blocking=False):
        logger.debug("[scheduler] Previous cycle still running — skipping")
        return
    _running = True
    try:
        settings = get_settings()

        if not force and not _any_market_open():
            logger.debug("[scheduler] Outside market hours (US + India) — skipping cycle")
            return

        if not force and not dry_run and _risk.is_kill_switch_active():
            logger.warning("[scheduler] Kill switch active — skipping cycle")
            return

        settings = get_settings()

        loop = asyncio.new_event_loop()
        try:
            # Dry-run: skip all broker I/O — we only need signal evaluation.
            if dry_run:
                broker = None
                account_id = ""
                svc = None

                def _svc_for(name: str) -> tuple[None, str]:  # type: ignore[misc]
                    return None, ""
            else:
                broker = get_broker()
                loop.run_until_complete(broker.authenticate())
                accounts = loop.run_until_complete(broker.get_accounts())
                account_id = accounts[0].account_id if accounts else ""
                svc = ExecutionService(broker)

                # Lazy per-broker execution-service cache for per-assignment broker
                # overrides. "default" reuses the global svc / account_id built
                # above. Anything else builds (authenticates, account lookup) on
                # first hit this cycle, then is reused for subsequent symbols.
                broker_svc_cache: dict[str, tuple[ExecutionService, str]] = {
                    "default": (svc, account_id),
                }

                def _svc_for(name: str) -> tuple[ExecutionService, str]:  # type: ignore[misc]
                    key = (name or "default").lower()
                    if key in broker_svc_cache:
                        return broker_svc_cache[key]
                    try:
                        b = _build_one(key)
                        loop.run_until_complete(b.authenticate())
                        accts = loop.run_until_complete(b.get_accounts())
                        aid = accts[0].account_id if accts else ""
                        pair = (ExecutionService(b), aid)
                        broker_svc_cache[key] = pair
                        return pair
                    except Exception as exc:
                        logger.warning(
                            "[scheduler] Broker override %r failed (%s) — falling back to default",
                            key, exc,
                        )
                        return broker_svc_cache["default"]

            from app.models.assignments import SymbolStrategyAssignment
            from app.schemas.orders import OrderRequest

            # ── Load per-symbol strategy assignments ─────────────
            with SessionLocal() as db:
                active_assignments = (
                    db.query(SymbolStrategyAssignment)
                    .filter_by(enabled=True)
                    .all()
                )
                assignments = [
                    # Normalize the symbol to uppercase at the source. current_positions
                    # and live_prices are keyed by UPPERCASE symbol (broker convention),
                    # but the held/price lookups below use asgn["symbol"] verbatim — so a
                    # lowercase/mixed-case assignment row (e.g. "aapl") silently missed its
                    # position, making SELL exits skip ("held=0.0") and BUY sizing pyramid
                    # past the cap. Uppercasing here keeps every downstream lookup aligned.
                    {"symbol": (a.symbol or "").upper(), "system": a.system, "strategy_name": a.strategy_name,
                     "max_capital_usd": a.max_capital_usd, "max_shares": a.max_shares,
                     "broker": a.broker or "default",
                     # Per-assignment Approach C trail %. None = system default (2.0%).
                     "tight_trail_pct": a.tight_trail_pct}
                    for a in active_assignments
                ]

            assigned_symbols = {a["symbol"] for a in assignments}

            # ── Fetch Schwab live prices for all assigned symbols ─
            all_symbols = list(assigned_symbols)
            live_prices: dict[str, float] = {}
            if all_symbols and not dry_run:
                try:
                    quotes = loop.run_until_complete(broker.get_quotes(all_symbols))
                    for sym, q in quotes.items():
                        price = q.last or q.ask or q.bid
                        if price and price > 0:
                            live_prices[sym] = float(price)
                    logger.info("[scheduler] Schwab live prices: %d/%d symbols", len(live_prices), len(all_symbols))
                except Exception as exc:
                    logger.warning("[scheduler] Schwab quote fetch failed, using yfinance: %s", exc)

            # ── Fetch current positions to size SELL orders correctly ─
            current_positions: dict[str, float] = {}
            if not dry_run:
                try:
                    positions = loop.run_until_complete(broker.get_positions(account_id))
                    for pos in positions:
                        current_positions[pos.symbol.upper()] = pos.quantity
                except Exception as exc:
                    logger.warning("[scheduler] Could not fetch positions: %s", exc)

            # ── Regime-aware open-position cap ──────────────────────────
            # Resolve the max number of distinct holdings the current market
            # regime allows (bull 10 / bear 3 / deep-bear 1, from settings).
            # Computed once per cycle: get_current_regime does a benchmark
            # fetch, so we don't want it per-order. A BUY that would open a
            # NEW symbol is blocked once we're at the cap; top-ups to symbols
            # we already hold don't count (they don't add a position). Fails
            # open — if regime detection errors, no cap is imposed.
            max_open_positions: int | None = None
            try:
                from datetime import datetime as _dt, timezone as _tz
                from app.services.market_regime import get_current_regime, get_regime_risk_caps
                _regime = get_current_regime(_dt.now(_tz.utc))
                max_open_positions = int(get_regime_risk_caps(_regime)["max_positions"])
                logger.info(
                    "[scheduler] Regime %s → max open positions %d (currently holding %d)",
                    getattr(_regime, "value", _regime), max_open_positions,
                    sum(1 for q in current_positions.values() if q > 0),
                )
            except Exception as exc:
                logger.warning("[scheduler] Regime cap unavailable, no position cap this cycle: %s", exc)

            # signals_to_act: list of
            #   (symbol, direction, label, entry_price, stop_price, system, strategy_name)
            # system + strategy_name identify the specific assignment that produced the
            # signal, so the execute loop can size/sell against THAT strategy's share
            # ledger (a symbol may now carry several independent assignments).
            signals_to_act: list[
                tuple[str, str, str, float, float | None, str, str]
            ] = []

            # ── 1. Run assigned strategies ───────────────────────
            for asgn in assignments:
                symbol = asgn["symbol"]
                system = asgn["system"]
                strategy_name = asgn["strategy_name"]

                try:
                    if system == "perplexity":
                        strat = next(
                            (s for s in PERPLEXITY_STRATEGIES if s.name == strategy_name), None
                        )
                        if strat is None or not strat.enabled:
                            continue
                        df = get_ohlcv(symbol, period="2y")
                        if df.empty or len(df) < 60:
                            continue
                        live = live_prices.get(symbol)
                        if live:
                            df = df.copy()
                            df.iloc[-1, df.columns.get_loc("Close")] = live
                        sig = strat.run(symbol, df)
                        entry = live or sig.entry_price or float(df["Close"].iloc[-1])
                        label = f"perplexity:{strategy_name}"
                        _persist_signal(symbol, sig.direction, label, entry, acted_on=False)
                        if sig.direction != "HOLD":
                            signals_to_act.append((symbol, sig.direction, label, entry, sig.stop_price, system, strategy_name))
                            logger.info(
                                "[scheduler] Assigned %s → %s: %s entry=%.2f stop=%s (%s)",
                                strategy_name, symbol, sig.direction, entry, sig.stop_price, sig.reason
                            )

                    elif system == "bollinger":
                        bollinger_configs = load_strategies_from_config()
                        matching = [c for c in bollinger_configs if c.name == strategy_name and c.enabled]
                        for config in matching:
                            prices = get_price_series(symbol, period="1y")
                            sigs = _engine.run(config, prices)
                            for s in sigs:
                                entry = live_prices.get(symbol) or s.price_at_signal or float(prices.iloc[-1])
                                _persist_signal(symbol, s.direction, strategy_name, entry, acted_on=False)
                                if s.direction != "HOLD":
                                    signals_to_act.append((symbol, s.direction, strategy_name, entry, None, system, strategy_name))
                                    logger.info("[scheduler] Assigned %s → %s: %s", strategy_name, symbol, s.direction)

                    elif system == "scanner":
                        from app.services.scanner.scanner_service import _make_generic_configs_full
                        from app.services.strategy.rules import evaluate_strategy

                        df = get_ohlcv(symbol, period="1y")
                        if df.empty or len(df) < 60:
                            logger.info(
                                "[scheduler] scanner-assigned %s skipped — insufficient data (len=%d)",
                                symbol, len(df),
                            )
                        else:
                            generic = _make_generic_configs_full(symbol)
                            match = next((c for c in generic if c.name == strategy_name and c.enabled), None)
                            if match is None:
                                logger.warning(
                                    "[scheduler] scanner-assigned %s: strategy %r not in generic set — "
                                    "remove or rename the assignment",
                                    symbol, strategy_name,
                                )
                            else:
                                live = live_prices.get(symbol)
                                if live:
                                    df = df.copy()
                                    df.iloc[-1, df.columns.get_loc("Close")] = live
                                prices = df["Close"].dropna()
                                pos_state = _live_position_state(
                                    symbol, df, current_positions.get(symbol.upper(), 0.0)
                                )
                                sig = evaluate_strategy(
                                    match.type, symbol, prices, match.params,
                                    ohlcv=df, position=pos_state,
                                )
                                entry = live or sig.price_at_signal or float(prices.iloc[-1])
                                label = f"scanner:{strategy_name}"
                                _persist_signal(symbol, sig.direction, label, entry, acted_on=False)
                                if sig.direction != "HOLD":
                                    signals_to_act.append((symbol, sig.direction, label, entry, None, system, strategy_name))
                                    logger.info(
                                        "[scheduler] Assigned scanner %s → %s: %s entry=%.2f",
                                        strategy_name, symbol, sig.direction, entry,
                                    )

                except Exception as exc:
                    logger.error("[scheduler] Assigned strategy %s on %s failed: %s", strategy_name, symbol, exc)

            # ── 2. Run general pool (consensus) for non-assigned symbols ─
            votes: dict = defaultdict(lambda: defaultdict(list))
            min_agree = int(settings.min_signal_agreement)

            if _bollinger_enabled():
                bollinger_configs = load_strategies_from_config()
                for config in bollinger_configs:
                    # assigned_symbols is uppercase; compare case-insensitively so an
                    # assigned symbol can never also enter the consensus pool (which
                    # would double-trade it via two paths with different SELL policies).
                    if not config.enabled or (config.symbol or "").upper() in assigned_symbols:
                        continue
                    try:
                        prices = get_price_series(config.symbol, period="1y")
                        sigs = _engine.run(config, prices)
                        for s in sigs:
                            if s.direction != "HOLD":
                                votes[(s.symbol or "").upper()][s.direction].append(config.name)
                    except Exception as exc:
                        logger.error("[scheduler] Bollinger %s failed: %s", config.name, exc)

            if _perplexity_enabled():
                # Use symbols from assignments or bollinger configs as the pool universe
                try:
                    pool_symbols = list({c.symbol for c in load_strategies_from_config()})
                except Exception:
                    pool_symbols = []
                for symbol in pool_symbols:
                    if (symbol or "").upper() in assigned_symbols:
                        continue
                    try:
                        df = get_ohlcv(symbol, period="2y")
                        if df.empty or len(df) < 60:
                            continue
                        pool_sigs = run_perplexity_signal(symbol, df)
                        for sig in pool_sigs:
                            if sig.direction != "HOLD":
                                votes[(sig.symbol or "").upper()][sig.direction].append(
                                    f"perplexity:{sig.strategy_name}"
                                )
                    except Exception as exc:
                        logger.error("[scheduler] Perplexity pool %s failed: %s", symbol, exc)

            # ── 3. Execute assigned signals (no consensus needed) ─
            if dry_run:
                logger.info("[scheduler] Dry run — %d assigned signal(s) evaluated, no orders placed.", len(signals_to_act))
            from app.services.strategy import strategy_ledger
            for symbol, direction, label, entry, stop, sig_system, sig_strategy in (
                [] if dry_run else signals_to_act
            ):
                # Resolve the exact assignment that fired this signal (a symbol may
                # carry several). Falls back to the first symbol match only if the
                # triple lookup misses (shouldn't happen).
                asgn = next(
                    (a for a in assignments
                     if a["symbol"] == symbol
                     and a["system"] == sig_system
                     and a["strategy_name"] == sig_strategy),
                    None,
                ) or next((a for a in assignments if a["symbol"] == symbol), None)
                asgn_cap = asgn["max_capital_usd"] if asgn else None
                asgn_shares = asgn["max_shares"] if asgn else None
                asgn_broker = asgn["broker"] if asgn else "default"
                # Auto-route by market: an India (NSE/BSE) symbol with no explicit
                # broker override goes to Zerodha rather than the US-oriented global
                # toggle, so US and India orders fire in their own sessions.
                if asgn_broker == "default":
                    try:
                        from app.services.markets import is_india_symbol
                        if is_india_symbol(symbol):
                            asgn_broker = "zerodha"
                    except Exception:
                        pass
                if direction == "BUY":
                    # Held-qty for the cap. Start from THIS strategy's ledger lot so
                    # several strategies on one symbol each size against their own
                    # cap. BUT when the symbol has only ONE assignment, the broker
                    # aggregate IS that strategy's lot — so take max(ledger, broker)
                    # to self-heal if the ledger lagged or an older fill was never
                    # attributed (otherwise a held=0 read re-buys the full cap every
                    # cycle and pyramids past max_capital_usd / max_shares).
                    strat_held = strategy_ledger.get_held(
                        symbol, sig_system, sig_strategy, asgn_broker
                    )
                    n_asgn_for_symbol = sum(
                        1 for a in assignments if a["symbol"] == symbol
                    )
                    if n_asgn_for_symbol <= 1:
                        strat_held = max(strat_held, current_positions.get(symbol, 0.0))
                    qty = _compute_quantity(
                        symbol, entry, stop, asgn_cap, asgn_shares,
                        held_qty=strat_held,
                    )
                    if qty <= 0:
                        logger.info(
                            "[scheduler] BUY %s (%s:%s) skipped — at/over cap "
                            "(strat_held=%.4f, cap=%s, shares=%s, entry=%.2f).",
                            symbol, sig_system, sig_strategy, strat_held,
                            asgn_cap, asgn_shares, entry,
                        )
                        continue
                    # Regime open-position cap: counts DISTINCT symbols, so it uses
                    # the broker aggregate (a symbol is one position regardless of how
                    # many strategies hold it). Only blocks a BUY that would open a
                    # brand-new symbol — if anything already holds it, this is a top-up.
                    held = current_positions.get(symbol, 0.0)
                    if max_open_positions is not None and held <= 0:
                        open_count = sum(1 for q in current_positions.values() if q > 0)
                        if open_count >= max_open_positions:
                            logger.info(
                                "[scheduler] BUY %s skipped — at regime position cap "
                                "(%d/%d open). New entries blocked until a slot frees.",
                                symbol, open_count, max_open_positions,
                            )
                            continue
                else:
                    # SELL signal: only sell THIS strategy's lot, clamped to what the
                    # broker actually shows. The broker reports one aggregate position
                    # per symbol; selling all of it would dump another strategy's
                    # shares too. min(ledger, broker) keeps each strategy's exit to its
                    # own lot, and never oversells if the ledger drifted high.
                    broker_held = current_positions.get(symbol, 0.0)
                    strat_held = strategy_ledger.get_held(
                        symbol, sig_system, sig_strategy, asgn_broker
                    )
                    qty = min(strat_held, broker_held)
                    if qty < 1.0:
                        logger.info(
                            "[scheduler] SELL %s (%s:%s) skipped — sellable=%.4f "
                            "(strat_ledger=%.4f, broker_held=%.4f, need >= 1.0)",
                            symbol, sig_system, sig_strategy, qty,
                            strat_held, broker_held,
                        )
                        continue

                    # ── Assigned strategy SELL → tight trailing stop ─────────
                    # Only the ASSIGNED strategy for this symbol can trigger
                    # the tight trail. Trail % comes from the assignment row
                    # (set during backtesting via the Promote UI). Falls back
                    # to 2.0% when not explicitly configured.
                    exec_svc, exec_acct = _svc_for(asgn_broker)
                    trail_pct = float(asgn.get("tight_trail_pct") or 2.0)

                    # Resolve (don't yet consume) the signal id so we can link it
                    # to the trail order. We only mark it acted_on if the trail
                    # actually gets placed — otherwise the position would be left
                    # naked AND the signal flagged done, so the reconcile job
                    # (which re-arms unprotected held positions with a recent SELL)
                    # could skip it. Leaving acted_on=False keeps it eligible.
                    sig_id = _pending_signal_id(symbol, direction, label)

                    logger.info(
                        "[scheduler] SELL signal %s: placing %.1f%% tight trail "
                        "(assignment trail_pct=%s)",
                        symbol, trail_pct, asgn.get("tight_trail_pct"),
                    )
                    trail_ok = loop.run_until_complete(exec_svc.tighten_trail_on_sell(
                        symbol=symbol,
                        quantity=qty,
                        account_id=exec_acct,
                        signal_price=entry,
                        trail_pct=trail_pct,
                        source="scheduler",
                        idempotency_suffix=str(int(entry * 100)),
                        signal_id=sig_id,
                    ))
                    if trail_ok:
                        _mark_signal_acted_on(symbol, direction, label)
                    else:
                        logger.error(
                            "[scheduler] SELL %s: tighten_trail_on_sell returned False "
                            "— position NOT protected; leaving signal un-acted so the "
                            "reconcile job retries next cycle.", symbol,
                        )

                    try:
                        from app.services.notifications.bus import notify_signal
                        notify_signal(symbol=symbol, direction=direction, strategy=label,
                                      source="scheduler", price=entry)
                    except Exception:
                        pass
                    continue  # skip the generic order block below — trail already placed

                # ── BUY order construction (SELL continues above) ────────────
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=qty,
                    limit_price=None,
                    stop_price=stop if direction == "BUY" else None,
                    source="scheduler",
                )
                # Mark the signal row we already wrote during evaluation as acted_on.
                sig_id = _mark_signal_acted_on(symbol, direction, label)
                try:
                    from app.services.notifications.bus import notify_signal
                    notify_signal(symbol=symbol, direction=direction, strategy=label,
                                  source="scheduler", price=entry)
                except Exception:
                    pass  # never let a notify failure block an order
                # Resolve the per-assignment broker. "default" reuses the global
                # svc/account_id; anything else routes only this symbol's order.
                exec_svc, exec_acct = _svc_for(asgn_broker)
                if asgn_broker != "default":
                    logger.info(
                        "[scheduler] Routing %s %s via broker override %r",
                        direction, symbol, asgn_broker,
                    )
                loop.run_until_complete(exec_svc.execute(
                    order_req,
                    account_id=exec_acct,
                    signal_id=sig_id,
                    estimated_price=entry,
                ))

            # ── 4. Execute consensus signals ─────────────────────
            for symbol, directions in ({} if dry_run else votes).items():
                # votes are keyed by the strategy signal's raw symbol, which may be
                # mixed-case; current_positions/live_prices are uppercase. Normalize
                # so the held lookup (and the SELL-skip / cap math) don't miss.
                symbol = (symbol or "").upper()
                qualifying = {d: v for d, v in directions.items() if len(v) >= min_agree}
                if len(qualifying) != 1:
                    continue
                direction, agreeing = next(iter(qualifying.items()))
                logger.info(
                    "[scheduler] Consensus: %s %s (%d/%d: %s)",
                    direction, symbol, len(agreeing), min_agree, agreeing,
                )
                if direction == "SELL":
                    held = current_positions.get(symbol, 0.0)
                    entry_p = live_prices.get(symbol) or 0.0
                    if held < 1.0:
                        logger.info(
                            "[scheduler] Consensus SELL %s skipped — held=%.4f (need >= 1.0)",
                            symbol, held,
                        )
                        continue
                    qty = held

                    # ── Consensus SELL → tight trailing stop (Approach C) ──
                    # Mirrors the assigned-SELL path so EVERY SELL signal
                    # (assigned or consensus) places the tight trail rather
                    # than dumping at market. trail_pct: use the matching
                    # assignment's tight_trail_pct if one exists, else 2.0%.
                    c_asgn = next((a for a in assignments if a["symbol"] == symbol), None)
                    c_broker = (c_asgn.get("broker") if c_asgn else "default") or "default"
                    # India auto-routing (same as the assigned path).
                    if c_broker == "default":
                        try:
                            from app.services.markets import is_india_symbol
                            if is_india_symbol(symbol):
                                c_broker = "zerodha"
                        except Exception:
                            pass
                    c_trail_pct = float(
                        (c_asgn.get("tight_trail_pct") if c_asgn else None) or 2.0
                    )
                    consensus_label = (
                        "consensus:" + "+".join(agreeing) if agreeing else "consensus"
                    )
                    sig_id = _persist_signal(symbol, direction, consensus_label, entry_p or None)
                    exec_svc, exec_acct = _svc_for(c_broker)
                    logger.info(
                        "[scheduler] Consensus SELL %s: placing %.1f%% tight trail "
                        "(broker=%s, trail_pct=%s)",
                        symbol, c_trail_pct, c_broker,
                        (c_asgn or {}).get("tight_trail_pct"),
                    )
                    c_trail_ok = loop.run_until_complete(exec_svc.tighten_trail_on_sell(
                        symbol=symbol,
                        quantity=qty,
                        account_id=exec_acct,
                        signal_price=entry_p,
                        trail_pct=c_trail_pct,
                        source="scheduler",
                        idempotency_suffix=f"consensus-{symbol}",
                        signal_id=sig_id,
                    ))
                    if not c_trail_ok:
                        logger.error(
                            "[scheduler] Consensus SELL %s: tighten_trail_on_sell "
                            "returned False — position NOT protected; reconcile job "
                            "will retry next cycle.", symbol,
                        )
                    try:
                        from app.services.notifications.bus import notify_signal
                        notify_signal(
                            symbol=symbol, direction=direction,
                            strategy=consensus_label, source="scheduler",
                            price=entry_p or None,
                        )
                    except Exception:
                        pass
                    continue  # SELL done via trail — skip the generic block below
                else:
                    entry_p = live_prices.get(symbol) or 0.0
                    # Consensus is "all strategies agreed" but the per-symbol
                    # cap is per-assignment. Reuse the matching assignment's
                    # cap (if any) so a consensus BUY honours the same
                    # max_capital_usd / max_shares the user set.
                    c_asgn = next((a for a in assignments if a["symbol"] == symbol), None)
                    c_cap = c_asgn.get("max_capital_usd") if c_asgn else None
                    c_shares = c_asgn.get("max_shares") if c_asgn else None
                    c_held = current_positions.get(symbol, 0.0)
                    qty = (
                        _compute_quantity(
                            symbol, entry_p, None, c_cap, c_shares,
                            held_qty=c_held,
                        )
                        if entry_p > 0
                        else _quantize_for_broker(1.0)
                    )
                    if qty <= 0:
                        logger.info(
                            "[scheduler] Consensus BUY %s skipped — at/over cap "
                            "(held=%.4f, cap=%s, shares=%s).",
                            symbol, c_held, c_cap, c_shares,
                        )
                        continue
                    # Same regime open-position cap as the assigned path. Only
                    # blocks a BUY that opens a new symbol; existing holdings exempt.
                    held = current_positions.get(symbol, 0.0)
                    if max_open_positions is not None and held <= 0:
                        open_count = sum(1 for q in current_positions.values() if q > 0)
                        if open_count >= max_open_positions:
                            logger.info(
                                "[scheduler] Consensus BUY %s skipped — at regime position cap "
                                "(%d/%d open).",
                                symbol, open_count, max_open_positions,
                            )
                            continue
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=qty,
                    source="scheduler",
                )
                # Strategy name is the consensus group — prefix + agreeing list
                # so Recent Fills tells you which strategies voted to enter.
                consensus_label = "consensus:" + "+".join(agreeing) if agreeing else "consensus"
                sig_id = _persist_signal(symbol, direction, consensus_label, entry_p or None)
                try:
                    from app.services.notifications.bus import notify_signal
                    notify_signal(symbol=symbol, direction=direction, strategy=consensus_label,
                                  source="scheduler", price=entry_p or None)
                except Exception:
                    pass
                loop.run_until_complete(svc.execute(
                    order_req,
                    account_id=account_id,
                    signal_id=sig_id,
                    estimated_price=entry_p or None,
                ))

            # ── Safety net: arm trails for held positions that got a SELL
            # signal but have no resting protective stop (failed prior cycle,
            # scanner-only SELL, etc.). Live cycles only.
            if not dry_run:
                try:
                    _reconcile_trail_stops(
                        loop, broker, account_id,
                        current_positions, assignments, live_prices,
                        _svc_for,
                    )
                except Exception as exc:
                    logger.warning("[scheduler] Trail reconcile pass failed: %s", exc)
        finally:
            loop.close()

    except Exception as exc:
        logger.exception("[scheduler] Cycle error: %s", exc)
    finally:
        _running = False
        _lock.release()


_scheduler: BackgroundScheduler | None = None
_SCANNER_WATCHLIST_INTERVAL_SECONDS = 900    # 15 min — small universe, cheap
_SCANNER_LARGE_INTERVAL_SECONDS = 4 * 3600   # 4 h — sp500/nasdaq100 each take 2–5 min
# Fast trail ratchet — the full strategy cycle is 15 min, far too slow for a
# trailing stop (a spike could round-trip within one cycle). This lightweight
# job ONLY re-evaluates armed floored-stops every minute (positions + quotes +
# tighten_trail_on_sell), so the bot-managed phase ratchets tightly and hands
# off to a broker-native trail as soon as the trail level clears the floor.
_FAST_TRAIL_INTERVAL_SECONDS = 60


def _run_fast_trail_job() -> None:
    """Ratchet armed tight-trails every minute during market hours.

    Reuses reconcile_trails_now(), which only touches held positions that have a
    qualifying assigned SELL signal — idempotent and a no-op when nothing is
    armed, so it's safe to run frequently. Skips outside market hours.
    """
    if not _any_market_open():
        return
    try:
        reconcile_trails_now()
    except Exception as exc:
        logger.error("[scheduler] Fast trail job failed: %s", exc)


def _run_scanner_job_universe(universe: str, top_n: int = 5) -> None:
    """Scheduled scanner job — scans the given universe during market hours.

    Skips silently outside market hours so the job can stay registered on a
    plain interval trigger without spinning yfinance overnight.
    """
    settings = get_settings()
    if not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
        return
    try:
        from app.schemas.scanner import ScanConfig
        from app.services.scanner.scanner_service import run_scan
        summary = run_scan(ScanConfig(universe=universe, top_n=top_n, auto_trade_top=False))
        logger.info(
            "[scheduler] Scanner[%s] — %d scanned, %d matches",
            universe, summary.total_scanned, summary.total_matches,
        )
    except Exception as e:
        logger.error("[scheduler] Scanner[%s] job failed: %s", universe, e)


def _run_scanner_job() -> None:
    """Back-compat shim — old call site still pointed at watchlist."""
    _run_scanner_job_universe("watchlist", top_n=5)


_POSITION_SYNC_INTERVAL_SECONDS = 900   # 15 min — cheap broker read


def _run_m1_daily_scan_job(session_label: str) -> None:
    """Scheduled M1 SIP advisor scan (advisory only — emits a notification digest).

    Fires pre-open and post-close. Does NOT place orders; it dip-weights the
    pie analysis and writes an M1 notification + toast. See
    app/services/m1/daily_scan.py.
    """
    try:
        from app.services.m1.daily_scan import run_daily_m1_scan
        summary = run_daily_m1_scan(session_label=session_label)
        logger.info("[scheduler] M1 %s scan: %s", session_label, summary)
    except Exception as exc:
        logger.exception("[scheduler] M1 %s scan failed: %s", session_label, exc)


def _run_gtc_fill_sync_job() -> None:
    """Reconcile broker fills into the DB so closes show up automatically.

    Runs every 15 min during market hours. Delegates to
    reconciliation.order_sync.sync_broker_orders_once(), which:
      • polls list_orders (ALL statuses) on the global broker AND Zerodha,
      • flips our own submitted/working rows to filled/cancelled,
      • INGESTS orphan fills with no DB row (broker-native TRAILING_STOP fills,
        manual closes) by creating a filled Order so FIFO can pair them,
      • materializes realized round-trips so the PnL page updates immediately.

    Previously this only flipped status='submitted' rows on the GLOBAL broker,
    so India fills and orphan closes left positions looking "open" forever.
    """
    settings = get_settings()
    if not _any_market_open():
        return
    try:
        from app.services.reconciliation.order_sync import sync_broker_orders_once
        sync_broker_orders_once(lookback_days=settings.order_sync_lookback_days)
    except Exception as exc:
        logger.error("[scheduler] Order fill sync job failed: %s", exc)


def _has_india_assignments() -> bool:
    """True if any enabled assignment is an India symbol (routes to Zerodha).

    Used to decide whether jobs that default to the global broker also need to
    process the Zerodha broker for India positions.
    """
    try:
        from app.models.assignments import SymbolStrategyAssignment
        from app.services.markets import is_india_symbol

        with SessionLocal() as db:
            rows = db.query(SymbolStrategyAssignment).filter_by(enabled=True).all()
            for a in rows:
                broker = (a.broker or "default").lower()
                if broker == "zerodha":
                    return True
                if broker == "default" and is_india_symbol(a.symbol):
                    return True
    except Exception as exc:
        logger.debug("[scheduler] _has_india_assignments check failed: %s", exc)
    return False


def _any_market_open() -> bool:
    """True if a session we trade is currently open — US always, PLUS the NSE
    (India) session whenever any enabled assignment routes to Zerodha.

    The trail/protection jobs guarded only on the US session
    (trading_start_time/tz), so an India (Zerodha) position's tight trail was
    never re-evaluated during the NSE session (it runs overnight in ET) — it
    only got the 15-min main cycle, never the 60-sec fast ratchet. Gate those
    jobs on this instead so India trails ratchet and exit at the same cadence
    as US ones whenever the NSE is open.
    """
    settings = get_settings()
    if is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
        return True
    try:
        if _has_india_assignments() and is_market_hours(
            settings.india_market_open, settings.india_market_close, settings.india_tz
        ):
            return True
    except Exception as exc:
        logger.debug("[scheduler] _any_market_open India check failed: %s", exc)
    return False


def _run_chandelier_trail_job() -> None:
    """Upgrade legacy static SELL STOPs to the chandelier ATR level.

    Runs every 15 min during market hours.

    With trailing_stop_enabled=True, new positions get a broker-native
    TRAILING_STOP at BUY time — the broker ratchets it automatically, so
    this job intentionally skips those symbols (no-op on TRAILING_STOP orders).

    It only handles the legacy case: positions that were opened BEFORE the
    trailing stop feature was enabled and still have a plain STOP order.
    For those, it computes highest_close - 3×ATR22 and replaces the old
    STOP if the new level is >= 0.5% higher.
    """
    settings = get_settings()
    if not _any_market_open():
        return
    if not settings.auto_protective_stop_enabled:
        return
    try:
        import asyncio as _aio
        from app.services.brokers.factory import get_broker
        from app.services.market_data.provider import get_ohlcv
        from app.services.strategy.rules import _atr_raw
        from app.schemas.orders import OrderRequest

        def _ratchet_broker(broker, account_id: str) -> None:
            """Ratchet legacy static STOPs up to the chandelier ATR level for
            one broker's positions. Broker-native TRAILING_STOPs are skipped
            (the broker ratchets those itself)."""
            positions = loop.run_until_complete(broker.get_positions(account_id))
            long_positions = {p.symbol.upper(): p.quantity for p in positions if p.quantity > 0}
            if not long_positions:
                return

            open_orders = loop.run_until_complete(broker.list_orders(account_id, status="working"))

            # Bucket open SELL orders by symbol — we need to know whether each
            # symbol already has a broker-native TRAILING_STOP (skip it) or only
            # a plain STOP (upgrade it).
            trailing_stop_symbols: set[str] = set()
            static_stop_by_symbol: dict[str, object] = {}
            for o in open_orders:
                sym = getattr(o, "symbol", "").upper()
                otype = getattr(o, "order_type", "").upper()
                side = getattr(o, "side", "").upper()
                if side != "SELL" or sym not in long_positions:
                    continue
                if otype == "TRAILING_STOP":
                    # Broker manages the ratchet — nothing for us to do.
                    trailing_stop_symbols.add(sym)
                elif otype == "STOP":
                    static_stop_by_symbol[sym] = o

            # Only process symbols with a plain STOP and NO trailing stop.
            for symbol, qty in long_positions.items():
                if symbol in trailing_stop_symbols:
                    logger.debug(
                        "[scheduler] Chandelier trail %s: TRAILING_STOP active, "
                        "broker handles ratchet — skipping", symbol,
                    )
                    continue
                existing_stop = static_stop_by_symbol.get(symbol)
                if not existing_stop:
                    continue

                old_stop = getattr(existing_stop, "stop_price", None) or 0.0
                if not old_stop:
                    try:
                        old_stop = float(getattr(existing_stop, "raw", {}).get("stopPrice", 0) or 0)
                    except Exception:
                        continue
                if old_stop <= 0:
                    continue

                try:
                    df = get_ohlcv(symbol, period="3mo")
                    if df.empty or len(df) < 22:
                        continue
                    atr = float(_atr_raw(df, 22).iloc[-1])
                    highest_close = float(df["Close"].tail(22).max())
                    new_stop = round(highest_close - 3.0 * atr, 2)
                except Exception as exc:
                    logger.debug("[scheduler] Chandelier trail %s failed: %s", symbol, exc)
                    continue

                if new_stop <= old_stop * 1.005:
                    continue

                logger.info(
                    "[scheduler] Chandelier trail %s: ratchet legacy STOP %.2f → %.2f",
                    symbol, old_stop, new_stop,
                )
                try:
                    loop.run_until_complete(
                        broker.cancel_order(existing_stop.broker_order_id, account_id)
                    )
                    new_req = OrderRequest(
                        symbol=symbol,
                        side="SELL",
                        order_type="STOP",
                        quantity=qty,
                        stop_price=new_stop,
                        time_in_force="GTC",
                        source="scheduler",
                        idempotency_key=f"trail-{symbol}-{int(new_stop*100)}",
                    )
                    loop.run_until_complete(broker.place_order(new_req, account_id))
                    logger.info("[scheduler] Chandelier trail %s: legacy STOP upgraded to %.2f", symbol, new_stop)
                except Exception as exc:
                    logger.error("[scheduler] Chandelier trail %s: cancel/replace failed: %s", symbol, exc)

        loop = _aio.new_event_loop()
        try:
            # The global broker handles US (Schwab/Webull) positions. India
            # symbols route to Zerodha per-assignment, so its positions live on
            # a different broker — ratchet those too. Zerodha has NO native
            # trailing stop, so its protective stops are ALWAYS static STOPs
            # that this job is the only thing keeping ratcheted upward.
            brokers_to_ratchet: list = []
            try:
                brokers_to_ratchet.append(get_broker())
            except Exception as exc:
                logger.error("[scheduler] Chandelier: could not build global broker: %s", exc)

            if _has_india_assignments():
                try:
                    from app.services.brokers.factory import _build_one
                    brokers_to_ratchet.append(_build_one("zerodha"))
                except Exception as exc:
                    logger.error("[scheduler] Chandelier: could not build Zerodha broker: %s", exc)

            seen_brokers: set[str] = set()
            for broker in brokers_to_ratchet:
                bname = getattr(broker, "name", "")
                if bname in seen_brokers:
                    continue
                seen_brokers.add(bname)
                try:
                    loop.run_until_complete(broker.authenticate())
                    accounts = loop.run_until_complete(broker.get_accounts())
                    account_id = accounts[0].account_id if accounts else ""
                    _ratchet_broker(broker, account_id)
                except Exception as exc:
                    logger.error("[scheduler] Chandelier trail job failed for %s: %s", bname, exc)
        finally:
            loop.close()
    except Exception as exc:
        logger.error("[scheduler] Chandelier trail job failed: %s", exc)


def _run_position_sync_job() -> None:
    """Scheduled reconciliation — snapshot broker positions during market hours.

    Skips outside US market hours so it doesn't poll the broker overnight.
    Read-only against the broker; append-only to position_snapshots.
    """
    if not _any_market_open():
        return
    try:
        from app.services.reconciliation.position_sync import sync_positions_once
        sync_positions_once()
    except Exception as exc:
        logger.error("[scheduler] Position sync job failed: %s", exc)


def start_scheduler() -> None:
    global _scheduler
    settings = get_settings()

    if not settings.scheduler_enabled:
        logger.info("[scheduler] Scheduler disabled by config")
        return

    from datetime import datetime, timedelta, timezone as _tz

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _run_cycle,
        trigger=IntervalTrigger(seconds=settings.scheduler_interval_seconds),
        id="strategy_cycle",
        name="Strategy Evaluation Cycle",
        replace_existing=True,
        max_instances=1,
        # A tick that fires late (server busy, prior cycle ran long) must still
        # run instead of being silently dropped — that's how an assigned BUY
        # window (e.g. FANG_Pullback_EMA50) got skipped for a whole hour.
        # coalesce collapses a backlog of missed ticks into one run.
        misfire_grace_time=settings.scheduler_interval_seconds,
        coalesce=True,
    )
    _scheduler.add_job(
        _run_scanner_job_universe,
        args=["watchlist", 5],
        trigger=IntervalTrigger(seconds=_SCANNER_WATCHLIST_INTERVAL_SECONDS),
        id="scanner_cycle_watchlist",
        name="Market Scanner — Watchlist",
        replace_existing=True,
        max_instances=1,
    )
    # Stagger sp500 and nasdaq100 so they don't both hit yfinance at once.
    # Each scan takes 2–5 min; a 10-min offset gives the first one room to
    # finish before the next starts.
    now = datetime.now(_tz.utc)
    _scheduler.add_job(
        _run_scanner_job_universe,
        args=["nasdaq100", 10],
        trigger=IntervalTrigger(
            seconds=_SCANNER_LARGE_INTERVAL_SECONDS,
            start_date=now + timedelta(minutes=2),
        ),
        id="scanner_cycle_nasdaq100",
        name="Market Scanner — NASDAQ 100",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _run_scanner_job_universe,
        args=["sp500", 10],
        trigger=IntervalTrigger(
            seconds=_SCANNER_LARGE_INTERVAL_SECONDS,
            start_date=now + timedelta(minutes=12),
        ),
        id="scanner_cycle_sp500",
        name="Market Scanner — S&P 500",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _run_position_sync_job,
        trigger=IntervalTrigger(
            seconds=_POSITION_SYNC_INTERVAL_SECONDS,
            start_date=now + timedelta(minutes=1),
        ),
        id="position_sync",
        name="Broker Position Reconciliation",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _run_chandelier_trail_job,
        trigger=IntervalTrigger(
            seconds=_POSITION_SYNC_INTERVAL_SECONDS,
            start_date=now + timedelta(minutes=3),
        ),
        id="chandelier_trail",
        name="Chandelier Trail Ratchet",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _run_gtc_fill_sync_job,
        trigger=IntervalTrigger(
            seconds=_POSITION_SYNC_INTERVAL_SECONDS,   # every 15 min
            start_date=now + timedelta(minutes=5),     # stagger after chandelier job
        ),
        id="gtc_fill_sync",
        name="GTC Order Fill Sync",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _run_fast_trail_job,
        trigger=IntervalTrigger(seconds=_FAST_TRAIL_INTERVAL_SECONDS),
        id="fast_trail_ratchet",
        name="Fast Tight-Trail Ratchet",
        replace_existing=True,
        # Don't pile up if a run overruns the 60s interval; just skip the tick.
        max_instances=1,
        coalesce=True,
    )
    # ── M1 SIP advisor — daily scans (advisory only, emits notifications) ──
    # Pre-open digest (~9:00 ET) and post-close review (~16:15 ET), in US/Eastern
    # so DST is handled automatically.
    _scheduler.add_job(
        _run_m1_daily_scan_job,
        args=["pre-open"],
        trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone="US/Eastern"),
        id="m1_scan_preopen",
        name="M1 SIP Advisor — Pre-Open",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
        coalesce=True,
    )
    _scheduler.add_job(
        _run_m1_daily_scan_job,
        args=["post-close"],
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=15, timezone="US/Eastern"),
        id="m1_scan_postclose",
        name="M1 SIP Advisor — Post-Close",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
        coalesce=True,
    )
    _scheduler.start()
    logger.info(
        f"[scheduler] Started — strategy_cycle={settings.scheduler_interval_seconds}s, "
        f"scanner watchlist={_SCANNER_WATCHLIST_INTERVAL_SECONDS}s, "
        f"scanner sp500/nasdaq100={_SCANNER_LARGE_INTERVAL_SECONDS}s, "
        f"position_sync={_POSITION_SYNC_INTERVAL_SECONDS}s, "
        f"fast_trail={_FAST_TRAIL_INTERVAL_SECONDS}s"
    )


def stop_scheduler() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] Stopped")


def run_once(*, force: bool = False, dry_run: bool = False) -> None:
    """Manually trigger one cycle.

    force=True    — bypasses market-hours gate.
    dry_run=True  — evaluates signals, writes signal rows, places NO orders.
                    Does NOT touch the kill switch, so the live scheduler can
                    run concurrently without interference.
    """
    _run_cycle(force=force, dry_run=dry_run)


def get_scheduler_status() -> dict:
    settings = get_settings()
    running = _scheduler is not None and _scheduler.running
    next_run = None
    if running:
        job = _scheduler.get_job("strategy_cycle")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {
        "enabled": settings.scheduler_enabled,
        "running": running,
        "interval_seconds": settings.scheduler_interval_seconds,
        "cycle_active": _running,
        "next_run": next_run,
        "run_bollinger": _bollinger_enabled(),
        "run_perplexity": _perplexity_enabled(),
    }
