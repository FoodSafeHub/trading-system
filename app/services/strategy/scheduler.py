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
) -> float:
    """
    Return shares to buy using fixed-fractional position sizing.

    max_capital_usd: per-symbol dollar cap set by the user on the assignment.
                     When set, shares are capped so position value never exceeds it.
    max_shares:      per-symbol shares cap. Used ONLY when max_capital_usd is
                     empty — dollar cap wins whenever both are set.
                     Falls back to 1 share if stop is missing or sizing is not viable.
    """
    if stop is None or stop <= 0 or stop >= entry:
        # No stop — cap by max_capital_usd if given, else max_shares, else 1.
        if max_capital_usd and entry > 0:
            return _quantize_for_broker(max_capital_usd / entry)
        if max_shares and max_shares > 0:
            return _quantize_for_broker(max_shares)
        return _quantize_for_broker(1.0)

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
    try:
        sz = calculate_position_size(
            symbol=symbol,
            entry_price=entry,
            stop_price=stop,
            account_value=settings.account_value,
            risk_pct_per_trade=settings.risk_pct_per_trade,
            max_position_size_usd=effective_max,
            max_account_risk_pct=settings.max_account_risk_pct,
        )
        if sz.viable and sz.shares >= 0.001:
            qty = _quantize_for_broker(sz.shares)
            logger.info(
                f"[scheduler] Position size {symbol}: raw={sz.shares:.4f} -> qty={qty} "
                f"@ ${entry:.2f}, stop ${stop:.2f}, risk ${sz.risk_amount:.2f}, "
                f"cap=${effective_max:.0f}"
            )
            return qty
    except Exception as exc:
        logger.warning(f"[scheduler] Position sizing failed for {symbol}: {exc}")
    return _quantize_for_broker(1.0)


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

        if not force and not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
            logger.debug("[scheduler] Outside market hours — skipping cycle")
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
                    {"symbol": a.symbol, "system": a.system, "strategy_name": a.strategy_name,
                     "max_capital_usd": a.max_capital_usd, "max_shares": a.max_shares,
                     "broker": a.broker or "default"}
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

            # signals_to_act: list of (symbol, direction, label, entry_price, stop_price)
            signals_to_act: list[tuple[str, str, str, float, float | None]] = []

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
                            signals_to_act.append((symbol, sig.direction, label, entry, sig.stop_price))
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
                                    signals_to_act.append((symbol, s.direction, strategy_name, entry, None))
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
                                    signals_to_act.append((symbol, sig.direction, label, entry, None))
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
                    if not config.enabled or config.symbol in assigned_symbols:
                        continue
                    try:
                        prices = get_price_series(config.symbol, period="1y")
                        sigs = _engine.run(config, prices)
                        for s in sigs:
                            if s.direction != "HOLD":
                                votes[s.symbol][s.direction].append(config.name)
                    except Exception as exc:
                        logger.error("[scheduler] Bollinger %s failed: %s", config.name, exc)

            if _perplexity_enabled():
                # Use symbols from assignments or bollinger configs as the pool universe
                try:
                    pool_symbols = list({c.symbol for c in load_strategies_from_config()})
                except Exception:
                    pool_symbols = []
                for symbol in pool_symbols:
                    if symbol in assigned_symbols:
                        continue
                    try:
                        df = get_ohlcv(symbol, period="2y")
                        if df.empty or len(df) < 60:
                            continue
                        pool_sigs = run_perplexity_signal(symbol, df)
                        for sig in pool_sigs:
                            if sig.direction != "HOLD":
                                votes[sig.symbol][sig.direction].append(
                                    f"perplexity:{sig.strategy_name}"
                                )
                    except Exception as exc:
                        logger.error("[scheduler] Perplexity pool %s failed: %s", symbol, exc)

            # ── 3. Execute assigned signals (no consensus needed) ─
            if dry_run:
                logger.info("[scheduler] Dry run — %d assigned signal(s) evaluated, no orders placed.", len(signals_to_act))
            for symbol, direction, label, entry, stop in ([] if dry_run else signals_to_act):
                asgn = next((a for a in assignments if a["symbol"] == symbol), None)
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
                    qty = _compute_quantity(symbol, entry, stop, asgn_cap, asgn_shares)
                    if qty <= 0:
                        logger.info("[scheduler] BUY %s skipped — sizing produced 0 shares (cap=%s, shares=%s, entry=%.2f)", symbol, asgn_cap, asgn_shares, entry)
                        continue
                    # Idempotent re-entry: an assigned BUY signal stays active for
                    # several hourly cycles. Once we already hold at least the
                    # target size, don't keep stacking — that would pyramid the
                    # position far past the user's cap. Re-buy only tops up toward
                    # the target if a partial fill left us short.
                    held = current_positions.get(symbol, 0.0)
                    if held >= qty:
                        logger.info(
                            "[scheduler] BUY %s skipped — already hold %.4f >= target %.4f",
                            symbol, held, qty,
                        )
                        continue
                    # Regime open-position cap: only blocks BUYs that would open
                    # a NEW symbol. A top-up to a symbol we already hold (held>0)
                    # is exempt — it doesn't increase the count of distinct
                    # positions. None = regime lookup failed, so no cap.
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
                    # SELL: only close positions we actually hold. Be strict —
                    # require a meaningful position (>= 1 share) to avoid the
                    # phantom-SELL burst that hit AAPL/SPY/GOOGL/AMZN/AMD on
                    # 2026-05-19 13:49 where the broker rejected every order
                    # as oversold/overbought.
                    held = current_positions.get(symbol, 0.0)
                    if held < 1.0:
                        logger.info(
                            "[scheduler] SELL %s skipped — held=%.4f (need >= 1.0)",
                            symbol, held,
                        )
                        continue
                    qty = held
                    # Cancel any resting STOP / TRAILING_STOP orders for this
                    # symbol before sending the MARKET SELL. Without this, the
                    # orphaned stop fires after the position is already flat,
                    # producing a phantom short or broker rejection.
                    try:
                        _exec_svc, _exec_acct = _svc_for(asgn_broker)
                        _open = loop.run_until_complete(
                            _exec_svc.broker.list_orders(_exec_acct, status="working")
                        )
                        for _o in _open:
                            if (getattr(_o, "symbol", "").upper() == symbol.upper()
                                    and getattr(_o, "side", "").upper() == "SELL"
                                    and getattr(_o, "order_type", "").upper() in ("STOP", "TRAILING_STOP")
                                    and getattr(_o, "broker_order_id", None)):
                                loop.run_until_complete(
                                    _exec_svc.broker.cancel_order(_o.broker_order_id, _exec_acct)
                                )
                                logger.info(
                                    "[scheduler] Cancelled resting %s %s before MARKET SELL",
                                    _o.order_type, symbol,
                                )
                    except Exception as _ce:
                        logger.warning(
                            "[scheduler] Could not cancel resting stops for %s: %s — proceeding with SELL",
                            symbol, _ce,
                        )
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
                else:
                    entry_p = live_prices.get(symbol) or 0.0
                    qty = _compute_quantity(symbol, entry_p, None) if entry_p > 0 else _quantize_for_broker(1.0)
                    if qty <= 0:
                        logger.info("[scheduler] Consensus BUY %s skipped — sizing produced 0 shares", symbol)
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


def _run_chandelier_trail_job() -> None:
    """Ratchet static SELL STOP orders upward using chandelier ATR math.

    Runs every 15 min during market hours. For each open long position that
    has a resting static STOP (not a broker-native trailing stop), compute the
    current chandelier stop (highest_close - 3×ATR22) and cancel+replace the
    old stop if the new level is meaningfully higher (>= 0.5% improvement).
    This upgrades positions entered before trailing_stop_enabled was set.
    """
    settings = get_settings()
    if not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
        return
    if not settings.auto_protective_stop_enabled:
        return
    try:
        import asyncio as _aio
        from app.services.brokers.factory import get_broker
        from app.services.market_data.provider import get_ohlcv
        from app.services.strategy.rules import _atr_raw
        from app.schemas.orders import OrderRequest
        from app.models.orders import Order

        loop = _aio.new_event_loop()
        try:
            broker = get_broker()
            loop.run_until_complete(broker.authenticate())
            accounts = loop.run_until_complete(broker.get_accounts())
            account_id = accounts[0].account_id if accounts else ""

            # Find all open positions
            positions = loop.run_until_complete(broker.get_positions(account_id))
            long_positions = {p.symbol.upper(): p.quantity for p in positions if p.quantity > 0}
            if not long_positions:
                return

            # Find resting SELL STOP orders for those symbols
            open_orders = loop.run_until_complete(broker.list_orders(account_id, status="working"))
            stop_by_symbol: dict[str, object] = {}
            for o in open_orders:
                if o.side == "SELL" and o.order_type in ("STOP", "stop") and o.symbol in long_positions:
                    stop_by_symbol[o.symbol] = o

            for symbol, qty in long_positions.items():
                existing_stop = stop_by_symbol.get(symbol)
                if not existing_stop:
                    continue
                old_stop = getattr(existing_stop, "stop_price", None) or 0.0
                if not old_stop:
                    # Try to parse from raw
                    try:
                        old_stop = float(existing_stop.raw.get("stopPrice", 0) or 0)
                    except Exception:
                        continue
                if old_stop <= 0:
                    continue

                # Compute chandelier stop
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

                # Only ratchet UP and only when the improvement is >= 0.5%
                if new_stop <= old_stop * 1.005:
                    continue

                logger.info(
                    "[scheduler] Chandelier trail %s: ratchet STOP %.2f → %.2f",
                    symbol, old_stop, new_stop,
                )
                # Cancel old stop, place new one
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
                    logger.info("[scheduler] Chandelier trail %s: new STOP placed @ %.2f", symbol, new_stop)
                except Exception as exc:
                    logger.error("[scheduler] Chandelier trail %s: cancel/replace failed: %s", symbol, exc)
        finally:
            loop.close()
    except Exception as exc:
        logger.error("[scheduler] Chandelier trail job failed: %s", exc)


def _run_position_sync_job() -> None:
    """Scheduled reconciliation — snapshot broker positions during market hours.

    Skips outside US market hours so it doesn't poll the broker overnight.
    Read-only against the broker; append-only to position_snapshots.
    """
    settings = get_settings()
    if not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
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
            seconds=_POSITION_SYNC_INTERVAL_SECONDS,   # same 15-min cadence
            start_date=now + timedelta(minutes=3),     # stagger after position sync
        ),
        id="chandelier_trail",
        name="Chandelier Trail Ratchet",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(
        f"[scheduler] Started — strategy_cycle={settings.scheduler_interval_seconds}s, "
        f"scanner watchlist={_SCANNER_WATCHLIST_INTERVAL_SECONDS}s, "
        f"scanner sp500/nasdaq100={_SCANNER_LARGE_INTERVAL_SECONDS}s, "
        f"position_sync={_POSITION_SYNC_INTERVAL_SECONDS}s"
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
