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
from app.services.brokers.factory import get_broker
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


def _compute_quantity(
    symbol: str,
    entry: float,
    stop: float | None,
    max_capital_usd: float | None = None,
) -> float:
    """
    Return shares to buy using fixed-fractional position sizing.

    max_capital_usd: per-symbol budget cap set by the user on the assignment.
                     When set, shares are capped so position value never exceeds it.
                     Falls back to 1 share if stop is missing or sizing is not viable.
    """
    if stop is None or stop <= 0 or stop >= entry:
        # No stop — cap by max_capital_usd if given, else 1 share
        if max_capital_usd and entry > 0:
            return max(round(max_capital_usd / entry, 6), 0.001)
        return 1.0

    settings = get_settings()
    # If user set a per-symbol cap, use that; otherwise use global max_position_size_usd
    effective_max = max_capital_usd if max_capital_usd else settings.max_position_size_usd
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
            logger.info(
                f"[scheduler] Position size {symbol}: {sz.shares:.4f} shares "
                f"@ ${entry:.2f}, stop ${stop:.2f}, risk ${sz.risk_amount:.2f}, "
                f"cap=${effective_max:.0f}"
            )
            return sz.shares
    except Exception as exc:
        logger.warning(f"[scheduler] Position sizing failed for {symbol}: {exc}")
    return 1.0


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


def _run_cycle() -> None:
    global _running
    if not _lock.acquire(blocking=False):
        logger.debug("[scheduler] Previous cycle still running — skipping")
        return
    _running = True
    try:
        settings = get_settings()

        if not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
            logger.debug("[scheduler] Outside market hours — skipping cycle")
            return

        if _risk.is_kill_switch_active():
            logger.warning("[scheduler] Kill switch active — skipping cycle")
            return

        settings = get_settings()
        broker = get_broker()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(broker.authenticate())
            accounts = loop.run_until_complete(broker.get_accounts())
            account_id = accounts[0].account_id if accounts else ""
            svc = ExecutionService(broker)

            from app.models.assignments import SymbolStrategyAssignment
            from app.schemas.orders import OrderRequest

            # ── Load per-symbol strategy assignments ─────────────
            with SessionLocal() as db:
                active_assignments = (
                    db.query(SymbolStrategyAssignment)
                    .filter_by(enabled=True)
                    .all()
                )
                # snapshot to plain dicts so we can close the session
                assignments = [
                    {"symbol": a.symbol, "system": a.system, "strategy_name": a.strategy_name,
                     "max_capital_usd": a.max_capital_usd}
                    for a in active_assignments
                ]

            assigned_symbols = {a["symbol"] for a in assignments}

            # signals_to_act: list of (symbol, direction, strategy_label, entry_price, stop_price)
            signals_to_act: list[tuple[str, str, str, float, float | None]] = []

            # ── 1. Run assigned strategies (one per symbol) ──────
            # Each assigned symbol runs exactly its assigned strategy.
            # A single signal from the assigned strategy is enough — no
            # consensus needed because the user explicitly chose this strategy.
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
                        df = get_ohlcv(symbol, period="1y")
                        if df.empty or len(df) < 210:
                            continue
                        sig = strat.run(symbol, df)
                        if sig.direction != "HOLD":
                            entry = sig.entry_price or float(df["Close"].iloc[-1])
                            signals_to_act.append((
                                symbol, sig.direction,
                                f"perplexity:{strategy_name}",
                                entry, sig.stop_price,
                            ))
                            logger.info(
                                f"[scheduler] Assigned {strategy_name} → {symbol}: "
                                f"{sig.direction} entry={entry:.2f} stop={sig.stop_price} ({sig.reason})"
                            )

                    elif system == "bollinger":
                        bollinger_configs = load_strategies_from_config()
                        matching = [c for c in bollinger_configs if c.name == strategy_name and c.enabled]
                        for config in matching:
                            prices = get_price_series(symbol, period="1y")
                            sigs = _engine.run(config, prices)
                            for s in sigs:
                                if s.direction != "HOLD":
                                    entry = s.price_at_signal or float(prices.iloc[-1])
                                    signals_to_act.append((
                                        symbol, s.direction, strategy_name, entry, None
                                    ))
                                    logger.info(f"[scheduler] Assigned {strategy_name} → {symbol}: {s.direction}")

                except Exception as exc:
                    logger.error(f"[scheduler] Assigned strategy {strategy_name} on {symbol} failed: {exc}")

            # ── 2. Run general pool for non-assigned symbols ──────
            # Bollinger and/or Perplexity run on symbols NOT already covered
            # by an assignment, using the original consensus approach.
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
                        logger.error(f"[scheduler] Bollinger {config.name} failed: {exc}")

            if _perplexity_enabled():
                try:
                    all_bollinger_symbols = list({c.symbol for c in load_strategies_from_config()})
                except Exception:
                    all_bollinger_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA",
                                             "MSFT", "AMZN", "META", "GOOGL", "JPM"]
                for symbol in all_bollinger_symbols:
                    if symbol in assigned_symbols:
                        continue
                    try:
                        df = get_ohlcv(symbol, period="1y")
                        if df.empty or len(df) < 210:
                            continue
                        pool_sigs = run_perplexity_signal(symbol, df)
                        for sig in pool_sigs:
                            if sig.direction != "HOLD":
                                votes[sig.symbol][sig.direction].append(
                                    f"perplexity:{sig.strategy_name}"
                                )
                    except Exception as exc:
                        logger.error(f"[scheduler] Perplexity pool {symbol} failed: {exc}")

            # ── 3. Execute assigned signals (no consensus needed) ─
            for symbol, direction, label, entry, stop in signals_to_act:
                asgn_cap = next((a["max_capital_usd"] for a in assignments if a["symbol"] == symbol), None)
                qty = _compute_quantity(symbol, entry, stop, asgn_cap) if direction == "BUY" else 1.0
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=qty,
                )
                loop.run_until_complete(svc.execute(order_req, account_id=account_id))

            # ── 4. Execute consensus signals for unassigned symbols
            for symbol, directions in votes.items():
                for direction, agreeing in directions.items():
                    if len(agreeing) >= min_agree:
                        logger.info(
                            f"[scheduler] Consensus: {direction} {symbol} "
                            f"({len(agreeing)}/{min_agree}: {agreeing})"
                        )
                        # For consensus signals we don't have a stop price, so fall back to 1
                        order_req = OrderRequest(
                            symbol=symbol,
                            side=direction,  # type: ignore[arg-type]
                            order_type="MARKET",
                            quantity=1,
                        )
                        loop.run_until_complete(svc.execute(order_req, account_id=account_id))
                    else:
                        logger.debug(
                            f"[scheduler] No consensus: {direction} {symbol} "
                            f"({len(agreeing)}/{min_agree})"
                        )
        finally:
            loop.close()

    except Exception as exc:
        logger.exception("[scheduler] Cycle error: %s", exc)
    finally:
        _running = False
        _lock.release()


_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> None:
    global _scheduler
    settings = get_settings()

    if not settings.scheduler_enabled:
        logger.info("[scheduler] Scheduler disabled by config")
        return

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _run_cycle,
        trigger=IntervalTrigger(seconds=settings.scheduler_interval_seconds),
        id="strategy_cycle",
        name="Strategy Evaluation Cycle",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(f"[scheduler] Started — interval={settings.scheduler_interval_seconds}s")


def stop_scheduler() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] Stopped")


def run_once() -> None:
    """Manually trigger one cycle (for scripts / CLI)."""
    _run_cycle()


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
