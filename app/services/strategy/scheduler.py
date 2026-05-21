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


def _persist_signal(
    symbol: str,
    direction: str,
    strategy_label: str,
    entry: float | None,
) -> int | None:
    """Write a Signal row so the resulting Order can join back to a strategy name.

    Without this, app/api/routes/orders.py:35-39 has no signal_id to look up
    and the dashboard's Recent Fills shows Strategy "—". The label may be
    prefixed (e.g. "perplexity:my_strategy") — we keep the prefix so the
    dashboard surfaces the originating system as well as the strategy.
    """
    try:
        with SessionLocal() as db:
            sig = Signal(
                strategy_name=strategy_label[:128],
                symbol=symbol.upper(),
                direction=direction,
                strength=1.0,
                price_at_signal=entry,
                acted_on=True,
            )
            db.add(sig)
            db.commit()
            db.refresh(sig)
            return sig.id
    except Exception as exc:
        logger.warning("[scheduler] Could not persist Signal row for %s/%s: %s", symbol, strategy_label, exc)
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
            return _quantize_for_broker(max_capital_usd / entry)
        return _quantize_for_broker(1.0)

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
                assignments = [
                    {"symbol": a.symbol, "system": a.system, "strategy_name": a.strategy_name,
                     "max_capital_usd": a.max_capital_usd}
                    for a in active_assignments
                ]

            assigned_symbols = {a["symbol"] for a in assignments}

            # ── Fetch Schwab live prices for all assigned symbols ─
            # Used to replace yfinance close for entry price accuracy.
            all_symbols = list(assigned_symbols)
            live_prices: dict[str, float] = {}
            if all_symbols:
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
            try:
                positions = loop.run_until_complete(broker.get_positions(account_id))
                for pos in positions:
                    current_positions[pos.symbol.upper()] = pos.quantity
            except Exception as exc:
                logger.warning("[scheduler] Could not fetch positions: %s", exc)

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
                        # Patch latest close with Schwab live price for accurate signal
                        live = live_prices.get(symbol)
                        if live:
                            df = df.copy()
                            df.iloc[-1, df.columns.get_loc("Close")] = live
                        sig = strat.run(symbol, df)
                        if sig.direction != "HOLD":
                            entry = live or sig.entry_price or float(df["Close"].iloc[-1])
                            signals_to_act.append((
                                symbol, sig.direction,
                                f"perplexity:{strategy_name}",
                                entry, sig.stop_price,
                            ))
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
                                if s.direction != "HOLD":
                                    entry = live_prices.get(symbol) or s.price_at_signal or float(prices.iloc[-1])
                                    signals_to_act.append((
                                        symbol, s.direction, strategy_name, entry, None
                                    ))
                                    logger.info("[scheduler] Assigned %s → %s: %s", strategy_name, symbol, s.direction)

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
            for symbol, direction, label, entry, stop in signals_to_act:
                asgn_cap = next((a["max_capital_usd"] for a in assignments if a["symbol"] == symbol), None)
                if direction == "BUY":
                    qty = _compute_quantity(symbol, entry, stop, asgn_cap)
                    if qty <= 0:
                        logger.info("[scheduler] BUY %s skipped — sizing produced 0 shares (cap=%s, entry=%.2f)", symbol, asgn_cap, entry)
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
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=qty,
                    limit_price=None,
                    stop_price=stop if direction == "BUY" else None,
                    source="scheduler",
                )
                sig_id = _persist_signal(symbol, direction, label, entry)
                loop.run_until_complete(svc.execute(
                    order_req,
                    account_id=account_id,
                    signal_id=sig_id,
                    estimated_price=entry,
                ))

            # ── 4. Execute consensus signals ─────────────────────
            for symbol, directions in votes.items():
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
_SCANNER_INTERVAL_SECONDS = 900  # scan every 15 minutes during market hours


def _run_scanner_job() -> None:
    """Scheduled scanner job — scans watchlist every 15 min during market hours."""
    settings = get_settings()
    if not is_market_hours(settings.trading_start_time, settings.trading_end_time, settings.tz):
        return
    try:
        from app.schemas.scanner import ScanConfig
        from app.services.scanner.scanner_service import run_scan
        summary = run_scan(ScanConfig(universe="watchlist", top_n=5, auto_trade_top=False))
        logger.info("[scheduler] Scanner — %d scanned, %d matches", summary.total_scanned, summary.total_matches)
    except Exception as e:
        logger.error("[scheduler] Scanner job failed: %s", e)


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
    _scheduler.add_job(
        _run_scanner_job,
        trigger=IntervalTrigger(seconds=_SCANNER_INTERVAL_SECONDS),
        id="scanner_cycle",
        name="Market Scanner",
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
