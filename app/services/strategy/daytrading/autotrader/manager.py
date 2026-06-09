"""
AutoTraderManager — single entry point for "flip the switch" day trading.

One `SingleStockTrader` runs per symbol. The manager owns:
- the brokerage instance shared across symbols,
- the EOD watchdog that auto-stops the whole switch at 4:00 PM ET,
- thread-safe start / stop / status accessors used by the API layer.

The trader itself already handles bar polling, entry/exit decisions,
risk governor, and EOD force-flatten at 3:45 PM ET. This manager just
wires it to a real broker and exposes a flat surface.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Any

from app.services.broker import BaseBroker, get_broker
from app.services.execution.service import ExecutionService
from app.services.strategy.daytrading.autotrader.single_stock_trader import (
    PolicyError,
    SingleStockTrader,
)
from app.services.strategy.daytrading.market_open import (
    is_market_open,
    is_pre_market,
    now_et,
    market_session,
)

logger = logging.getLogger(__name__)

# Auto-stop the switch at this ET time. Trader self-flattens at 3:45 PM ET
# (see exit_manager._EOD_FORCE_FLAT); we give it 15 min, then power down.
_EOD_AUTO_STOP = dtime(16, 0)


@dataclass
class AutoTraderConfig:
    """Per-switch configuration captured when the user flips it on."""
    symbols: list[str]
    direction_mode: str = "long_only"          # long_only | short_only | both
    trail_mode: str = "atr"                    # ema | atr | candle
    partial_tp: bool = True
    risk_per_trade_pct: float = 0.01
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 6
    max_consecutive_losses: int = 3
    initial_capital: float = 10_000.0
    broker_name: str = "paper"                 # paper | alpaca

    # Entry-path mode: keep legacy by default so existing flips behave identically.
    # "native_strategy" delegates to strategy.generate_signals() — see
    # NativeStrategyEntry for the supported strategy list.
    entry_mode: str = "legacy_entry_decider"   # legacy_entry_decider | native_strategy
    native_strategies: list[str] | None = None
    tight_trail_on_exit_signal: bool = True    # arm tight trail on momentum-fade exit

    # Filled by the manager after start():
    started_at_et: str | None = None
    started_by_market_state: str | None = None


class AutoTraderManager:
    """
    Process-wide singleton that owns active SingleStockTrader instances.

    Lifecycle (mirroring a real trader's day):
      flip_on()  → spin up one trader per symbol, EOD watchdog thread,
                   refuse to start outside RTH unless force=True.
      flip_off() → stop all traders. Does NOT close open positions unless
                   flatten=True, because manual oversight may want to
                   carry/scale-out a single name.
    """

    def __init__(self) -> None:
        self._traders: dict[str, SingleStockTrader] = {}
        self._lock = threading.RLock()
        self._broker: BaseBroker | None = None
        self._execution_service: ExecutionService | None = None
        self._account_id: str = ""
        self._config: AutoTraderConfig | None = None
        self._eod_thread: threading.Thread | None = None
        self._eod_stop = threading.Event()

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_on(self) -> bool:
        with self._lock:
            return any(t._running for t in self._traders.values())

    def flip_on(self, config: AutoTraderConfig, *, force: bool = False) -> dict[str, Any]:
        """
        Activate auto-trading for every symbol in config.symbols.

        force=False (default) refuses to start outside RTH so a midnight click
        doesn't silently no-op for hours. Pass force=True to start in pre-market
        — the trader will idle internally until 9:30 ET.
        """
        with self._lock:
            if self.is_on:
                raise RuntimeError(
                    "Auto-trader already running. Call flip_off first or use replace=True via the API."
                )

            # Allow start if ANY symbol's market is open or in pre-market.
            # This correctly handles mixed US + India watchlists where NSE
            # hours (09:15–15:30 IST) don't overlap with US RTH.
            any_market_tradeable = any(
                is_market_open(sym) or is_pre_market(sym)
                for sym in config.symbols
            ) if config.symbols else is_market_open() or is_pre_market()
            if not force and not any_market_tradeable:
                raise RuntimeError(
                    f"Market is closed for all symbols (US time: {now_et().strftime('%H:%M %Z')}). "
                    f"Pass force=true to arm anyway."
                )

            self._broker = get_broker(config.broker_name, initial_capital=config.initial_capital) \
                if config.broker_name == "paper" else get_broker(config.broker_name)

            # Build ExecutionService so live orders go through the full pipeline:
            # risk gates → DB persistence → audit → broker submit → protective stop.
            # Paper mode still works — PaperBroker fills immediately at last price.
            self._execution_service = ExecutionService(self._broker)

            # Resolve account_id: authenticate then fetch accounts. Silently
            # falls back to "" which ExecutionService accepts (some brokers don't
            # require an explicit account ID for single-account setups).
            self._account_id = ""
            try:
                import asyncio as _asyncio
                _asyncio.run(self._broker.authenticate())
                accts = _asyncio.run(self._broker.get_accounts())
                if accts:
                    self._account_id = accts[0].account_id or ""
                    logger.info(
                        "[autotrader] resolved account_id=%s for broker=%s",
                        self._account_id, config.broker_name,
                    )
            except Exception as _ae:
                logger.warning(
                    "[autotrader] could not resolve account_id (broker=%s): %s — "
                    "orders will still route but account_id will be empty.",
                    config.broker_name, _ae,
                )

            started: list[str] = []
            errors: dict[str, str] = {}
            for raw_sym in config.symbols:
                sym = raw_sym.strip().upper()
                if not sym or sym in self._traders:
                    continue
                trader = SingleStockTrader(
                    symbol=sym,
                    broker=self._broker,
                    execution_service=self._execution_service,
                    account_id=self._account_id,
                    direction_mode=config.direction_mode,
                    trail_mode=config.trail_mode,  # type: ignore[arg-type]
                    partial_tp=config.partial_tp,
                    risk_per_trade_pct=config.risk_per_trade_pct,
                    max_daily_loss_pct=config.max_daily_loss_pct,
                    max_trades_per_day=config.max_trades_per_day,
                    max_consecutive_losses=config.max_consecutive_losses,
                    initial_capital=config.initial_capital,
                    entry_mode=config.entry_mode,
                    native_strategies=config.native_strategies,
                    tight_trail_on_exit_signal=config.tight_trail_on_exit_signal,
                )
                try:
                    trader.start()
                    self._traders[sym] = trader
                    started.append(sym)
                except PolicyError as e:
                    errors[sym] = f"policy blocked: {e}"
                    logger.warning("[autotrader] %s blocked by policy: %s", sym, e)
                except Exception as e:
                    errors[sym] = str(e)
                    logger.exception("[autotrader] failed to start %s", sym)

            if not started:
                # Nothing came up — release the broker and bail with detail.
                self._broker = None
                raise RuntimeError(f"No traders started. Errors: {errors}")

            config.started_at_et = now_et().isoformat()
            self._config = config
            self._start_eod_watchdog()

            logger.info(
                "[autotrader] switch ON — symbols=%s, broker=%s, mode=%s",
                started, config.broker_name, config.direction_mode,
            )
            return {
                "running": True,
                "started": started,
                "errors": errors,
                "started_at_et": config.started_at_et,
            }

    def flip_off(self, *, flatten: bool = False) -> dict[str, Any]:
        """
        Stop every active trader.

        flatten=True closes all open positions at market before stopping.
        Default is False so an accidental click doesn't dump open trades.
        """
        with self._lock:
            symbols = list(self._traders.keys())
            flattened: list[str] = []
            stopped: list[str] = []
            for sym, trader in self._traders.items():
                if flatten:
                    try:
                        trader.force_flatten(reason="Manager flip_off(flatten=True)")
                        flattened.append(sym)
                    except Exception as e:
                        logger.error("[autotrader] flatten %s failed: %s", sym, e)
                trader.stop()
                stopped.append(sym)
            self._traders.clear()
            self._stop_eod_watchdog()
            self._broker = None
            self._execution_service = None
            self._account_id = ""
            self._config = None
            logger.info("[autotrader] switch OFF — stopped=%s, flattened=%s", stopped, flattened)
            return {"running": False, "stopped": stopped, "flattened": flattened}

    def flatten(self, symbol: str | None = None) -> dict[str, Any]:
        """Force-flatten one symbol (or all) without stopping the loop."""
        with self._lock:
            results: dict[str, str] = {}
            targets = [symbol.upper()] if symbol else list(self._traders.keys())
            for sym in targets:
                trader = self._traders.get(sym)
                if not trader:
                    results[sym] = "not_running"
                    continue
                try:
                    trader.force_flatten(reason=f"Manual flatten via API")
                    results[sym] = "flattened"
                except Exception as e:
                    results[sym] = f"error: {e}"
            return results

    def decision_summary(
        self, symbol: str | None = None, limit: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate native-vs-legacy decision data across active traders.

        symbol=None aggregates every running trader; otherwise restricts to
        the named one. `limit` caps how many recent decision_log entries each
        trader looks at.
        """
        with self._lock:
            if symbol:
                t = self._traders.get(symbol.upper())
                if not t:
                    return {"error": "not_running", "symbol": symbol.upper()}
                return {
                    "now_et": now_et().isoformat(),
                    "symbols": {symbol.upper(): t.decision_summary(limit=limit)},
                }
            return {
                "now_et": now_et().isoformat(),
                "symbols": {
                    sym: t.decision_summary(limit=limit)
                    for sym, t in self._traders.items()
                },
            }

    def status(self) -> dict[str, Any]:
        """Snapshot of every active trader plus manager-level config."""
        with self._lock:
            traders = {sym: t.get_status() for sym, t in self._traders.items()}
            return {
                "running": self.is_on,
                "now_et": now_et().isoformat(),
                "market_open": is_market_open(),
                "config": _config_as_dict(self._config) if self._config else None,
                "traders": traders,
            }

    # ── EOD watchdog ─────────────────────────────────────────────────────────

    def _start_eod_watchdog(self) -> None:
        if self._eod_thread and self._eod_thread.is_alive():
            return
        self._eod_stop.clear()
        self._eod_thread = threading.Thread(
            target=self._eod_loop, name="AutoTraderEOD", daemon=True
        )
        self._eod_thread.start()

    def _stop_eod_watchdog(self) -> None:
        self._eod_stop.set()
        # Don't join — daemon, and the manager lock would deadlock the loop.

    def _eod_loop(self) -> None:
        """Auto-stop the switch at 4:00 PM ET."""
        while not self._eod_stop.is_set():
            if self._eod_stop.wait(timeout=30):
                return
            try:
                if now_et().time() >= _EOD_AUTO_STOP and self.is_on:
                    logger.info("[autotrader] EOD reached — auto-stopping switch")
                    self.flip_off(flatten=False)  # trader already flattened at 3:45
                    return
            except Exception as e:
                logger.error("[autotrader] EOD watchdog error: %s", e)


def _config_as_dict(cfg: AutoTraderConfig) -> dict[str, Any]:
    return {
        "symbols": cfg.symbols,
        "direction_mode": cfg.direction_mode,
        "trail_mode": cfg.trail_mode,
        "partial_tp": cfg.partial_tp,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "max_daily_loss_pct": cfg.max_daily_loss_pct,
        "max_trades_per_day": cfg.max_trades_per_day,
        "max_consecutive_losses": cfg.max_consecutive_losses,
        "initial_capital": cfg.initial_capital,
        "broker_name": cfg.broker_name,
        "entry_mode": cfg.entry_mode,
        "native_strategies": cfg.native_strategies,
        "started_at_et": cfg.started_at_et,
    }


# Process-wide singleton — one switch for the whole app.
_manager: AutoTraderManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> AutoTraderManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = AutoTraderManager()
    return _manager
