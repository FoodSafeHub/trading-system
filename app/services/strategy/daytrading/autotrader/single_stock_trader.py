"""
SingleStockTrader — top-level orchestrator for one-symbol intraday auto-trading.

Workflow
--------
1. User calls set_symbol("TSLA") and start().
2. Background thread polls for new 1m / 5m bars every 30 seconds.
3. On each bar:
   a. FLAT  → EntryDecider.decide() → place order if BUY or SELL_SHORT.
   b. LONG/SHORT → PositionManager.evaluate() → move stops / partial exits.
                   ExitManager.evaluate()    → decide whether to close.
4. All decisions are logged with a one-line explanation string.
5. Force-flatten fires at 3:45 PM ET regardless.

Thread safety: all state reads/writes are guarded by _lock.
"""
from __future__ import annotations

import logging
import threading
import time as _time
from datetime import datetime, date
from typing import Any, Callable, Literal

import pandas as pd

from app.services.strategy.daytrading.autotrader.entry_decider import EntryDecider
from app.services.strategy.daytrading.autotrader.exit_manager import ExitManager
from app.services.strategy.daytrading.autotrader.native_entry import (
    NativeStrategyEntry,
    SUPPORTED_NATIVE_STRATEGIES,
)
from app.services.strategy.daytrading.autotrader.position_manager import PositionManager, TrailMode
from app.services.strategy.daytrading.autotrader.trade_state import (
    State, TradeRecord, TradeStateMachine,
)
from app.services.strategy.daytrading.brain.market_state import (
    MarketStateResult, classify_market_state,
)
from app.services.strategy.daytrading.brain.risk_governor import RiskGovernor
from app.services.strategy.daytrading.market_open import ET, is_market_open, now_et
from app.services.strategy.daytrading.brain.symbol_policy import allows_live, get_policy

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC   = 30


class PolicyError(RuntimeError):
    """Raised when a symbol's deployment policy blocks live auto-trading."""     # how often to check for new bars
_COOLDOWN_AFTER_LOSS = 2      # bars to wait before re-entering after a loss
_REENTRY_BAR_LOCKOUT = 1      # bars to skip after any exit (win or loss)


class SingleStockTrader:
    """
    Full-lifecycle trader for one symbol.

    Parameters
    ----------
    symbol          : ticker to trade (e.g. "AAPL")
    broker          : BaseBroker instance (paper or live)
    direction_mode  : "long_only" | "short_only" | "both"
    trail_mode      : "ema" | "atr" | "candle"
    partial_tp      : take partial profits at +1R
    risk_per_trade_pct : fraction of capital to risk per trade
    max_daily_loss_pct : kill-switch if daily P&L drops below this %
    max_trades_per_day : hard cap on total trades today
    initial_capital : used for risk/size calculations
    """

    def __init__(
        self,
        symbol: str,
        broker=None,           # BaseBroker or None (paper sim mode)
        direction_mode: str = "long_only",
        trail_mode: TrailMode = "atr",
        partial_tp: bool = True,
        risk_per_trade_pct: float = 0.01,
        max_daily_loss_pct: float = 2.0,
        max_trades_per_day: int = 6,
        max_consecutive_losses: int = 3,
        initial_capital: float = 10_000.0,
        on_trade_update: Callable[[dict], None] | None = None,
        entry_mode: str = "legacy_entry_decider",
        native_strategies: list[str] | None = None,
    ):
        self.symbol = symbol.upper()
        self._broker = broker
        self.direction_mode = direction_mode
        self.initial_capital = initial_capital
        self.on_trade_update = on_trade_update  # callback for UI updates

        # Entry-path mode: "legacy_entry_decider" (default, unchanged behavior)
        # or "native_strategy" (delegate to strategy.generate_signals).
        if entry_mode not in ("legacy_entry_decider", "native_strategy"):
            raise ValueError(
                f"entry_mode must be 'legacy_entry_decider' or 'native_strategy', got {entry_mode!r}"
            )
        self.entry_mode = entry_mode

        # Core components
        self.tsm = TradeStateMachine()
        self.entry_decider = EntryDecider(
            direction_mode=direction_mode,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        self.native_entry = NativeStrategyEntry(
            direction_mode=direction_mode,
            risk_per_trade_pct=risk_per_trade_pct,
            native_strategies=native_strategies or list(SUPPORTED_NATIVE_STRATEGIES),
        )
        self.position_manager = PositionManager(
            partial_tp=partial_tp,
            trail_mode=trail_mode,
        )
        self.exit_manager = ExitManager()
        self.risk_governor = RiskGovernor(config={
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_trades_per_day": max_trades_per_day,
            "max_consecutive_losses": max_consecutive_losses,
        })

        # Live data snapshots
        self._df_1m: pd.DataFrame | None = None
        self._df_5m: pd.DataFrame | None = None
        self._df_15m: pd.DataFrame | None = None
        self._market_state: MarketStateResult | None = None

        # Status
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._last_bar_ts_5m: pd.Timestamp | None = None
        self._last_market_state_str = "UNKNOWN"
        self._last_heartbeat: datetime = datetime.now(ET)

        # Safety: cooldown and re-entry lockout
        self._bars_since_exit: int = 0           # increments each bar after exit
        self._last_exit_was_loss: bool = False

        # Trade log (all today's decisions, not just closed trades)
        self._decision_log: list[dict] = []
        self._last_no_trade_reason: str = ""
        self._active_trail_mode: str = ""        # updated by PositionManager

    # ── Public API ────────────────────────────────────────────────────────────

    def set_symbol(self, symbol: str) -> None:
        """Switch to a new symbol. Blocks if a position is open."""
        with self._lock:
            if self.tsm.has_position:
                raise RuntimeError(
                    f"Cannot change symbol while a position is open in {self.symbol}"
                )
            self.symbol = symbol.upper()
            self._df_1m = None
            self._df_5m = None
            self._df_15m = None
            logger.info("Symbol set to %s", self.symbol)

    def start(self, regime: str | None = None) -> None:
        """
        Start the background polling thread.
        Raises PolicyError if the symbol's deployment policy blocks live trading.
        Pass regime= if already known (e.g. from a prior spy-regime check).
        """
        if self._running:
            return
        ok, reason = allows_live(self.symbol, regime)
        if not ok:
            raise PolicyError(reason)
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"AutoTrader-{self.symbol}",
            daemon=True,
        )
        self._thread.start()
        logger.info("SingleStockTrader started for %s", self.symbol)

    def stop(self) -> None:
        """Stop the loop. Does NOT close any open position."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("SingleStockTrader stopped for %s", self.symbol)

    def force_flatten(self, reason: str = "Manual force flatten") -> None:
        """Close any open position immediately at market."""
        with self._lock:
            if not self.tsm.has_position:
                return
            close_price = self._last_price()
            self._execute_full_exit(close_price, reason)

    def get_status(self) -> dict[str, Any]:
        """Snapshot of current trader state — for the UI panel."""
        with self._lock:
            tsm = self.tsm
            unrealized = 0.0
            r_multiple = None
            if tsm.has_position:
                p = self._last_price()
                if p > 0:
                    if tsm.side == "LONG":
                        unrealized = (p - tsm.entry_price) * tsm.qty
                        risk = tsm.entry_price - tsm.initial_stop
                        r_multiple = (p - tsm.entry_price) / risk if risk > 0 else 0.0
                    else:
                        unrealized = (tsm.entry_price - p) * tsm.qty
                        risk = tsm.initial_stop - tsm.entry_price
                        r_multiple = (tsm.entry_price - p) / risk if risk > 0 else 0.0

            heartbeat_age_s = (datetime.now(ET) - self._last_heartbeat).total_seconds()

            return {
                "symbol": self.symbol,
                "state": tsm.state.value,
                "side": tsm.side or "—",
                "entry_price": round(tsm.entry_price, 4) if tsm.has_position else None,
                "current_stop": round(tsm.current_stop, 4) if tsm.has_position else None,
                "first_target": round(tsm.first_target, 4) if tsm.has_position else None,
                "trailing_stop": round(tsm.trailing_stop, 4) if tsm.has_position else None,
                "qty": tsm.qty,
                "unrealized_pnl": round(unrealized, 2),
                "r_multiple": round(r_multiple, 2) if r_multiple is not None else None,
                "realized_pnl": round(tsm.daily_pnl, 2),
                "trades_today": tsm.trades_today,
                "consecutive_losses": tsm.consecutive_losses,
                "strategy": tsm.strategy or "—",
                "entry_mode": self.entry_mode,
                "native_strategies": list(self.native_entry.native_strategies),
                "market_state": self._last_market_state_str,
                "management_profile": _describe_management_profile(
                    self._last_market_state_str, tsm.strategy or ""
                ),
                "active_trail_mode": self._active_trail_mode or "—",
                "running": self._running,
                "heartbeat_age_s": round(heartbeat_age_s, 0),
                "block_reason": tsm.block_reason,
                "last_entry_reason": tsm.entry_reason or "—",
                "last_no_trade_reason": self._last_no_trade_reason,
                "bars_since_exit": self._bars_since_exit,
                "cooldown_bars_remaining": max(
                    0,
                    (_COOLDOWN_AFTER_LOSS if self._last_exit_was_loss else _REENTRY_BAR_LOCKOUT)
                    - self._bars_since_exit
                ),
                "decision_log": list(self._decision_log[-20:]),
                "session_trades": [t.to_dict() for t in tsm.session_trades],
            }

    def on_new_bar(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Called externally when a new bar is available.
        Thread-safe; drives evaluation without the polling loop.
        """
        with self._lock:
            if timeframe == "1m":
                self._df_1m = df
            elif timeframe == "5m":
                self._df_5m = df
            elif timeframe == "15m":
                self._df_15m = df

        self._evaluate_cycle()

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while self._running:
            try:
                if is_market_open():
                    self._refresh_data()
                    self._evaluate_cycle()
            except Exception as e:
                logger.error("AutoTrader loop error: %s", e, exc_info=True)

            for _ in range(_POLL_INTERVAL_SEC * 2):
                if not self._running:
                    return
                _time.sleep(0.5)

    def _refresh_data(self) -> None:
        """Pull fresh bars from yfinance."""
        from app.services.strategy.daytrading.runner import fetch_intraday
        try:
            self._df_1m  = fetch_intraday(self.symbol, "1m",  "1d")
            self._df_5m  = fetch_intraday(self.symbol, "5m",  "5d")
            self._df_15m = fetch_intraday(self.symbol, "15m", "60d")
            # SPY for market state
            df_spy = fetch_intraday("SPY", "5m", "2d")
            if not df_spy.empty and not (self._df_5m is None or self._df_5m.empty):
                self._market_state = classify_market_state(self._df_5m, df_spy)
                self._last_market_state_str = self._market_state.state
        except Exception as e:
            logger.warning("Data refresh error: %s", e)

    def _evaluate_cycle(self) -> None:
        """One full evaluation: check state, decide, act."""
        with self._lock:
            df_5m = self._df_5m
            df_1m = self._df_1m
            df_15m = self._df_15m

            if df_5m is None or df_5m.empty:
                return

            # Detect new 5m bar (avoid processing same bar twice)
            last_ts = df_5m.index[-1] if not df_5m.empty else None
            if last_ts == self._last_bar_ts_5m:
                return
            self._last_bar_ts_5m = last_ts

            # EOD force-flatten check
            now_t = now_et().time()
            from app.services.strategy.daytrading.autotrader.exit_manager import _EOD_FORCE_FLAT
            if now_t >= _EOD_FORCE_FLAT and self.tsm.has_position:
                close = self._last_price()
                self._execute_full_exit(close, f"EOD force flatten at {now_t.strftime('%H:%M')}")
                return

            state = self.tsm.state
            self._last_heartbeat = datetime.now(ET)

            if state == State.FLAT:
                # Cooldown guard: wait N bars after exit before re-entering
                cooldown = (
                    _COOLDOWN_AFTER_LOSS if self._last_exit_was_loss
                    else _REENTRY_BAR_LOCKOUT
                )
                if self._bars_since_exit < cooldown:
                    self._bars_since_exit += 1
                    self._log(
                        "COOLDOWN",
                        f"Bar {self._bars_since_exit}/{cooldown} cooldown "
                        f"({'loss' if self._last_exit_was_loss else 'lockout'})",
                        "debug",
                    )
                    return
                self.evaluate_entry()
            elif state == State.BLOCKED:
                self._log("BLOCKED", self.tsm.block_reason, "warning")
            elif self.tsm.has_position:
                self._bars_since_exit = 0
                self.manage_open_trade()
            elif state == State.EXITED:
                self.tsm.reset_after_exit()
                self._log("RESET", "Trade closed — entering cooldown", "info")

    def evaluate_entry(self) -> None:
        """Called when FLAT — checks for a new entry signal."""
        # ── Risk governor check ────────────────────────────────────────────────
        risk_state = self.risk_governor.build_risk_state(
            today_trades=[t.to_dict() for t in self.tsm.session_trades],
            initial_capital=self.initial_capital,
            open_positions=0,
        )
        gov = self.risk_governor.check_can_trade(risk_state)
        if not gov.allowed:
            self.tsm.block(gov.reason)
            self._log("BLOCKED", gov.reason, "warn")
            return

        # ── Entry decision ─────────────────────────────────────────────────────
        # Dispatch on entry_mode: legacy_entry_decider keeps current behavior;
        # native_strategy delegates to strategy.generate_signals() so a named
        # strategy in live trading means the same thing as in backtest.
        decider = self.native_entry if self.entry_mode == "native_strategy" else self.entry_decider
        decision = decider.decide(
            symbol=self.symbol,
            df_1m=self._df_1m if self._df_1m is not None else pd.DataFrame(),
            df_5m=self._df_5m,
            df_15m=self._df_15m if self._df_15m is not None else pd.DataFrame(),
            market_state=self._market_state,
            account_equity=self.initial_capital + self.tsm.daily_pnl,
        )

        # Diagnostics: tag every decision with the mode that produced it so the
        # decision_log and UI panel can compare native vs legacy behavior.
        if isinstance(decision.checks, dict):
            decision.checks.setdefault("entry_mode", self.entry_mode)
            decision.checks.setdefault(
                "strategy_source",
                "native" if self.entry_mode == "native_strategy" else "legacy_scoring",
            )

        if not decision.is_tradeable:
            self._last_no_trade_reason = decision.entry_reason
            self._log(
                "NO_TRADE",
                f"[{self.entry_mode}] {decision.entry_reason}",
                "debug",
            )
            return

        # ── Compute position size ──────────────────────────────────────────────
        equity = self.initial_capital + self.tsm.daily_pnl
        price = self._last_price()
        if price <= 0:
            return

        risk_dollar = equity * self.entry_decider.risk_per_trade_pct
        stop_distance = abs(price - decision.stop_price)
        if stop_distance <= 0:
            return

        raw_qty = risk_dollar / stop_distance * decision.size_multiplier * gov.size_multiplier
        qty = max(1.0, round(raw_qty, 0))

        # ── Execute entry ──────────────────────────────────────────────────────
        side = "LONG" if decision.action == "BUY" else "SHORT"
        filled_price = self._place_order(decision.action, qty)
        if filled_price <= 0:
            return

        self.tsm.open_position(
            symbol=self.symbol,
            side=side,
            entry_price=filled_price,
            qty=qty,
            stop=decision.stop_price,
            target=decision.target_price,
            strategy=decision.chosen_strategy,
            entry_reason=decision.entry_reason,
        )
        self.position_manager.reset()
        self.exit_manager.reset()

        self._log(
            f"ENTRY {side}",
            f"{decision.entry_reason} | qty={qty:.0f} entry={filled_price:.2f} "
            f"stop={decision.stop_price:.2f} target={decision.target_price:.2f} "
            f"conf={decision.confidence:.2f}",
            "info",
        )
        self._notify_update()

    def manage_open_trade(self) -> None:
        """Called when LONG/SHORT — updates stops, manages exits."""
        df_5m = self._df_5m
        df_1m = self._df_1m
        ms_str = self._last_market_state_str

        # ── Position manager: stop moves / partial exits ───────────────────────
        pm_update = self.position_manager.evaluate(self.tsm, df_5m, df_1m, ms_str)
        if pm_update.action == "MOVE_STOP" and pm_update.new_stop:
            old = self.tsm.current_stop
            self.tsm.current_stop = pm_update.new_stop
            self.tsm.trailing_stop = pm_update.new_stop
            if self.tsm.state not in (State.TRAILING, State.PARTIAL_EXIT_TAKEN):
                self.tsm.transition(State.TRAILING, "Activating trail")
            self._log("MOVE_STOP", pm_update.reason, "info")
            self._notify_update()

        elif pm_update.action == "PARTIAL_EXIT" and pm_update.exit_qty > 0:
            close = self._last_price()
            exit_filled = self._place_exit_order(
                "SELL" if self.tsm.side == "LONG" else "BUY_COVER",
                pm_update.exit_qty,
            )
            if exit_filled > 0:
                self.tsm.qty -= pm_update.exit_qty
                if self.tsm.state not in (State.PARTIAL_EXIT_TAKEN,):
                    self.tsm.transition(State.PARTIAL_EXIT_TAKEN, pm_update.reason)
                self._log("PARTIAL_EXIT", pm_update.reason, "info")
                self._notify_update()

        elif pm_update.action == "ACTIVATE_TRAIL" and pm_update.new_stop:
            self.tsm.current_stop = pm_update.new_stop
            self.tsm.trailing_stop = pm_update.new_stop
            self.tsm.transition(State.TRAILING, pm_update.reason)
            self._active_trail_mode = pm_update.trail_mode_used
            self._log("TRAIL_ACTIVATED", pm_update.reason, "info")
            self._notify_update()

        # ── Exit manager: decide whether to close ─────────────────────────────
        ex_decision = self.exit_manager.evaluate(self.tsm, df_5m, df_1m, ms_str)
        if ex_decision.action == "FULL_EXIT":
            price = ex_decision.exit_price or self._last_price()
            self._execute_full_exit(price, ex_decision.reason)

        elif ex_decision.action == "MOVE_STOP" and ex_decision.new_stop:
            self.tsm.current_stop = ex_decision.new_stop
            self.tsm.trailing_stop = ex_decision.new_stop
            self._log("MOVE_STOP", ex_decision.reason, "info")
            self._notify_update()

    # ── Execution helpers ─────────────────────────────────────────────────────

    def _execute_full_exit(self, price: float, reason: str) -> None:
        """Close entire position at market and record the trade."""
        if not self.tsm.has_position:
            return
        action = "SELL" if self.tsm.side == "LONG" else "BUY_COVER"
        filled = self._place_exit_order(action, self.tsm.qty)
        actual_exit = filled if filled > 0 else price

        side_before_close = self.tsm.side
        record = self.tsm.close_position(
            exit_price=actual_exit,
            exit_time=datetime.now(ET),
            exit_reason=reason,
        )
        self._last_exit_was_loss = record.pnl < 0
        self._bars_since_exit = 0
        self._active_trail_mode = ""
        self._last_no_trade_reason = ""
        self._log(
            f"EXIT {side_before_close}",
            f"{reason} | pnl={record.pnl:+.2f} ({record.pnl_pct:+.3f}%) "
            f"@ {actual_exit:.2f}",
            "info",
        )
        self._notify_update()

    def _place_order(self, action: str, qty: float) -> float:
        """Submit entry order. Returns fill price (0.0 on failure).

        NOTE: this path calls the legacy broker interface directly, which means
        it does NOT go through ExecutionService — no DB Order row, no risk
        engine check, no audit event. That makes any live order fired here
        invisible in the Order History table. Until the path is rewritten to
        route through ExecutionService, live submission is hard-blocked. Paper
        sim mode still works.
        """
        if self._broker is None:
            # Paper sim: fill at last price
            return self._last_price()
        logger.error(
            "[autotrader] BLOCKED live order: %s %s qty=%s — legacy broker path "
            "is disabled because it bypasses ExecutionService (no DB row, no "
            "risk gate, no audit). Wire SingleStockTrader through "
            "ExecutionService before re-enabling.",
            action, self.symbol, qty,
        )
        return 0.0

    def _place_exit_order(self, action: str, qty: float) -> float:
        """Submit exit order. Returns fill price (0.0 on failure).

        See _place_order for the bypass-block rationale.
        """
        if self._broker is None:
            return self._last_price()
        logger.error(
            "[autotrader] BLOCKED live exit: %s %s qty=%s — legacy broker path "
            "is disabled because it bypasses ExecutionService.",
            action, self.symbol, qty,
        )
        return 0.0

    def _last_price(self) -> float:
        """Return the latest close from 5m data."""
        if self._df_5m is not None and not self._df_5m.empty:
            return float(self._df_5m["Close"].iloc[-1])
        if self._df_1m is not None and not self._df_1m.empty:
            return float(self._df_1m["Close"].iloc[-1])
        return 0.0

    def _log(self, event: str, reason: str, level: str = "info") -> None:
        """Append a structured decision log entry."""
        now = now_et().strftime("%H:%M:%S")
        entry = {
            "time": now,
            "event": event,
            "symbol": self.symbol,
            "state": self.tsm.state.value,
            "reason": reason,
        }
        self._decision_log.append(entry)
        if len(self._decision_log) > 200:
            self._decision_log = self._decision_log[-200:]

        log_fn = getattr(logger, level if level in ("info", "warning", "error", "debug") else "info")
        log_fn("[%s] %s %s — %s", now, event, self.symbol, reason)

    def _notify_update(self) -> None:
        if self.on_trade_update:
            try:
                self.on_trade_update(self.get_status())
            except Exception:
                pass


def _describe_management_profile(market_state: str, strategy: str) -> str:
    """One-line human description of the active trade management regime."""
    regime_desc = {
        "TREND_UP":   "Let winners run — wide trail, needs 3 fade signals",
        "TREND_DOWN": "Let winners run — wide trail, needs 3 fade signals",
        "CHOPPY":     "Quick profits — tight trail, exits on 2 fade signals",
        "HIGH_VOL":   "Ultra-tight — exits on first reversal signal",
        "NEWS_RISK":  "Flatten immediately — news risk active",
        "UNKNOWN":    "Default — moderate settings",
    }.get(market_state, "Default")

    strategy_desc = {
        "ORBBreakout":       "25% partial, trail @ +1.5R",
        "VWAPMeanReversion": "50% partial, trail @ +1.0R",
        "EMAMomentum":       "33% partial, trail @ +1.25R",
    }.get(strategy, "")

    if strategy_desc:
        return f"{regime_desc} | {strategy_desc}"
    return regime_desc
