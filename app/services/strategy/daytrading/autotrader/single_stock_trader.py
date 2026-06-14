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
from app.services.strategy.daytrading.market_open import (
    ET, IST, is_market_open, now_et, market_session,
)
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
        max_trades_per_day: int = 0,   # 0 = unlimited (no artificial trade-count cap)
        max_consecutive_losses: int = 3,
        initial_capital: float = 10_000.0,
        on_trade_update: Callable[[dict], None] | None = None,
        entry_mode: str = "legacy_entry_decider",
        native_strategies: list[str] | None = None,
        execution_service=None,   # ExecutionService | None — required for live orders
        account_id: str = "",     # broker account ID passed to ExecutionService
        tight_trail_on_exit_signal: bool = True,  # ride momentum after a sell signal
        account_type: str = "cash",   # "cash" | "margin" — drives PDT guard
        pdt_guard: bool = True,        # master switch for the PDT block (no-op when exempt)
        max_open_positions: int = 1,   # concurrent positions (forwarded to RiskGovernor)
    ):
        self.symbol = symbol.upper()
        self._broker = broker
        self._execution_service = execution_service
        self._account_id = account_id
        self.direction_mode = direction_mode
        self.initial_capital = initial_capital
        self.on_trade_update = on_trade_update  # callback for UI updates
        # When True, a momentum-fade exit signal does NOT immediately market-sell.
        # Instead it arms a tight trailing stop FLOORED at the signal price, so we
        # ride any further momentum while guaranteeing we never exit below the
        # price where the sell signal fired.
        self.tight_trail_on_exit_signal = tight_trail_on_exit_signal

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
        self.exit_manager = ExitManager(symbol=symbol)
        self.risk_governor = RiskGovernor(config={
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_trades_per_day": max_trades_per_day,
            "max_consecutive_losses": max_consecutive_losses,
            "max_open_positions": max_open_positions,
        })

        # PDT (Pattern Day Trader) guard. Paper mode is always exempt; for live
        # the tracker decides exemption from account_type + equity at gate time.
        # broker is None ⇒ paper sim; otherwise treat as a real broker.
        self.account_type = (account_type or "cash").lower()
        self.pdt_guard = bool(pdt_guard)
        # broker is None ⇒ in-process paper sim. A real BaseBroker exposes
        # .is_paper (True for PaperBroker / Alpaca paper).
        self._is_paper_account = broker is None or bool(getattr(broker, "is_paper", False))

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

        # Safety: cooldown and re-entry lockout.
        # Seed high so a brand-new bot is NOT held in a phantom cooldown on its
        # very first bar (there was no prior exit to cool down from). The real
        # cooldown is set to 0 only after an actual exit in _execute_full_exit.
        self._bars_since_exit: int = 999
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
                "tight_trail_enabled": self.tight_trail_on_exit_signal,
                "tight_trail_armed": tsm.tight_trail_armed,
                "tight_trail_floor": round(tsm.tight_trail_floor, 4) if tsm.tight_trail_armed else None,
                "tight_trail_signal_reason": tsm.tight_trail_signal_reason,
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
                "pdt": self.pdt_status(),
            }

    def decision_summary(self, limit: int | None = None) -> dict[str, Any]:
        """Aggregate the recent decision_log into a native-vs-legacy report.

        Read-only — touches no execution state, just the existing log entries
        written by `_log(... checks=decision.checks)`. Each FLAT cycle records
        ONE entry whose `checks.entry_mode` is the live mode and whose
        `checks.shadow` is what the OTHER mode would have done.

        The summary classifies each cycle into one of four agreement buckets:
          - both_tradeable     : live tradeable AND shadow tradeable
          - native_only        : only the native side tradeable
          - legacy_only        : only the legacy side tradeable
          - neither_tradeable  : both said NO_TRADE

        It also aggregates the native rejection-category histogram, top
        legacy low-score reason snippets, the native winning-strategy
        frequency, and a chronological list of disagreement cycles for
        targeted chart review.

        Parameters
        ----------
        limit : int | None
            Look at only the most-recent N decision_log entries. None = all.
        """
        with self._lock:
            log = list(self._decision_log)
        if limit is not None and limit > 0:
            log = log[-limit:]

        # Only entries that actually represent an entry-evaluation cycle.
        cycles = [e for e in log if e.get("event") in ("NO_TRADE",) or str(e.get("event", "")).startswith("ENTRY")]

        total = len(cycles)
        native_tradeable = 0
        legacy_tradeable = 0
        both = 0
        native_only = 0
        legacy_only = 0
        neither = 0
        native_strategy_count: dict[str, int] = {}
        native_reject_categories: dict[str, int] = {}
        native_gate_counts: dict[str, int] = {}
        legacy_low_score_counts: dict[str, int] = {}
        disagreements: list[dict[str, Any]] = []

        for entry in cycles:
            checks = entry.get("checks") or {}
            if not isinstance(checks, dict):
                continue
            live_mode = checks.get("entry_mode")
            event = str(entry.get("event", ""))
            live_tradeable = event.startswith("ENTRY")

            shadow = checks.get("shadow") or {}
            shadow_tradeable = bool(shadow.get("tradeable", False))

            # Map live + shadow to native/legacy tradeability.
            if live_mode == "native_strategy":
                native_now = live_tradeable
                legacy_now = shadow_tradeable
            else:
                native_now = shadow_tradeable
                legacy_now = live_tradeable

            native_tradeable += int(native_now)
            legacy_tradeable += int(legacy_now)
            if native_now and legacy_now:
                both += 1
            elif native_now and not legacy_now:
                native_only += 1
            elif legacy_now and not native_now:
                legacy_only += 1
            else:
                neither += 1

            # Native rejection-category histogram (from whichever side native was on).
            native_checks = checks if live_mode == "native_strategy" else (shadow.get("checks") or {})
            if isinstance(native_checks, dict):
                gate = native_checks.get("gate")
                if gate:
                    native_gate_counts[gate] = native_gate_counts.get(gate, 0) + 1
                cats = native_checks.get("native_rejection_categories") or {}
                if isinstance(cats, dict):
                    for cat in cats.values():
                        native_reject_categories[cat] = native_reject_categories.get(cat, 0) + 1
                winner = native_checks.get("winning_strategy")
                if winner:
                    native_strategy_count[winner] = native_strategy_count.get(winner, 0) + 1

            # Legacy "low score" reason histogram. Legacy NO_TRADE reasons look
            # like "Long score 0.28 < min 0.45; Short score 0.12 < min 0.45".
            legacy_checks = checks if live_mode == "legacy_entry_decider" else (shadow.get("checks") or {})
            legacy_reason = entry.get("reason") if live_mode == "legacy_entry_decider" else shadow.get("reason")
            if isinstance(legacy_reason, str) and not (legacy_now):
                token = _classify_legacy_reason(legacy_reason)
                if token:
                    legacy_low_score_counts[token] = legacy_low_score_counts.get(token, 0) + 1

            # Track disagreement cycles for review.
            if (native_now and not legacy_now) or (legacy_now and not native_now):
                disagreements.append({
                    "time": entry.get("time"),
                    "side_traded": "native" if native_now else "legacy",
                    "native_chosen_strategy": (native_checks or {}).get("winning_strategy"),
                    "live_reason": entry.get("reason"),
                    "shadow_reason": shadow.get("reason"),
                })

        return {
            "symbol": self.symbol,
            "entry_mode_live": self.entry_mode,
            "cycles_observed": total,
            "native_tradeable": native_tradeable,
            "legacy_tradeable": legacy_tradeable,
            "agreement": {
                "both_tradeable": both,
                "native_only": native_only,
                "legacy_only": legacy_only,
                "neither_tradeable": neither,
            },
            "native_rejection_categories": _sorted_hist(native_reject_categories),
            "native_gate_counts": _sorted_hist(native_gate_counts),
            "native_winning_strategy_freq": _sorted_hist(native_strategy_count),
            "legacy_low_score_reasons": _sorted_hist(legacy_low_score_counts),
            "disagreements": disagreements[-20:],   # cap for response size
            "disagreement_count": len(disagreements),
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
            # Always update heartbeat so the UI knows the thread is alive,
            # even when the market is closed.
            self._last_heartbeat = datetime.now(ET)
            try:
                if is_market_open(self.symbol):
                    self._refresh_data()
                    self._evaluate_cycle()
                else:
                    # Market closed — log once, keep thread alive.
                    logger.debug("AutoTrader idle — market closed for %s", self.symbol)
            except Exception as e:
                logger.error("AutoTrader loop error: %s", e, exc_info=True)

            for _ in range(_POLL_INTERVAL_SEC * 2):
                if not self._running:
                    return
                _time.sleep(0.5)

    def _refresh_data(self) -> None:
        """Pull fresh bars. Uses Upstox for Indian symbols, yfinance for US."""
        from app.services.strategy.daytrading.runner import fetch_intraday
        from app.services.markets import is_india_symbol
        try:
            self._df_1m  = fetch_intraday(self.symbol, "1m",  "1d")
            self._df_5m  = fetch_intraday(self.symbol, "5m",  "5d")
            self._df_15m = fetch_intraday(self.symbol, "15m", "60d")
            # Use the correct market index for regime classification
            _ref_sym = "^NSEI" if is_india_symbol(self.symbol) else "SPY"
            try:
                df_ref = fetch_intraday(_ref_sym, "5m", "2d")
            except Exception:
                df_ref = None
            if not (self._df_5m is None or self._df_5m.empty):
                self._market_state = classify_market_state(
                    self._df_5m,
                    df_ref if df_ref is not None and not df_ref.empty else None,
                )
                self._last_market_state_str = self._market_state.state
                logger.debug("Market state: %s (conf=%.0f%%)",
                             self._last_market_state_str,
                             self._market_state.confidence * 100)
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

            # EOD force-flatten check (market-aware: IST for NSE, ET for US)
            from app.services.strategy.daytrading.autotrader.exit_manager import (
                _EOD_FORCE_FLAT, _EOD_FORCE_FLAT_IST,
            )
            from datetime import datetime as _dt
            _sess = market_session(self.symbol)
            now_t = _dt.now(_sess.tz).time()
            _eod_flat = _EOD_FORCE_FLAT_IST if _sess.tz is IST else _EOD_FORCE_FLAT
            if now_t >= _eod_flat and self.tsm.has_position:
                close = self._last_price()
                _tz_label = "IST" if _sess.tz is IST else "ET"
                self._execute_full_exit(close, f"EOD force flatten at {now_t.strftime('%H:%M')} {_tz_label}")
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

        # ── PDT guard (live sub-$25k margin only; no-op otherwise) ─────────────
        pdt = self._check_pdt()
        if pdt is not None and not pdt.allowed:
            # Do NOT permanently BLOCK the day here — the rolling window can free
            # up, and exits on existing positions must still run. Just skip this
            # entry cycle with a clear, logged reason.
            self._last_no_trade_reason = pdt.reason
            self._log("PDT_BLOCK", pdt.reason, "warning")
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

        # Shadow run the OTHER mode read-only so the decision log shows what
        # native would have done while legacy is live (or vice versa). Never
        # executes — just populates checks["shadow"].
        shadow = self._shadow_decide()
        if shadow is not None and isinstance(decision.checks, dict):
            decision.checks["shadow"] = shadow

        if not decision.is_tradeable:
            self._last_no_trade_reason = decision.entry_reason
            self._log(
                "NO_TRADE",
                f"[{self.entry_mode}] {decision.entry_reason}",
                "debug",
                checks=decision.checks if isinstance(decision.checks, dict) else None,
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
            exit_plan=getattr(decision, "exit_plan", None),
        )
        self.position_manager.reset()
        self.exit_manager.reset()

        self._log(
            f"ENTRY {side}",
            f"[{self.entry_mode}] {decision.entry_reason} | qty={qty:.0f} entry={filled_price:.2f} "
            f"stop={decision.stop_price:.2f} target={decision.target_price:.2f} "
            f"conf={decision.confidence:.2f}",
            "info",
            checks=decision.checks if isinstance(decision.checks, dict) else None,
        )
        self._notify_update()

    def manage_open_trade(self) -> None:
        """Called when LONG/SHORT — updates stops, manages exits."""
        df_5m = self._df_5m
        df_1m = self._df_1m
        ms_str = self._last_market_state_str

        # Guard: track how many partial exits fire this bar across both managers.
        # If both PositionManager and ExitManager return PARTIAL_EXIT in the same
        # bar, skip the ExitManager one (same-bar double scale-out is always wrong).
        _partial_exits_this_bar = 0

        # ── Position manager: stop moves / partial exits ───────────────────────
        pm_update = self.position_manager.evaluate(self.tsm, df_5m, df_1m, ms_str)
        if pm_update.action == "MOVE_STOP" and pm_update.new_stop:
            self.tsm.current_stop = pm_update.new_stop
            self.tsm.trailing_stop = pm_update.new_stop
            if self.tsm.state not in (State.TRAILING, State.PARTIAL_EXIT_TAKEN):
                self.tsm.transition(State.TRAILING, "Activating trail")
            self._log("MOVE_STOP", pm_update.reason, "info")
            self._notify_update()

        elif pm_update.action == "PARTIAL_EXIT" and pm_update.exit_qty > 0:
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
                _partial_exits_this_bar += 1

        elif pm_update.action == "ACTIVATE_TRAIL" and pm_update.new_stop:
            self.tsm.current_stop = pm_update.new_stop
            self.tsm.trailing_stop = pm_update.new_stop
            self.tsm.transition(State.TRAILING, pm_update.reason)
            self._active_trail_mode = pm_update.trail_mode_used
            self._log("TRAIL_ACTIVATED", pm_update.reason, "info")
            self._notify_update()

        # ── Tight-trail update: once armed, ratchet the floor as price rises ──
        # The floor (signal price) is the worst case; the trail can only improve.
        if self.tsm.tight_trail_armed:
            self._update_tight_trail(df_5m, df_1m, ms_str)

        # ── Exit manager: decide whether to close ─────────────────────────────
        ex_decision = self.exit_manager.evaluate(self.tsm, df_5m, df_1m, ms_str)
        if ex_decision.action == "FULL_EXIT":
            price = ex_decision.exit_price or self._last_price()

            # Momentum-fade exit (fade_signals populated) → instead of an
            # immediate market sell, arm a tight profit-protecting trail floored
            # at the signal price so we ride any further momentum but never exit
            # below where the sell signal fired. Hard stops / trailing-stop hits
            # (no fade_signals) always execute immediately.
            _is_momentum_fade = bool(getattr(ex_decision, "fade_signals", None))
            if self.tsm.tight_trail_armed and _is_momentum_fade:
                # Already riding a tight trail — repeated fade signals are
                # expected; let the trailing floor govern the exit, don't force.
                self._log(
                    "TIGHT_TRAIL_HOLD",
                    f"Repeat fade signal while tight trail armed "
                    f"(floor={self.tsm.tight_trail_floor:.2f}) — holding for momentum",
                    "debug",
                )
            elif (
                self.tight_trail_on_exit_signal
                and _is_momentum_fade
                and not self.tsm.tight_trail_armed
                and self._position_in_profit()
            ):
                self._arm_tight_trail(price, ex_decision.reason, ex_decision.fade_signals)
            else:
                self._execute_full_exit(price, ex_decision.reason)

        elif ex_decision.action == "PARTIAL_EXIT":
            if _partial_exits_this_bar > 0:
                logger.warning(
                    "[%s] double PARTIAL_EXIT in one bar — skipping ExitManager tier "
                    "(PositionManager already scaled out this bar)",
                    self.symbol,
                )
            else:
                self._execute_partial_exit_from_exit_manager(ex_decision)
                _partial_exits_this_bar += 1

        elif ex_decision.action == "MOVE_STOP" and ex_decision.new_stop:
            self.tsm.current_stop = ex_decision.new_stop
            self.tsm.trailing_stop = ex_decision.new_stop
            self._log("MOVE_STOP", ex_decision.reason, "info")
            self._notify_update()

    # ── Execution helpers ─────────────────────────────────────────────────────

    def _execute_partial_exit_from_exit_manager(self, ex_decision: Any) -> None:
        """Scale out the position for one ExitPlan tier returned by ExitManager.

        ExitManager.evaluate() calls tsm.advance_scale_level() before returning
        PARTIAL_EXIT, so _scale_level_idx already points to the NEXT unpassed
        tier.  The tier we just consumed is therefore at index _scale_level_idx - 1.

        Option B (no exit_qty on ExitDecision): derive pct_to_close from
        exit_plan.scale_levels[consumed_idx] so ExitDecision stays clean.
        """
        if not self.tsm.has_position:
            return

        ep = self.tsm.exit_plan
        consumed_idx = self.tsm._scale_level_idx - 1   # advance_scale_level already called

        if ep is None or consumed_idx < 0 or consumed_idx >= len(ep.scale_levels):
            # Fallback: no plan — treat as FULL_EXIT so position doesn't stall
            logger.warning(
                "[%s] PARTIAL_EXIT from ExitManager but no valid ExitPlan scale level "
                "(ep=%s, consumed_idx=%d) — falling back to FULL_EXIT",
                self.symbol, ep, consumed_idx,
            )
            price = getattr(ex_decision, "exit_price", None) or self._last_price()
            self._execute_full_exit(price, ex_decision.reason + " [no plan — forced full]")
            return

        pct = ep.scale_levels[consumed_idx].pct_to_close
        exit_qty = max(1.0, round(self.tsm.qty * pct, 0))
        # Safety cap: never exit more than we hold
        exit_qty = min(exit_qty, self.tsm.qty)

        action = "SELL" if self.tsm.side == "LONG" else "BUY_COVER"
        fill_px = self._place_exit_order(action, exit_qty)

        if fill_px <= 0:
            # Order rejected — roll back the scale level so ExitManager retries next bar
            self.tsm._scale_level_idx = max(0, self.tsm._scale_level_idx - 1)
            self._log(
                "PARTIAL_EXIT_FAILED",
                f"Scale tier {consumed_idx + 1}: order rejected, rolled back idx. {ex_decision.reason}",
                "warning",
            )
            return

        self.tsm.qty -= exit_qty

        if self.tsm.state not in (State.PARTIAL_EXIT_TAKEN, State.TRAILING):
            self.tsm.transition(State.PARTIAL_EXIT_TAKEN, ex_decision.reason)

        self._log(
            "PARTIAL_EXIT",
            (
                f"Scale tier {consumed_idx + 1}/{len(ep.scale_levels)}: "
                f"{pct:.0%} × {exit_qty:.0f} shares @ {fill_px:.4f} | "
                f"{ex_decision.reason} | "
                f"remaining qty={self.tsm.qty:.0f}"
            ),
            "info",
        )
        self._notify_update()

        # If all scale levels consumed and no runner planned, close the remainder now
        if (
            self.tsm._scale_level_idx >= len(ep.scale_levels)
            and ep.trail_type == "none"
            and self.tsm.qty > 0
        ):
            price = getattr(ex_decision, "exit_price", None) or self._last_price()
            self._execute_full_exit(
                price,
                f"All scale levels consumed, no runner (trail_type=none) — full close",
            )

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

    # ── Tight-trail-on-exit-signal ────────────────────────────────────────────

    def _position_in_profit(self) -> bool:
        """True if the open position is currently profitable vs entry."""
        if not self.tsm.has_position:
            return False
        price = self._last_price()
        if price <= 0:
            return False
        if self.tsm.side == "LONG":
            return price > self.tsm.entry_price
        return price < self.tsm.entry_price

    def _arm_tight_trail(self, signal_price: float, fade_reason: str,
                         fade_signals: list[str]) -> None:
        """Convert a momentum-fade SELL signal into a profit-protecting tight trail.

        Instead of an immediate market sell, we set a tight trailing stop floored
        at ``signal_price`` (the price the sell signal fired at). For a LONG, the
        stop can only ratchet UP from here — so the worst-case exit is exactly the
        signal price (never below it), and any further momentum is captured.
        """
        floor = round(signal_price, 4)
        self.tsm.tight_trail_armed = True
        self.tsm.tight_trail_floor = floor
        self.tsm.tight_trail_signal_reason = fade_reason

        # Set the initial trailing stop AT the floor (signal price). For a LONG
        # this is below current price; if price reverses immediately we exit here,
        # locking the profit the sell signal identified.
        self.tsm.current_stop = floor
        self.tsm.trailing_stop = floor
        if self.tsm.state != State.TRAILING:
            self.tsm.transition(State.TRAILING, "Tight trail armed on exit signal")
        self._active_trail_mode = "tight_signal_floor"

        _sig = ", ".join(fade_signals) if fade_signals else fade_reason
        self._log(
            "TIGHT_TRAIL_ARMED",
            f"Sell signal at {floor:.2f} → armed tight trail (floor={floor:.2f}, "
            f"riding momentum). Will NOT exit below {floor:.2f}. Signals: {_sig}",
            "info",
        )
        self._notify_update()

    def _update_tight_trail(self, df_5m, df_1m, ms_str: str) -> None:
        """Ratchet the tight trail upward (LONG) / downward (SHORT) as price moves.

        Uses a tight candle-based trail but NEVER lets the stop fall below the
        armed floor (the original sell-signal price).
        """
        price = self._last_price()
        if price <= 0:
            return

        # Compute a tight candle trail from the current bar structure
        try:
            from app.services.strategy.daytrading.autotrader.position_manager import (
                _candle_trail,
            )
            cand = _candle_trail(price, self.tsm.side, df_5m)
        except Exception:
            cand = None

        floor = self.tsm.tight_trail_floor
        new_stop = self.tsm.current_stop

        if self.tsm.side == "LONG":
            # Trail can only go up; never below the signal-price floor
            candidate = max(floor, cand) if cand else floor
            if candidate > new_stop:
                new_stop = candidate
        else:  # SHORT
            candidate = min(floor, cand) if cand else floor
            if candidate < new_stop:
                new_stop = candidate

        if new_stop != self.tsm.current_stop:
            self.tsm.current_stop = round(new_stop, 4)
            self.tsm.trailing_stop = self.tsm.current_stop
            self._log(
                "TIGHT_TRAIL_MOVE",
                f"Tight trail → {self.tsm.current_stop:.2f} "
                f"(floor={floor:.2f}, price={price:.2f}) — locking momentum gains",
                "debug",
            )
            self._notify_update()

    def _place_order(self, action: str, qty: float) -> float:
        """Submit an entry order. Returns fill price (0.0 on failure).

        Paper mode (broker is None): fills immediately at last bar close.
        Live mode: routes through ExecutionService for full risk gating,
        DB persistence, audit logging, and broker submission.
        """
        if self._broker is None:
            # Paper sim — fill at last price, no broker needed.
            return self._last_price()

        if self._execution_service is None:
            logger.error(
                "[autotrader] BLOCKED live order: %s %s qty=%s — "
                "no ExecutionService configured. Pass execution_service= "
                "when constructing SingleStockTrader for live trading.",
                action, self.symbol, qty,
            )
            return 0.0

        return self._submit_via_execution_service(action, qty, is_exit=False)

    def _place_exit_order(self, action: str, qty: float) -> float:
        """Submit an exit order. Returns fill price (0.0 on failure).

        Same routing as _place_order — paper fills at last price,
        live goes through ExecutionService.
        """
        if self._broker is None:
            return self._last_price()

        if self._execution_service is None:
            logger.error(
                "[autotrader] BLOCKED live exit: %s %s qty=%s — "
                "no ExecutionService configured.",
                action, self.symbol, qty,
            )
            return 0.0

        return self._submit_via_execution_service(action, qty, is_exit=True)

    def _submit_via_execution_service(
        self, action: str, qty: float, *, is_exit: bool
    ) -> float:
        """Route an order through ExecutionService (live mode only).

        Runs the full 8-gate pipeline: risk check → buying power → persist →
        preview → submit → confirm → fill → protective stop.

        The polling loop runs in a daemon thread so we use asyncio.run() to
        drive the coroutine to completion without touching the main event loop.
        """
        import asyncio
        import uuid
        from app.schemas.orders import OrderRequest

        side = action.upper()
        if side not in ("BUY", "SELL"):
            logger.error("[autotrader] Unknown order action: %s", action)
            return 0.0

        idempotency_key = (
            f"autotrader-{'exit' if is_exit else 'entry'}"
            f"-{self.symbol}-{side}-{int(qty * 100)}"
            f"-{int(self._last_price() * 100)}"
        )

        order_req = OrderRequest(
            symbol=self.symbol,
            side=side,
            order_type="MARKET",
            quantity=qty,
            time_in_force="DAY",
            source="autotrader",
            idempotency_key=idempotency_key,
        )

        try:
            db_order = asyncio.run(
                self._execution_service.execute(
                    order_req,
                    account_id=self._account_id,
                    estimated_price=self._last_price(),
                )
            )
        except Exception as exc:
            logger.error(
                "[autotrader] ExecutionService raised for %s %s: %s",
                side, self.symbol, exc,
            )
            return 0.0

        if db_order is None:
            # Risk engine or buying-power blocked the order — already logged
            # by ExecutionService; just return 0 so the caller treats it as
            # a no-fill and leaves the state machine in its current state.
            logger.warning(
                "[autotrader] %s %s blocked by risk gate or buying power.",
                side, self.symbol,
            )
            return 0.0

        # Use the recorded fill_price if available, else fall back to last bar.
        fill_price = getattr(db_order, "fill_price", None)
        if fill_price and float(fill_price) > 0:
            return float(fill_price)
        return self._last_price()

    # ── PDT guard ──────────────────────────────────────────────────────────────

    def _pdt_equity(self) -> float:
        """Best-effort current account equity for the $25k PDT exemption test.

        Uses initial_capital + realized session P&L. This is intentionally simple
        and conservative — if the true equity is higher, the worst case is the
        guard staying on slightly longer than strictly required, which is the safe
        direction for a compliance check.
        """
        return self.initial_capital + self.tsm.daily_pnl

    def _pdt_round_trips(self) -> list[dict]:
        """Pull recent live round-trips from the realized-trades table for the
        rolling PDT window. Returns an empty list on any failure (fail-open is
        acceptable because the guard only ever *adds* a restriction; a transient
        DB hiccup must not wedge live trading).
        """
        from datetime import timedelta
        try:
            from app.db import SessionLocal
            from app.models.realized_trades import RealizedTrade
            cutoff = datetime.now(ET) - timedelta(days=10)  # > 5 biz days of slack
            with SessionLocal() as db:
                rows = (
                    db.query(RealizedTrade)
                    .filter(RealizedTrade.symbol == self.symbol)
                    .filter(RealizedTrade.sell_at >= cutoff)
                    .filter(RealizedTrade.is_paper.is_(False))
                    .all()
                )
                return [
                    {"symbol": r.symbol, "buy_at": r.buy_at, "sell_at": r.sell_at}
                    for r in rows
                ]
        except Exception as e:
            logger.debug("[pdt] round-trip fetch failed: %s", e)
            return []

    def _check_pdt(self):
        """Return a PDTDecision when the guard is enabled, else None.

        No-op (returns None) for paper accounts or when pdt_guard is off, so the
        common path pays nothing. The PDTTracker itself also short-circuits when
        the account is cash or funded ≥ $25k.
        """
        if not self.pdt_guard or self._is_paper_account:
            return None
        try:
            from app.services.strategy.daytrading.brain.pdt_tracker import PDTTracker
            tracker = PDTTracker(
                account_type=self.account_type,
                equity=self._pdt_equity(),
                is_paper=self._is_paper_account,
            )
            return tracker.check_can_open_day_trade(
                self._pdt_round_trips(), symbol=self.symbol
            )
        except Exception as e:
            logger.warning("[pdt] guard evaluation failed (fail-open): %s", e)
            return None

    def pdt_status(self) -> dict[str, Any]:
        """Read-only PDT standing for this symbol — surfaced by the status API."""
        try:
            from app.services.strategy.daytrading.brain.pdt_tracker import PDTTracker
            tracker = PDTTracker(
                account_type=self.account_type,
                equity=self._pdt_equity(),
                is_paper=self._is_paper_account,
            )
            return tracker.build_status(self._pdt_round_trips()).to_dict()
        except Exception as e:
            logger.debug("[pdt] status build failed: %s", e)
            return {"guard_active": False, "exempt_reason": f"unavailable: {e}"}

    def _last_price(self) -> float:
        """Return the latest close from 5m data."""
        if self._df_5m is not None and not self._df_5m.empty:
            return float(self._df_5m["Close"].iloc[-1])
        if self._df_1m is not None and not self._df_1m.empty:
            return float(self._df_1m["Close"].iloc[-1])
        return 0.0

    def _log(
        self,
        event: str,
        reason: str,
        level: str = "info",
        checks: dict | None = None,
    ) -> None:
        """Append a structured decision log entry.

        When `checks` is provided (typically `decision.checks`), it lands in the
        log entry so the status endpoint can show legacy_scoring vs native
        comparisons (long_score/short_score for legacy, native_rejections and
        winning_strategy for native, plus shadow snapshot of the other mode).
        """
        now = now_et().strftime("%H:%M:%S")
        entry: dict[str, Any] = {
            "time": now,
            "event": event,
            "symbol": self.symbol,
            "state": self.tsm.state.value,
            "reason": reason,
        }
        if checks:
            entry["checks"] = checks
        self._decision_log.append(entry)
        if len(self._decision_log) > 200:
            self._decision_log = self._decision_log[-200:]

        log_fn = getattr(logger, level if level in ("info", "warning", "error", "debug") else "info")
        log_fn("[%s] %s %s — %s", now, event, self.symbol, reason)

    def _shadow_decide(self) -> dict | None:
        """Run the inactive decider read-only for diagnostic comparison.

        Returns a compact dict (not a full EntryDecision) so the decision log
        stays small. Never executes — purely informational. Errors are swallowed
        so a shadow failure can't block the live path.
        """
        try:
            other = self.entry_decider if self.entry_mode == "native_strategy" else self.native_entry
            other_mode_name = (
                "legacy_entry_decider" if self.entry_mode == "native_strategy" else "native_strategy"
            )
            shadow = other.decide(
                symbol=self.symbol,
                df_1m=self._df_1m if self._df_1m is not None else pd.DataFrame(),
                df_5m=self._df_5m,
                df_15m=self._df_15m if self._df_15m is not None else pd.DataFrame(),
                market_state=self._market_state,
                account_equity=self.initial_capital + self.tsm.daily_pnl,
            )
            return {
                "mode": other_mode_name,
                "action": shadow.action,
                "tradeable": shadow.is_tradeable,
                "confidence": round(shadow.confidence, 3),
                "chosen_strategy": shadow.chosen_strategy,
                "reason": shadow.entry_reason,
                "stop_price": shadow.stop_price,
                "target_price": shadow.target_price,
                "checks": shadow.checks if isinstance(shadow.checks, dict) else {},
            }
        except Exception as e:
            logger.debug("[shadow] decide failed: %s", e)
            return {"mode": "shadow", "error": f"{type(e).__name__}: {e}"}

    def _notify_update(self) -> None:
        if self.on_trade_update:
            try:
                self.on_trade_update(self.get_status())
            except Exception:
                pass


def _sorted_hist(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Return a histogram dict as a sorted list of {category, count} for
    deterministic JSON output. Empty input → empty list."""
    return [
        {"category": k, "count": v}
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _classify_legacy_reason(reason: str) -> str | None:
    """Bucket free-text legacy NO_TRADE reasons into stable category tokens.

    Mirrors the strings EntryDecider.decide() emits. Returns None when the
    reason doesn't match a known bucket so noise stays out of the histogram.
    """
    r = reason.lower()
    if "too late in day" in r:
        return "late_entry_window"
    if "long score" in r and "short score" in r:
        return "both_scores_below_min"
    if "long score" in r:
        return "long_score_below_min"
    if "short score" in r:
        return "short_score_below_min"
    if "r:r" in r and "minimum" in r:
        return "rr_below_min"
    if "too extended" in r:
        return "price_too_extended"
    if "news_risk" in r:
        return "news_risk"
    if "stop" in r and "not below entry" in r:
        return "invalid_stop"
    if "insufficient" in r:
        return "insufficient_bars"
    if "invalid stop" in r:
        return "invalid_stop_target"
    return None


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
        "ORBBreakout":       "Scale: 40%@ORB×1.2 → 30%@ORB×2.0 → EMA9 runner",
        "VWAPMeanReversion": "Scale: 60%@VWAP → 30%@VWAP+overshoot → small runner",
        "EMAMomentum":       "Scale: 40%@1.5R → 30%@2.5R → EMA trail",
        "OpeningGapFade":    "Scale: 60%@60%fill → ATR trail on runner",
        "SupertrendTrend":   "Scale: 30%@2R → ST-line trail to flip",
        "NRSqueezeBreakout": "Scale: 35%@2R → 30%@3.5R → structure trail",
    }.get(strategy, "")

    if strategy_desc:
        return f"{regime_desc} | {strategy_desc}"
    return regime_desc
