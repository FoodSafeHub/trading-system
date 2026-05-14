"""
DayTradingBrain — top-level orchestrator that ties together all brain sub-modules.

Call order for a live signal:
  1. evaluate_market_state()  → MarketStateResult
  2. route_strategies()       → RoutingDecision   (which strategies allowed)
  3. governor.check_can_trade() → GovernorDecision (risk/daily limits)
  4. execution_guard.validate() → GuardDecision    (signal-level checks)
  5. performance_memory weight  → size adjustment
  6. allocate_position_size()  → final $ size

For backtesting, pass apply_brain=True to filter_signals().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from app.services.strategy.daytrading.brain.execution_guard import ExecutionGuard
from app.services.strategy.daytrading.brain.market_state import (
    MarketStateResult,
    classify_market_state,
)
from app.services.strategy.daytrading.brain.performance_memory import (
    PerformanceMemory,
    get_global_memory,
)
from app.services.strategy.daytrading.brain.risk_governor import (
    GovernorDecision,
    RiskGovernor,
    RiskState,
)
from app.services.strategy.daytrading.brain.strategy_router import (
    RoutingDecision,
    route_strategies,
)
from app.services.strategy.daytrading.models import DayTradeSignal


@dataclass
class BrainDecision:
    """Full record of what the brain decided for one signal and why."""
    accepted: bool
    strategy: str
    symbol: str
    market_state: str
    state_confidence: float
    size_multiplier: float          # 0.0–1.0 combined
    rejection_reason: str           # "" if accepted
    explanation: str                # human-readable summary
    signal: dict | None = None      # original signal dict
    checks: dict[str, Any] = field(default_factory=dict)   # all sub-module outputs


@dataclass
class BrainStatus:
    """Snapshot of brain state — for the UI panel."""
    market_state: str
    state_confidence: float
    state_reasons: list[str]
    enabled_strategies: list[str]
    disabled_strategies: list[str]
    disabled_reasons: dict[str, str]
    kill_switch: bool
    kill_switch_reason: str
    trades_today: int
    losses_in_a_row: int
    daily_pnl_pct: float
    size_multiplier: float          # routing × governor combined
    performance_stats: list[dict]   # from PerformanceMemory
    routing_summary: str


class DayTradingBrain:
    """
    Orchestrates all brain sub-modules into a single decision pipeline.

    Usage:
        brain = DayTradingBrain()
        status = brain.evaluate_market_state(df_5m, df_spy_5m)
        decisions = brain.filter_signals(signals, account_state, market_context)
    """

    def __init__(
        self,
        governor_config: dict | None = None,
        guard_config: dict | None = None,
        memory: PerformanceMemory | None = None,
    ):
        self.governor = RiskGovernor(config=governor_config)
        self.guard = ExecutionGuard(config=guard_config)
        self.memory = memory or get_global_memory()
        self._last_market_state: MarketStateResult | None = None
        self._last_routing: RoutingDecision | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_market_state(
        self,
        df_5m: pd.DataFrame,
        df_spy_5m: pd.DataFrame | None = None,
        user_disabled: list[str] | None = None,
    ) -> BrainStatus:
        """
        Classify the market and route strategies.
        Returns a BrainStatus suitable for the UI panel.
        Does NOT require trade history — call this on startup or each bar.
        """
        ms = classify_market_state(df_5m, df_spy_5m)
        routing = route_strategies(ms, user_disabled=user_disabled)
        self._last_market_state = ms
        self._last_routing = routing

        return BrainStatus(
            market_state=ms.state,
            state_confidence=ms.confidence,
            state_reasons=ms.reasons,
            enabled_strategies=routing.enabled,
            disabled_strategies=routing.disabled,
            disabled_reasons=routing.disabled_reasons,
            kill_switch=False,
            kill_switch_reason="",
            trades_today=0,
            losses_in_a_row=0,
            daily_pnl_pct=0.0,
            size_multiplier=routing.size_multiplier,
            performance_stats=self.memory.get_all_stats(),
            routing_summary=routing.routing_summary,
        )

    def build_status(
        self,
        df_5m: pd.DataFrame,
        df_spy_5m: pd.DataFrame | None = None,
        today_trades: list[dict] | None = None,
        initial_capital: float = 100_000.0,
        open_positions: int = 0,
        user_disabled: list[str] | None = None,
    ) -> BrainStatus:
        """Full status including risk governor state."""
        ms = classify_market_state(df_5m, df_spy_5m)
        routing = route_strategies(ms, user_disabled=user_disabled)
        self._last_market_state = ms
        self._last_routing = routing

        risk_state = self.governor.build_risk_state(
            today_trades or [], initial_capital, open_positions
        )

        combined_size = round(routing.size_multiplier * risk_state.size_multiplier, 2)

        return BrainStatus(
            market_state=ms.state,
            state_confidence=ms.confidence,
            state_reasons=ms.reasons,
            enabled_strategies=routing.enabled,
            disabled_strategies=routing.disabled,
            disabled_reasons=routing.disabled_reasons,
            kill_switch=risk_state.kill_switch_triggered,
            kill_switch_reason=risk_state.kill_switch_reason,
            trades_today=risk_state.trades_today,
            losses_in_a_row=risk_state.consecutive_losses,
            daily_pnl_pct=risk_state.daily_pnl_pct,
            size_multiplier=combined_size,
            performance_stats=self.memory.get_all_stats(),
            routing_summary=routing.routing_summary,
        )

    def filter_signals(
        self,
        signals: list[dict | DayTradeSignal],
        account_state: dict | None = None,
        market_context: dict | None = None,
        current_bar_time: datetime | None = None,
    ) -> list[BrainDecision]:
        """
        Run every signal through the full brain pipeline.
        Returns a BrainDecision for each input signal (accepted or rejected).

        account_state keys: today_trades (list), initial_capital, open_positions
        market_context keys: df_5m, df_spy_5m (optional, for fresh state classification)
        """
        account = account_state or {}
        today_trades: list[dict] = account.get("today_trades", [])
        initial_capital: float = float(account.get("initial_capital", 100_000.0))
        open_positions: int = int(account.get("open_positions", 0))

        # Classify market state if fresh data provided
        if market_context and "df_5m" in market_context:
            ms = classify_market_state(
                market_context["df_5m"],
                market_context.get("df_spy_5m"),
            )
            routing = route_strategies(ms)
            self._last_market_state = ms
            self._last_routing = routing
        else:
            ms = self._last_market_state
            routing = self._last_routing

        if ms is None or routing is None:
            # No market state available — pass all signals with a note
            return [
                self._accept_no_state(s) for s in signals
            ]

        risk_state = self.governor.build_risk_state(
            today_trades, initial_capital, open_positions
        )

        decisions: list[BrainDecision] = []
        for sig in signals:
            d = self._evaluate_one(sig, ms, routing, risk_state, current_bar_time)
            decisions.append(d)
            # If accepted, update open position count for subsequent signals
            if d.accepted:
                open_positions += 1
                risk_state.open_positions = open_positions
                risk_state.trades_today += 1

        return decisions

    def should_trade_now(
        self,
        strategy: str,
        today_trades: list[dict] | None = None,
        initial_capital: float = 100_000.0,
        open_positions: int = 0,
    ) -> tuple[bool, str]:
        """
        Quick boolean check — is it safe to take a new trade right now?
        Returns (allowed, reason).
        """
        risk_state = self.governor.build_risk_state(
            today_trades or [], initial_capital, open_positions
        )
        routing = self._last_routing

        if routing and strategy not in routing.enabled:
            return False, routing.disabled_reasons.get(strategy, f"{strategy} not allowed in current market state.")

        gov = self.governor.check_can_trade(
            risk_state,
            routing_size_multiplier=routing.size_multiplier if routing else 1.0,
        )
        return gov.allowed, gov.reason

    def should_stop_for_day(
        self,
        today_trades: list[dict],
        initial_capital: float = 100_000.0,
    ) -> tuple[bool, str]:
        """Returns (stop, reason) — True means kill switch is on."""
        risk_state = self.governor.build_risk_state(today_trades, initial_capital)
        return risk_state.kill_switch_triggered, risk_state.kill_switch_reason

    def allocate_position_size(
        self,
        signal: dict | DayTradeSignal,
        account_state: dict,
        risk_state: RiskState | None = None,
    ) -> dict:
        """
        Compute dollar and share size for a signal.
        Returns: {shares, dollar_size, risk_per_share, risk_dollars, size_multiplier}
        """
        capital = float(account_state.get("capital", account_state.get("initial_capital", 100_000.0)))
        risk_pct = float(account_state.get("risk_per_trade_pct", 1.0))  # 1% default

        if isinstance(signal, dict):
            entry = float(signal.get("entry_price", 0))
            stop = float(signal.get("stop_price", 0))
        else:
            entry = signal.entry_price
            stop = signal.stop_price

        risk_per_share = abs(entry - stop) if entry > 0 and stop > 0 else 0.0

        # Combine routing + governor multipliers
        routing_mult = self._last_routing.size_multiplier if self._last_routing else 1.0
        gov_mult = risk_state.size_multiplier if risk_state else 1.0

        # Memory weight for this strategy/symbol/state
        strategy = signal.get("strategy", "") if isinstance(signal, dict) else signal.strategy
        symbol = signal.get("symbol", "") if isinstance(signal, dict) else signal.symbol
        state_name = self._last_market_state.state if self._last_market_state else "UNKNOWN"
        mem_weight = self.memory.get_strategy_weight(strategy, symbol, state_name)

        combined_mult = round(routing_mult * gov_mult * mem_weight, 2)
        risk_dollars = capital * (risk_pct / 100) * combined_mult

        shares = int(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
        dollar_size = round(shares * entry, 2)

        return {
            "shares": shares,
            "dollar_size": dollar_size,
            "risk_per_share": round(risk_per_share, 4),
            "risk_dollars": round(risk_dollars, 2),
            "size_multiplier": combined_mult,
        }

    def explain_decision(self, decision: BrainDecision) -> str:
        """Return a long-form explanation suitable for the UI expander."""
        lines = [
            f"**Strategy**: {decision.strategy}  |  **Symbol**: {decision.symbol}",
            f"**Market State**: {decision.market_state} (confidence {decision.state_confidence:.0%})",
            f"**Decision**: {'✅ ACCEPTED' if decision.accepted else '❌ REJECTED'}",
            "",
        ]
        if not decision.accepted:
            lines.append(f"**Rejection Reason**: {decision.rejection_reason}")
            lines.append("")

        checks = decision.checks
        if "routing" in checks:
            r = checks["routing"]
            lines.append(f"**Routing**: {r.get('reason', '')}")
        if "governor" in checks:
            g = checks["governor"]
            lines.append(f"**Risk Governor**: {g.get('reason', '')}")
        if "guard" in checks:
            gd = checks["guard"]
            lines.append(f"**Execution Guard**: {gd.get('reason', '')}")
            guard_checks = gd.get("checks", {})
            for k, v in guard_checks.items():
                mark = "✓" if v else "✗"
                lines.append(f"  {mark} {k}")
        if "memory_weight" in checks:
            lines.append(f"**Performance Weight**: {checks['memory_weight']:.0%} (historical edge in this state)")

        lines.append("")
        lines.append(f"**Combined Size Multiplier**: {decision.size_multiplier:.0%}")
        return "\n".join(lines)

    def record_trade(self, trade: dict) -> None:
        """Pass a completed trade to performance memory."""
        self.memory.record_trade(trade)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_one(
        self,
        sig: dict | DayTradeSignal,
        ms: MarketStateResult,
        routing: RoutingDecision,
        risk_state: RiskState,
        current_bar_time: datetime | None,
    ) -> BrainDecision:
        sig_dict = sig if isinstance(sig, dict) else _signal_to_dict(sig)
        strategy = sig_dict.get("strategy", "")
        symbol = sig_dict.get("symbol", "")
        checks: dict[str, Any] = {}

        # ── 1. Strategy routing ───────────────────────────────────────────────
        if strategy not in routing.enabled:
            reason = routing.disabled_reasons.get(strategy, f"{strategy} not allowed in {ms.state}.")
            checks["routing"] = {"allowed": False, "reason": reason}
            return BrainDecision(
                accepted=False,
                strategy=strategy,
                symbol=symbol,
                market_state=ms.state,
                state_confidence=ms.confidence,
                size_multiplier=0.0,
                rejection_reason=reason,
                explanation=f"Routing blocked: {reason}",
                signal=sig_dict,
                checks=checks,
            )
        checks["routing"] = {"allowed": True, "reason": routing.enabled_reasons.get(strategy, "Allowed.")}

        # ── 2. Risk governor ──────────────────────────────────────────────────
        gov_decision = self.governor.check_can_trade(risk_state, routing.size_multiplier)
        checks["governor"] = {"allowed": gov_decision.allowed, "reason": gov_decision.reason}
        if not gov_decision.allowed:
            return BrainDecision(
                accepted=False,
                strategy=strategy,
                symbol=symbol,
                market_state=ms.state,
                state_confidence=ms.confidence,
                size_multiplier=0.0,
                rejection_reason=gov_decision.reason,
                explanation=f"Risk governor blocked: {gov_decision.reason}",
                signal=sig_dict,
                checks=checks,
            )

        # ── 3. Execution guard ────────────────────────────────────────────────
        guard_result = self.guard.validate(sig_dict, current_bar_time)
        checks["guard"] = {
            "accepted": guard_result.accepted,
            "reason": guard_result.reason,
            "checks": guard_result.checks,
        }
        if not guard_result.accepted:
            return BrainDecision(
                accepted=False,
                strategy=strategy,
                symbol=symbol,
                market_state=ms.state,
                state_confidence=ms.confidence,
                size_multiplier=0.0,
                rejection_reason=guard_result.reason,
                explanation=f"Execution guard blocked: {guard_result.reason}",
                signal=sig_dict,
                checks=checks,
            )

        # ── 4. Performance memory weight ──────────────────────────────────────
        mem_weight = self.memory.get_strategy_weight(strategy, symbol, ms.state)
        checks["memory_weight"] = mem_weight
        if mem_weight == 0.0:
            reason = self.memory.explain_weight(strategy, symbol, ms.state)
            return BrainDecision(
                accepted=False,
                strategy=strategy,
                symbol=symbol,
                market_state=ms.state,
                state_confidence=ms.confidence,
                size_multiplier=0.0,
                rejection_reason=f"Performance memory disabled: {reason}",
                explanation=reason,
                signal=sig_dict,
                checks=checks,
            )

        # ── All checks passed ─────────────────────────────────────────────────
        combined_size = round(gov_decision.size_multiplier * mem_weight, 2)
        explanation = (
            f"{strategy} on {symbol} accepted in {ms.state} "
            f"(conf {ms.confidence:.0%}). "
            f"Size: {combined_size:.0%}. "
            f"{guard_result.reason}"
        )

        return BrainDecision(
            accepted=True,
            strategy=strategy,
            symbol=symbol,
            market_state=ms.state,
            state_confidence=ms.confidence,
            size_multiplier=combined_size,
            rejection_reason="",
            explanation=explanation,
            signal=sig_dict,
            checks=checks,
        )

    def _accept_no_state(self, sig: dict | DayTradeSignal) -> BrainDecision:
        sig_dict = sig if isinstance(sig, dict) else _signal_to_dict(sig)
        return BrainDecision(
            accepted=True,
            strategy=sig_dict.get("strategy", ""),
            symbol=sig_dict.get("symbol", ""),
            market_state="UNKNOWN",
            state_confidence=0.0,
            size_multiplier=0.5,   # half size when state unknown
            rejection_reason="",
            explanation="No market state available — accepted at half size.",
            signal=sig_dict,
        )


def _signal_to_dict(sig: DayTradeSignal) -> dict:
    return {
        "strategy": sig.strategy,
        "symbol": sig.symbol,
        "direction": sig.direction,
        "entry_price": sig.entry_price,
        "stop_price": sig.stop_price,
        "target_price": sig.target_price,
        "confidence": sig.confidence,
        "signal_time": str(sig.signal_time) if sig.signal_time else "",
        "indicators": sig.indicators or {},
        "reason": sig.reason,
        "regime": sig.regime,
        "r_multiple": sig.r_multiple,
        "risk_reward": sig.risk_reward,
    }
