"""
Risk Governor — enforces hard daily trading limits.

Rules enforced:
  - max_daily_loss_pct: if daily P&L drops below this %, kill switch triggers
  - max_consecutive_losses: after N losses in a row, stop trading
  - max_trades_per_day: hard cap on total trades
  - max_open_positions: never hold more than this simultaneously (default 1)
  - size_reduction_after_loss: after any loss, reduce size to this fraction
    until a winner resets the counter

The governor is stateless per session — it is rebuilt from the day's trade log
each time it is called, so it is correct even if the process restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskState:
    # Current session counts
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0   # as % of starting capital
    open_positions: int = 0

    # Kill switch
    kill_switch_triggered: bool = False
    kill_switch_reason: str = ""

    # Size state
    size_multiplier: float = 1.0  # 1.0 = full, 0.5 = half after loss

    # Config snapshot (for explain)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernorDecision:
    allowed: bool
    reason: str
    size_multiplier: float        # 0.0 = blocked, 0.5 = half, 1.0 = full
    risk_state: RiskState


class RiskGovernor:
    """
    Stateless evaluator — call check_can_trade() before each new signal.
    Build RiskState from today's completed trade log each time.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "max_daily_loss_pct": 2.0,        # stop if down 2% on the day
        "max_consecutive_losses": 3,       # stop after 3 losses in a row
        "max_trades_per_day": 6,           # no more than 6 trades per day
        "max_open_positions": 1,           # never have 2 open at once
        "size_reduction_after_loss": 0.5,  # half size after any loss
        "size_reset_after_win": True,      # full size restored after a winner
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def build_risk_state(
        self,
        today_trades: list[dict],
        initial_capital: float,
        open_positions: int = 0,
    ) -> RiskState:
        """
        Reconstruct today's risk state from the closed trade log.
        today_trades: list of trade dicts with keys pnl, pnl_pct, outcome.
        """
        state = RiskState(open_positions=open_positions, config=self.config)
        state.trades_today = len(today_trades)

        consecutive = 0
        for t in today_trades:
            pnl = float(t.get("pnl", 0))
            state.daily_pnl += pnl
            if pnl > 0:
                state.wins_today += 1
                consecutive = 0
                if self.config["size_reset_after_win"]:
                    state.size_multiplier = 1.0
            else:
                state.losses_today += 1
                consecutive += 1
                state.size_multiplier = self.config["size_reduction_after_loss"]

        state.consecutive_losses = consecutive
        state.daily_pnl_pct = (state.daily_pnl / initial_capital * 100) if initial_capital > 0 else 0.0

        # Evaluate kill switch
        if state.daily_pnl_pct <= -self.config["max_daily_loss_pct"]:
            state.kill_switch_triggered = True
            state.kill_switch_reason = (
                f"Daily loss limit hit: {state.daily_pnl_pct:.2f}% "
                f"(max {self.config['max_daily_loss_pct']}%)."
            )
        elif state.consecutive_losses >= self.config["max_consecutive_losses"]:
            state.kill_switch_triggered = True
            state.kill_switch_reason = (
                f"{state.consecutive_losses} consecutive losses — "
                f"max allowed is {self.config['max_consecutive_losses']}."
            )
        elif state.trades_today >= self.config["max_trades_per_day"]:
            state.kill_switch_triggered = True
            state.kill_switch_reason = (
                f"Max trades per day reached: {state.trades_today} "
                f"(limit {self.config['max_trades_per_day']})."
            )

        return state

    def check_can_trade(
        self,
        risk_state: RiskState,
        routing_size_multiplier: float = 1.0,
    ) -> GovernorDecision:
        """
        Given current RiskState, decide if a new trade is allowed.
        Returns GovernorDecision with combined size multiplier.
        """
        if risk_state.kill_switch_triggered:
            return GovernorDecision(
                allowed=False,
                reason=f"KILL SWITCH: {risk_state.kill_switch_reason}",
                size_multiplier=0.0,
                risk_state=risk_state,
            )

        if risk_state.open_positions >= self.config["max_open_positions"]:
            return GovernorDecision(
                allowed=False,
                reason=f"Max open positions ({self.config['max_open_positions']}) already reached.",
                size_multiplier=0.0,
                risk_state=risk_state,
            )

        # Combine risk governor size with routing size
        combined_size = risk_state.size_multiplier * routing_size_multiplier
        combined_size = round(max(0.0, min(1.0, combined_size)), 2)

        reason_parts = []
        if risk_state.size_multiplier < 1.0:
            reason_parts.append(
                f"Size reduced to {risk_state.size_multiplier:.0%} after recent loss."
            )
        if routing_size_multiplier < 1.0:
            reason_parts.append(
                f"Market-state size multiplier: {routing_size_multiplier:.0%}."
            )

        reason = " ".join(reason_parts) if reason_parts else "All risk checks passed."

        return GovernorDecision(
            allowed=True,
            reason=reason,
            size_multiplier=combined_size,
            risk_state=risk_state,
        )
