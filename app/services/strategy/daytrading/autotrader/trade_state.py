"""
TradeState — deterministic state machine for a single intraday position.

States
------
FLAT             No position open, accepting entries.
PENDING_ENTRY    Entry order submitted but not yet confirmed filled.
LONG             Long position open.
SHORT            Short position open.
PARTIAL_EXIT_TAKEN  Scaled out ≥25% after reaching first target.
TRAILING         Beyond first target; trailing stop is active.
EXITED           Trade closed this session (win or loss).
BLOCKED          Risk governor has killed trading for the day.

Valid transitions are enforced; illegal moves raise ValueError so bugs
surface immediately rather than silently corrupting position state.
"""
from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.strategy.daytrading.risk_templates import ExitPlan


class State(str, Enum):
    FLAT             = "FLAT"
    PENDING_ENTRY    = "PENDING_ENTRY"
    LONG             = "LONG"
    SHORT            = "SHORT"
    PARTIAL_EXIT_TAKEN = "PARTIAL_EXIT_TAKEN"
    TRAILING         = "TRAILING"
    EXITED           = "EXITED"
    BLOCKED          = "BLOCKED"


# Legal (from_state, to_state) transitions
_TRANSITIONS: set[tuple[State, State]] = {
    (State.FLAT,              State.PENDING_ENTRY),
    (State.FLAT,              State.LONG),          # sync fill (paper)
    (State.FLAT,              State.SHORT),
    (State.FLAT,              State.BLOCKED),
    (State.PENDING_ENTRY,     State.LONG),
    (State.PENDING_ENTRY,     State.SHORT),
    (State.PENDING_ENTRY,     State.FLAT),           # fill rejected / cancelled
    (State.LONG,              State.PARTIAL_EXIT_TAKEN),
    (State.LONG,              State.TRAILING),
    (State.LONG,              State.EXITED),
    (State.SHORT,             State.PARTIAL_EXIT_TAKEN),
    (State.SHORT,             State.TRAILING),
    (State.SHORT,             State.EXITED),
    (State.PARTIAL_EXIT_TAKEN, State.TRAILING),
    (State.PARTIAL_EXIT_TAKEN, State.EXITED),
    (State.TRAILING,          State.EXITED),
    (State.EXITED,            State.FLAT),           # reset for next session / test
    (State.BLOCKED,           State.FLAT),           # end-of-day reset
}


@dataclass
class TradeRecord:
    """Immutable snapshot of one completed trade for the session log."""
    symbol: str
    side: str               # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy: str
    entry_reason: str
    initial_stop: float
    final_stop: float
    max_favorable_excursion: float  # MFE in $
    max_adverse_excursion: float    # MAE in $

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "qty": self.qty,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "exit_reason": self.exit_reason,
            "strategy": self.strategy,
            "entry_reason": self.entry_reason,
            "initial_stop": round(self.initial_stop, 4),
            "final_stop": round(self.final_stop, 4),
            "mfe": round(self.max_favorable_excursion, 2),
            "mae": round(self.max_adverse_excursion, 2),
        }


@dataclass
class TradeStateMachine:
    """
    Owns the current state and the active position fields.

    All field mutations happen through transition() — never set .state directly.
    """

    state: State = State.FLAT

    # ── Active position fields (only meaningful in LONG/SHORT/PARTIAL/TRAILING) ──
    symbol: str = ""
    side: str = ""                  # "LONG" | "SHORT"
    entry_price: float = 0.0
    entry_time: datetime | None = None
    qty: float = 0.0
    initial_qty: float = 0.0

    # Stop / target levels
    initial_stop: float = 0.0
    current_stop: float = 0.0
    first_target: float = 0.0
    trailing_stop: float = 0.0

    # Tight-trail-on-exit-signal: when a sell-signal arms a profit-protecting
    # trail instead of an immediate market exit, this floors the stop at the
    # signal price so a LONG never exits below (SHORT never above) it.
    tight_trail_armed: bool = False
    tight_trail_floor: float = 0.0   # signal price the trail must never cross
    tight_trail_signal_reason: str = ""

    # Excursion tracking
    max_favorable_excursion: float = 0.0   # highest unrealised gain reached
    max_adverse_excursion: float = 0.0     # deepest unrealised loss reached

    # Metadata
    strategy: str = ""
    entry_reason: str = ""
    block_reason: str = ""
    # Structured exit plan from risk_templates; None = legacy single-target behaviour
    exit_plan: "ExitPlan | None" = None
    # Index into exit_plan.scale_levels tracking which scale-out tier is next
    _scale_level_idx: int = 0

    # Session history (all closed trades today)
    session_trades: list[TradeRecord] = field(default_factory=list)

    # Audit trail: list of (timestamp, from_state, to_state, reason)
    _history: list[tuple[datetime, str, str, str]] = field(default_factory=list)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        return self.state == State.FLAT

    @property
    def has_position(self) -> bool:
        return self.state in (
            State.LONG, State.SHORT,
            State.PARTIAL_EXIT_TAKEN, State.TRAILING,
        )

    @property
    def in_long(self) -> bool:
        return self.side == "LONG" and self.has_position

    @property
    def in_short(self) -> bool:
        return self.side == "SHORT" and self.has_position

    @property
    def trades_today(self) -> int:
        return len(self.session_trades)

    @property
    def daily_pnl(self) -> float:
        return sum(t.pnl for t in self.session_trades)

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for t in reversed(self.session_trades):
            if t.pnl <= 0:
                count += 1
            else:
                break
        return count

    # ── State machine ─────────────────────────────────────────────────────────

    def transition(self, new_state: State, reason: str = "") -> None:
        """Advance to new_state, enforcing valid transitions."""
        if (self.state, new_state) not in _TRANSITIONS:
            raise ValueError(
                f"Illegal transition {self.state} -> {new_state}: {reason}"
            )
        old = self.state
        self.state = new_state
        self._history.append((datetime.now(), old.value, new_state.value, reason))

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        stop: float,
        target: float,
        strategy: str,
        entry_reason: str,
        exit_plan: "ExitPlan | None" = None,
    ) -> None:
        """Transition FLAT/PENDING → LONG/SHORT and populate position fields."""
        new_state = State.LONG if side == "LONG" else State.SHORT
        self.transition(new_state, f"Opened {side} @ {entry_price:.2f}")

        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.qty = qty
        self.initial_qty = qty
        self.initial_stop = stop
        self.current_stop = stop
        self.trailing_stop = stop
        self.first_target = target
        self.strategy = strategy
        self.entry_reason = entry_reason
        self.exit_plan = exit_plan
        self._scale_level_idx = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0

    def close_position(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> TradeRecord:
        """Record trade result, transition to EXITED, return TradeRecord."""
        if self.side == "LONG":
            pnl = (exit_price - self.entry_price) * self.qty
            pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            pnl = (self.entry_price - exit_price) * self.qty
            pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

        record = TradeRecord(
            symbol=self.symbol,
            side=self.side,
            entry_price=self.entry_price,
            exit_price=exit_price,
            qty=self.initial_qty,
            entry_time=self.entry_time or exit_time,
            exit_time=exit_time,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=exit_reason,
            strategy=self.strategy,
            entry_reason=self.entry_reason,
            initial_stop=self.initial_stop,
            final_stop=self.current_stop,
            max_favorable_excursion=self.max_favorable_excursion,
            max_adverse_excursion=self.max_adverse_excursion,
        )
        self.session_trades.append(record)
        self.transition(State.EXITED, exit_reason)
        return record

    def update_excursion(self, current_price: float) -> None:
        """Call on every new bar to track MFE / MAE."""
        if not self.has_position:
            return
        if self.side == "LONG":
            gain = (current_price - self.entry_price) * self.qty
        else:
            gain = (self.entry_price - current_price) * self.qty
        if gain > 0:
            self.max_favorable_excursion = max(self.max_favorable_excursion, gain)
        else:
            self.max_adverse_excursion = max(self.max_adverse_excursion, abs(gain))

    def reset_after_exit(self) -> None:
        """Transition EXITED → FLAT, clear position fields."""
        self.transition(State.FLAT, "Reset after exit")
        self.symbol = ""
        self.side = ""
        self.entry_price = 0.0
        self.entry_time = None
        self.qty = 0.0
        self.initial_qty = 0.0
        self.initial_stop = 0.0
        self.current_stop = 0.0
        self.first_target = 0.0
        self.trailing_stop = 0.0
        self.tight_trail_armed = False
        self.tight_trail_floor = 0.0
        self.tight_trail_signal_reason = ""
        self.strategy = ""
        self.entry_reason = ""
        self.exit_plan = None
        self._scale_level_idx = 0

    @property
    def next_scale_level(self):
        """Return the next unpassed ScaleLevel, or None if all taken."""
        if self.exit_plan is None:
            return None
        levels = self.exit_plan.scale_levels
        if self._scale_level_idx >= len(levels):
            return None
        return levels[self._scale_level_idx]

    def advance_scale_level(self) -> None:
        """Mark the current scale level as consumed."""
        self._scale_level_idx += 1

    def block(self, reason: str) -> None:
        """Transition FLAT → BLOCKED; sets block reason."""
        self.transition(State.BLOCKED, reason)
        self.block_reason = reason

    def unblock(self) -> None:
        """Transition BLOCKED → FLAT (end-of-day reset)."""
        self.transition(State.FLAT, "Unblocked")
        self.block_reason = ""

    def history_lines(self) -> list[str]:
        return [
            f"{ts.strftime('%H:%M:%S')}  {frm} -> {to}  [{reason}]"
            for ts, frm, to, reason in self._history
        ]
