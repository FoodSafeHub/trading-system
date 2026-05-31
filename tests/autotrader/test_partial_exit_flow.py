"""
Tests for the ExitPlan scale-out wiring in SingleStockTrader.

All tests run in paper-sim mode (broker=None) with synthetic 5m bars.
No network, no real broker, no thread — we call manage_open_trade() directly.

Scenarios
---------
A. US (SPY / US_ETF):
   Entry → tier-1 PARTIAL_EXIT → tier-2 PARTIAL_EXIT → FULL_EXIT (stop on runner)

B. US retrace:
   Entry → tier-1 PARTIAL_EXIT (BE move) → stop hit on remaining qty

C. NSE VWAP (RELIANCE / NSE_LARGE_CAP):
   Entry → two fixed scale tiers → all-flat (trail_type="none" triggers full close)
"""
from __future__ import annotations

import types
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
import pytz

from app.services.strategy.daytrading.autotrader.single_stock_trader import SingleStockTrader
from app.services.strategy.daytrading.autotrader.trade_state import State, TradeStateMachine
from app.services.strategy.daytrading.autotrader.exit_manager import ExitDecision
from app.services.strategy.daytrading.autotrader.position_manager import PositionUpdate
from app.services.strategy.daytrading.risk_templates import (
    ExitPlan,
    ScaleLevel,
    orb_exit_plan,
    vwap_exit_plan,
)

ET = pytz.timezone("America/New_York")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_5m_bars(n: int, base: float = 450.0, step: float = 0.0) -> pd.DataFrame:
    """Synthetic 5m bars with enough structure for ATR / volume indicators."""
    start = datetime(2024, 6, 3, 9, 30, tzinfo=ET)
    idx = pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(n)])
    prices = [base + step * i for i in range(n)]
    return pd.DataFrame(
        {
            "Open":   prices,
            "High":   [p + 0.50 for p in prices],
            "Low":    [p - 0.50 for p in prices],
            "Close":  prices,
            "Volume": [2_000_000] * n,
        },
        index=idx,
    )


def _make_trader(symbol: str = "SPY") -> SingleStockTrader:
    """SingleStockTrader in paper-sim (broker=None) with all defaults."""
    t = SingleStockTrader(
        symbol=symbol,
        broker=None,
        direction_mode="long_only",
        initial_capital=50_000.0,
    )
    # Supply synthetic bars so the trader never needs network access
    t._df_5m = _make_5m_bars(40)
    t._df_1m = None
    t._df_15m = _make_5m_bars(20)
    t._last_market_state_str = "TREND_UP"
    return t


def _open_long(
    trader: SingleStockTrader,
    entry: float,
    stop: float,
    target: float,
    qty: float,
    exit_plan: ExitPlan | None = None,
) -> None:
    """Directly open a LONG position on the TSM, bypassing entry evaluation."""
    trader.position_manager.reset()
    trader.exit_manager.reset()
    trader.tsm.open_position(
        symbol=trader.symbol,
        side="LONG",
        entry_price=entry,
        qty=qty,
        stop=stop,
        target=target,
        strategy="ORBBreakout",
        entry_reason="test setup",
        exit_plan=exit_plan,
    )


# ── Scenario A: two scale-out tiers then stop on runner ───────────────────────

def test_scenario_a_two_tier_scale_then_stop():
    """
    ENTRY → PARTIAL_EXIT tier1 (40%) → PARTIAL_EXIT tier2 (30%) → FULL_EXIT (stop).
    After tier1: qty=60, state=PARTIAL_EXIT_TAKEN.
    After tier2: qty=42, state=PARTIAL_EXIT_TAKEN.
    After stop:  qty=0,  state=EXITED.
    No net-short at any step.
    """
    trader = _make_trader("SPY")
    entry, stop, orb_range = 450.0, 448.50, 2.0   # risk = 1.50/share
    ep = orb_exit_plan("US_ETF", entry=entry, stop=stop, orb_range=orb_range)

    _open_long(trader, entry=entry, stop=stop, target=entry + orb_range * 2.0,
               qty=100.0, exit_plan=ep)
    assert trader.tsm.state == State.LONG
    assert trader.tsm.qty == 100.0

    # ── Tier 1: ExitManager returns PARTIAL_EXIT ──────────────────────────────
    tier1_decision = ExitDecision(action="PARTIAL_EXIT", reason="Scale tier 1")

    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=tier1_decision):
            # Advance the scale idx as ExitManager would have done
            trader.tsm.advance_scale_level()
            trader.manage_open_trade()

    assert trader.tsm.qty == pytest.approx(60.0)          # 40% of 100 exited
    assert trader.tsm.state == State.PARTIAL_EXIT_TAKEN
    assert trader.tsm._scale_level_idx == 1
    assert trader.tsm.side == "LONG"                       # no flip

    # ── Tier 2: ExitManager returns another PARTIAL_EXIT ─────────────────────
    tier2_decision = ExitDecision(action="PARTIAL_EXIT", reason="Scale tier 2")

    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=tier2_decision):
            trader.tsm.advance_scale_level()
            trader.manage_open_trade()

    # 30% of 60 = 18 → 42 remaining
    assert trader.tsm.qty == pytest.approx(42.0)
    assert trader.tsm._scale_level_idx == 2
    assert trader.tsm.side == "LONG"

    # ── Runner stop hit ───────────────────────────────────────────────────────
    stop_decision = ExitDecision(
        action="FULL_EXIT", reason="Hard stop hit", exit_price=stop, urgency="high"
    )
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=stop_decision):
            trader.manage_open_trade()

    assert trader.tsm.state == State.EXITED
    assert len(trader.tsm.session_trades) == 1
    # Ensure we never went short
    for rec in trader.tsm.session_trades:
        assert rec.side == "LONG"


# ── Scenario B: tier1 scale, then stop on runner at break-even ───────────────

def test_scenario_b_scale1_then_stop_at_be():
    """
    ENTRY → tier-1 PARTIAL_EXIT → stop hit at break-even.
    Net P&L = scale1 profit + 0 (runner exits at entry price).
    """
    trader = _make_trader("AAPL")
    entry, stop, orb_range = 180.0, 178.60, 1.50
    ep = orb_exit_plan("US_LARGE_CAP", entry=entry, stop=stop, orb_range=orb_range)

    _open_long(trader, entry=entry, stop=stop,
               target=entry + orb_range * 2.0, qty=50.0, exit_plan=ep)

    # Tier 1 partial
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(action="PARTIAL_EXIT", reason="tier1")):
            trader.tsm.advance_scale_level()
            trader.manage_open_trade()

    qty_after_tier1 = trader.tsm.qty
    assert qty_after_tier1 < 50.0     # some exited
    assert qty_after_tier1 > 0.0      # not fully flat

    # Break-even stop move (PositionManager moves stop to entry)
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="MOVE_STOP",
                                                  new_stop=entry,
                                                  reason="breakeven")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(action="HOLD", reason="hold")):
            trader.manage_open_trade()

    assert trader.tsm.current_stop == pytest.approx(entry)

    # Stop hit at break-even (paper fill at last_price() ≈ entry)
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(
                              action="FULL_EXIT",
                              reason="Hard stop hit (BE)",
                              exit_price=entry,
                              urgency="high",
                          )):
            trader.manage_open_trade()

    assert trader.tsm.state == State.EXITED
    # tsm.qty retains the qty at the moment of closure (not zeroed);
    # verify via session_trades instead.
    assert len(trader.tsm.session_trades) >= 1
    record = trader.tsm.session_trades[-1]
    assert record.side == "LONG"


# ── Scenario C: NSE VWAP two fixed tiers, no runner → auto full-close ─────────

def test_scenario_c_nse_vwap_two_tier_no_runner():
    """
    RELIANCE (NSE_LARGE_CAP) VWAP reversion.
    ExitPlan trail_type="none" → after last tier, _execute_partial_exit_from_exit_manager
    should trigger a full close automatically.
    """
    trader = _make_trader("RELIANCE")
    trader._df_5m = _make_5m_bars(30, base=2480.0, step=1.0)

    entry, stop, vwap_level, atr = 2480.0, 2474.40, 2485.0, 4.67
    ep = vwap_exit_plan(
        "NSE_LARGE_CAP",
        entry=entry,
        stop=stop,
        vwap=vwap_level,
        atr=atr,
        direction="BUY",
    )
    # Confirm no runner in NSE VWAP plan
    assert ep.trail_type == "none"
    assert len(ep.scale_levels) == 2

    _open_long(trader, entry=entry, stop=stop,
               target=vwap_level + 0.4 * atr, qty=50.0, exit_plan=ep)

    # Tier 1 (70% at VWAP)
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(action="PARTIAL_EXIT", reason="VWAP tier1")):
            trader.tsm.advance_scale_level()
            trader.manage_open_trade()

    qty_after_tier1 = trader.tsm.qty
    assert qty_after_tier1 == pytest.approx(15.0)   # 70% of 50 = 35 exited → 15 left

    # Tier 2 (30% of remaining + no runner → auto full-close)
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(action="PARTIAL_EXIT",
                                                    reason="VWAP tier2 + no runner")):
            trader.tsm.advance_scale_level()
            trader.manage_open_trade()

    # After last tier with no runner, position must be fully closed
    assert trader.tsm.state == State.EXITED
    # Verify no net-short: all fills were SELL (LONG side), never BUY_COVER into negative
    for rec in trader.tsm.session_trades:
        assert rec.side == "LONG"


# ── Guard: double partial in same bar is skipped ─────────────────────────────

def test_double_partial_same_bar_skipped():
    """
    If PositionManager AND ExitManager both return PARTIAL_EXIT in the same bar,
    ExitManager's tier is skipped and a warning is logged.
    """
    trader = _make_trader("SPY")
    entry, stop, orb_range = 450.0, 448.50, 2.0
    ep = orb_exit_plan("US_ETF", entry=entry, stop=stop, orb_range=orb_range)
    _open_long(trader, entry=entry, stop=stop, target=entry + 4.0,
               qty=100.0, exit_plan=ep)

    pm_partial = PositionUpdate(action="PARTIAL_EXIT", exit_qty=20.0,
                                reason="PM tier (BE)")
    ex_partial = ExitDecision(action="PARTIAL_EXIT", reason="ExitMgr tier 1")

    with patch.object(trader.position_manager, "evaluate", return_value=pm_partial):
        with patch.object(trader.exit_manager, "evaluate", return_value=ex_partial):
            with patch.object(trader, "_place_exit_order", return_value=450.0):
                # Do NOT advance scale_level — simulate PM partial happening first
                trader.manage_open_trade()

    # Only PositionManager's partial should have fired (20 shares)
    # ExitManager's was blocked; scale_level_idx still at 0
    assert trader.tsm._scale_level_idx == 0   # ExitManager tier not consumed
    # qty reduced by PM partial
    assert trader.tsm.qty == pytest.approx(80.0)


# ── State transitions are legal ───────────────────────────────────────────────

def test_state_transitions_are_legal():
    """
    After opening a position and executing two scale-outs, every TSM state
    encountered must be in the set of expected active-position states.
    """
    valid_active = {State.LONG, State.PARTIAL_EXIT_TAKEN, State.TRAILING, State.EXITED}

    trader = _make_trader("QQQ")
    entry, stop, orb_range = 380.0, 378.50, 1.50
    ep = orb_exit_plan("US_ETF", entry=entry, stop=stop, orb_range=orb_range)
    _open_long(trader, entry=entry, stop=stop, target=entry + 3.0,
               qty=60.0, exit_plan=ep)

    states_seen = [trader.tsm.state]

    for tier in range(len(ep.scale_levels)):
        with patch.object(trader.position_manager, "evaluate",
                          return_value=PositionUpdate(action="HOLD", reason="hold")):
            with patch.object(trader.exit_manager, "evaluate",
                              return_value=ExitDecision(action="PARTIAL_EXIT",
                                                        reason=f"tier{tier}")):
                trader.tsm.advance_scale_level()
                trader.manage_open_trade()
        states_seen.append(trader.tsm.state)

    # Close the runner
    with patch.object(trader.position_manager, "evaluate",
                      return_value=PositionUpdate(action="HOLD", reason="hold")):
        with patch.object(trader.exit_manager, "evaluate",
                          return_value=ExitDecision(action="FULL_EXIT",
                                                    reason="trail stop",
                                                    exit_price=382.0)):
            trader.manage_open_trade()

    states_seen.append(trader.tsm.state)
    for s in states_seen:
        assert s in valid_active, f"Unexpected state: {s}"
    assert states_seen[-1] == State.EXITED
