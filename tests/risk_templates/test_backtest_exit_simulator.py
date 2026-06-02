"""
Tests for backtest_exit_simulator — the bar-by-bar engine that consumes
ExitPlan in the day-trading backtester.

What we are pinning:
    * ExitPlan with scale-out tiers is honored (multi-leg result, not one bracket)
    * Break-even stop move actually fires
    * Trail activates and ratchets correctly
    * Hard time exit (session EOD) overrides everything
    * Legacy path (exit_plan=None) still produces the simple single bracket
    * Short side stops/targets/trail behave symmetrically
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.strategy.daytrading.backtest_exit_simulator import (
    SimulatedExit, aggregate_pnl, simulate_exit,
)
from app.services.strategy.daytrading.risk_templates import ExitPlan, ScaleLevel


def _bars(rows: list[tuple[float, float, float, float]],
          start: str = "2026-01-02 09:35") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(
        rows, columns=["Open", "High", "Low", "Close"], index=idx
    ).assign(Volume=1e5)


# ── Scale-out tier honoring ──────────────────────────────────────────────────


class TestScaleOuts:
    def test_two_scale_tiers_both_fire_and_leave_runner(self):
        """Both tiers should fire; remaining qty walks until EOD."""
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=1.0,
            scale_levels=[
                ScaleLevel(trigger_r=1.0, pct_to_close=0.40, trigger_price=101.0),
                ScaleLevel(trigger_r=2.0, pct_to_close=0.30, trigger_price=102.0),
            ],
            trail_type="none", max_hold_bars=20,
            hard_exit_time_et="23:59",   # disable hard-time
        )
        bars = _bars([
            (100.5, 101.2, 100.4, 101.0),   # tier 1 hits
            (101.0, 102.1, 100.9, 102.0),   # tier 2 hits
            (102.0, 102.5, 101.8, 102.3),   # runner drifts
            (102.3, 102.4, 102.0, 102.1),
        ])
        sim = simulate_exit(
            "BUY", entry_price=100.0, initial_stop=99.0, initial_target=103.0,
            qty=10.0, signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        reasons = [leg.reason for leg in sim.legs]
        assert "scale_1" in reasons
        assert "scale_2" in reasons
        # final leg closes the runner via EOD
        assert sim.legs[-1].reason == "eod"
        assert sim.closed_qty == 10.0
        assert sim.breakeven_hit is True   # crossed 1R en route

    def test_naive_one_shot_not_applied_when_exit_plan_exists(self):
        """If scale tiers are defined, hitting the legacy `initial_target`
        alone must NOT cause a single-bracket TARGET exit."""
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=99.0,   # never trip BE
            scale_levels=[
                ScaleLevel(trigger_r=1.0, pct_to_close=0.50, trigger_price=101.0),
            ],
            trail_type="none", max_hold_bars=20,
            hard_exit_time_et="23:59",
        )
        # Single bar that punches all the way to the legacy target.
        bars = _bars([
            (100.0, 103.5, 99.5, 102.5),
            (102.5, 102.6, 102.0, 102.2),
            (102.2, 102.3, 102.0, 102.1),
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 103.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        # We must see scale_1, then an EOD close on the runner — NOT a single
        # "target" leg for the full qty.
        assert sim.legs[0].reason == "scale_1"
        assert sim.legs[0].qty == 5
        assert any(l.reason == "eod" for l in sim.legs)


# ── Break-even stop move ─────────────────────────────────────────────────────


class TestBreakeven:
    def test_breakeven_moves_stop_to_entry_then_stops_out_at_entry(self):
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=1.0,
            scale_levels=[],                 # no scale-outs — single position
            trail_type="none", max_hold_bars=20,
            hard_exit_time_et="23:59",
        )
        # Push to +1R, then come back through entry — stop should now be 100.
        bars = _bars([
            (100.0, 101.2, 99.8, 101.0),     # crosses +1R
            (101.0, 101.1, 99.5, 99.7),      # dips through entry — hits BE stop at 100
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 110.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        assert sim.breakeven_hit is True
        assert sim.legs[-1].reason == "breakeven_stop"
        assert sim.legs[-1].price == pytest.approx(100.0)
        assert sim.primary_outcome == "STOPPED"


# ── Trailing stop ────────────────────────────────────────────────────────────


class TestTrail:
    def test_prior_bar_low_trail_ratchets_and_stops_runner(self):
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=1.0,
            scale_levels=[
                ScaleLevel(trigger_r=1.0, pct_to_close=0.50, trigger_price=101.0),
            ],
            trail_type="prior_bar_low_5m", trail_trigger_r=1.0,
            max_hold_bars=20, hard_exit_time_et="23:59",
        )
        bars = _bars([
            (100.5, 101.2, 100.4, 101.0),    # scale_1 fires
            (101.0, 101.6, 100.9, 101.5),    # trail active; prior_low = 100.4
            (101.5, 102.0, 101.4, 101.9),    # ratchet up to 100.9
            (101.9, 102.3, 101.8, 102.1),    # ratchet up to 101.4
            (102.1, 102.2, 100.5, 100.8),    # big down bar -> below ratcheted 101.4 -> trail hit
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 110.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        assert sim.trail_activated is True
        # Final leg is the trail stop, not target/EOD
        assert sim.legs[-1].reason == "trail_stop"
        # And the trail price ratcheted above entry — much better than initial 99.0
        assert sim.final_stop > 100.0


# ── Hard time exit ───────────────────────────────────────────────────────────


class TestHardTimeExit:
    def test_eod_force_flat_closes_remaining(self):
        # US strategy with ET hard exit at 09:50; bars cross that clock.
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=99.0,
            scale_levels=[],
            trail_type="none", max_hold_bars=99,
            hard_exit_time_et="09:50",   # bar 4 onwards triggers
            hard_exit_time_ist="14:45",
        )
        bars = _bars([
            (100.0, 100.6, 99.9, 100.5),     # 09:35
            (100.5, 101.0, 100.4, 100.9),    # 09:40
            (100.9, 101.1, 100.7, 101.0),    # 09:45
            (101.0, 101.2, 100.8, 101.1),    # 09:50 ← hard-exit fires
            (101.1, 101.5, 100.9, 101.3),    # never reached
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 110.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        assert sim.legs[-1].reason == "hard_exit_09:50"
        assert sim.primary_outcome == "EOD_EXIT"
        assert sim.closed_qty == 10.0


# ── Legacy back-compat (no ExitPlan) ─────────────────────────────────────────


class TestLegacyNoExitPlan:
    def test_legacy_single_target(self):
        bars = _bars([
            (100.0, 100.5, 99.8, 100.4),
            (100.4, 103.5, 100.2, 103.2),   # touches target 103
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 103.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=None, symbol="AAPL",
            max_hold_bars_default=60,
        )
        assert len(sim.legs) == 1
        assert sim.legs[0].reason == "target"
        assert sim.legs[0].price == pytest.approx(103.0)
        assert sim.primary_outcome == "TARGET"

    def test_legacy_stop(self):
        bars = _bars([
            (100.0, 100.2, 98.5, 98.8),    # punches through 99 stop
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 103.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=None, symbol="AAPL",
        )
        assert sim.legs[0].reason == "stop"
        assert sim.legs[0].price == pytest.approx(99.0)
        assert sim.primary_outcome == "STOPPED"

    def test_legacy_time_exit(self):
        # No target/stop hit — should close at max_hold_bars.
        bars = _bars([(100.0, 100.3, 99.7, 100.1)] * 5)
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 110.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=None, symbol="AAPL",
            max_hold_bars_default=3,
        )
        assert sim.legs[-1].reason == "max_hold_bars"
        assert sim.primary_outcome == "TIME_EXIT"


# ── Short side symmetry ──────────────────────────────────────────────────────


class TestShortSide:
    def test_short_scale_then_stop(self):
        ep = ExitPlan(
            initial_stop_price=101.0, breakeven_r=99.0,
            scale_levels=[
                ScaleLevel(trigger_r=1.0, pct_to_close=0.50, trigger_price=99.0),
            ],
            trail_type="none", max_hold_bars=20,
            hard_exit_time_et="23:59",
        )
        # Short from 100, stop 101, scale at 99 (down moves are profit).
        bars = _bars([
            (100.0, 100.2, 98.9, 99.0),     # scale hit at 99
            (99.0, 101.5, 98.9, 101.2),     # stop hit at 101 on the runner
        ])
        sim = simulate_exit(
            "SELL", 100.0, 101.0, 95.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        reasons = [l.reason for l in sim.legs]
        assert "scale_1" in reasons
        assert reasons[-1] == "stop"
        assert sim.primary_outcome == "STOPPED"


# ── aggregate_pnl ────────────────────────────────────────────────────────────


class TestAggregatePnl:
    def test_long_multi_leg_pnl(self):
        ep = ExitPlan(
            initial_stop_price=99.0, breakeven_r=99.0,
            scale_levels=[
                ScaleLevel(trigger_r=1.0, pct_to_close=0.50, trigger_price=101.0),
                ScaleLevel(trigger_r=2.0, pct_to_close=1.00, trigger_price=102.0),
            ],
            trail_type="none", max_hold_bars=20, hard_exit_time_et="23:59",
        )
        bars = _bars([
            (100.0, 101.2, 99.8, 101.0),    # scale_1: 5 shares @ 101
            (101.0, 102.2, 100.9, 102.1),   # scale_2: 5 shares @ 102 (100% of remaining)
        ])
        sim = simulate_exit(
            "BUY", 100.0, 99.0, 110.0, qty=10.0,
            signal_time=bars.index[0], future_bars=bars,
            exit_plan=ep, symbol="AAPL",
        )
        agg = aggregate_pnl("BUY", 100.0, sim)
        # 5×(101-100) + 5×(102-100) = 5 + 10 = 15
        assert agg["gross_pnl"] == pytest.approx(15.0)
        # vwap = (5*101 + 5*102) / 10 = 101.5
        assert agg["exit_price"] == pytest.approx(101.5)
        assert agg["qty"] == pytest.approx(10.0)
