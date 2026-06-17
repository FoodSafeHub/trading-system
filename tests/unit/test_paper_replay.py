"""
Tests for the paper-replay simulator and the bar-time overrides it relies on.

Problem fixed: a paper autotrader (broker=None) used to sit completely idle when
the live market was closed — it never fetched data or evaluated, so "nothing got
updated", unlike the simulation backtest. paper_replay makes it step through
recent historical bars using BAR time (not wall-clock) for every time gate, so it
produces real decisions/trades. These tests use synthetic frames (no network).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.services.strategy.daytrading.market_open import ET, is_past_last_entry
from app.services.strategy.daytrading.autotrader.entry_decider import EntryDecider
from app.services.strategy.daytrading.autotrader.exit_manager import ExitManager
from app.services.strategy.daytrading.autotrader.single_stock_trader import (
    SingleStockTrader,
)
from app.services.strategy.daytrading.autotrader.trade_state import State


# ── is_past_last_entry override ─────────────────────────────────────────────────

def test_is_past_last_entry_uses_override():
    # 10:00 ET bar -> before the 15:15 cutoff, even though wall-clock may be after.
    morning = datetime(2026, 6, 12, 10, 0, tzinfo=ET)
    assert is_past_last_entry("AAPL", now=morning) is False
    # 15:30 ET bar -> after the cutoff.
    late = datetime(2026, 6, 12, 15, 30, tzinfo=ET)
    assert is_past_last_entry("AAPL", now=late) is True


def test_entry_decider_respects_now_override():
    # Build a minimal 5m frame; with a late override the decider must reject on time.
    idx = pd.date_range("2026-06-12 09:30", periods=20, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6},
        index=idx,
    )
    dec = EntryDecider(direction_mode="both")
    late = datetime(2026, 6, 12, 15, 45, tzinfo=ET)
    out = dec.decide("AAPL", df, df, df, market_state=None, now_override=late)
    assert out.action == "NO_TRADE"
    assert "Too late in day" in out.entry_reason


# ── ExitManager now-override + the now_time bug ─────────────────────────────────

def _open_long(tsm_price=100.0):
    """Return a state machine with an open long position for exit testing."""
    from app.services.strategy.daytrading.autotrader.trade_state import TradeStateMachine
    tsm = TradeStateMachine()
    tsm.open_position(
        symbol="AAPL", side="LONG", entry_price=tsm_price, qty=10,
        stop=tsm_price - 2, target=tsm_price + 4, strategy="EMAMomentum",
        entry_reason="test",
    )
    return tsm


def test_exit_manager_eod_uses_override_not_wallclock():
    # A 10:00 ET bar must NOT trigger the EOD force-flatten even though wall-clock
    # is likely after hours when the test runs.
    idx = pd.date_range("2026-06-12 09:30", periods=20, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0, "Volume": 1e6},
        index=idx,
    )
    tsm = _open_long()
    em = ExitManager(symbol="AAPL")
    morning = datetime(2026, 6, 12, 10, 0, tzinfo=ET)
    dec = em.evaluate(tsm, df, df, "TREND_UP", now=morning)
    assert "EOD force flatten" not in dec.reason


def test_exit_manager_eod_fires_after_cutoff_via_override():
    idx = pd.date_range("2026-06-12 09:30", periods=20, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0, "Volume": 1e6},
        index=idx,
    )
    tsm = _open_long()
    em = ExitManager(symbol="AAPL")
    after_eod = datetime(2026, 6, 12, 15, 50, tzinfo=ET)  # past 15:45 US flatten
    dec = em.evaluate(tsm, df, df, "TREND_UP", now=after_eod)
    assert dec.action == "FULL_EXIT"
    assert "EOD force flatten" in dec.reason


def test_exit_manager_eod_warn_does_not_crash():
    # Regression: the EOD-approach (15:30) block referenced an undefined `now_time`
    # and crashed on any profitable position near EOD. Drive it past 15:30 with a
    # profitable long and assert no NameError.
    idx = pd.date_range("2026-06-12 09:30", periods=20, freq="5min", tz=ET)
    # Rising closes so the long is profitable (r_multiple > 0).
    closes = [100 + i * 0.2 for i in range(20)]
    df = pd.DataFrame(
        {"Open": closes, "High": [c + 0.3 for c in closes],
         "Low": [c - 0.3 for c in closes], "Close": closes, "Volume": 1e6},
        index=idx,
    )
    tsm = _open_long(tsm_price=100.0)
    tsm.current_stop = 99.0
    em = ExitManager(symbol="AAPL")
    warn_time = datetime(2026, 6, 12, 15, 35, tzinfo=ET)  # past 15:30 warn, before 15:45 flat
    # Must not raise NameError('now_time').
    dec = em.evaluate(tsm, df, df, "TREND_UP", now=warn_time)
    assert dec is not None


# ── Replay mechanics (no network: inject synthetic history) ─────────────────────

def _trader_with_synthetic_history(n_bars=120):
    """Build a paper trader and inject a synthetic intraday 5m frame for replay."""
    t = SingleStockTrader(
        "AAPL", broker=None, entry_mode="native_strategy",
        direction_mode="both", paper_replay=True,
    )
    # Two synthetic trading days of 5m bars, 09:30–15:55 ET.
    idx = pd.date_range("2026-06-11 09:30", periods=n_bars, freq="5min", tz=ET)
    # Mild uptrend with noise so indicators are well-defined.
    closes = [100 + (i % 30) * 0.1 for i in range(n_bars)]
    df = pd.DataFrame(
        {"Open": closes, "High": [c + 0.2 for c in closes],
         "Low": [c - 0.2 for c in closes], "Close": closes, "Volume": 1e6},
        index=idx,
    )
    t._replay_5m = df
    t._replay_15m = df
    t._replay_1m = df
    t._replay_loaded = True
    t._replay_end_idx = len(df) - 1
    t._replay_start_idx = 15
    t._replay_cursor = t._replay_start_idx - 1
    return t, df


def test_replay_disabled_when_broker_present():
    broker = type("B", (), {"is_paper": False})()
    t = SingleStockTrader("AAPL", broker=broker, paper_replay=True)
    assert t.paper_replay is False  # live broker => no replay


def test_replay_advances_sim_clock_and_populates_decisions():
    t, df = _trader_with_synthetic_history(n_bars=80)
    assert t._sim_now is None
    t._replay_step()
    # After one step the sim clock is set to a bar timestamp and a decision logged.
    assert t._sim_now is not None
    assert t.get_status()["replay_active"] is True
    assert len(t._decision_log) >= 1


def test_replay_completes_and_idles_with_persisted_state():
    t, df = _trader_with_synthetic_history(n_bars=60)
    for _ in range(200):
        t._replay_step()
        if t.get_status().get("replay_done"):
            break
    st = t.get_status()
    assert st["replay_done"] is True
    # Decision log persists (it grew across the window) and didn't reset to empty.
    assert len(t._decision_log) > 5
    # A REPLAY_COMPLETE marker is logged exactly once.
    completes = [e for e in t._decision_log if e["event"] == "REPLAY_COMPLETE"]
    assert len(completes) == 1


def test_replay_does_not_advance_past_end():
    t, df = _trader_with_synthetic_history(n_bars=40)
    for _ in range(500):
        t._replay_step()
    # Cursor never exceeds the window end.
    assert t._replay_cursor <= t._replay_end_idx
