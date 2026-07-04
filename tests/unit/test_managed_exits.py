"""Unit tests for the stateful managed-exits trade-management module."""
import logging

import numpy as np
import pandas as pd
import pytest

from app.services.strategy.managed_exits import (
    EXIT_PRESETS,
    ManagedExitConfig,
    manage_trade,
)


def _frame(opens, highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(opens), freq="D")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes},
                        index=idx)


def _flat_then(path: list[tuple[float, float, float, float]], warmup: int = 20) -> pd.DataFrame:
    """20 warm-up bars at 100 (for ATR ≈ 1) followed by explicit OHLC bars."""
    o = [100.0] * warmup + [p[0] for p in path]
    h = [100.5] * warmup + [p[1] for p in path]
    lo = [99.5] * warmup + [p[2] for p in path]
    c = [100.0] * warmup + [p[3] for p in path]
    return _frame(o, h, lo, c)


ENTRY = 20  # first bar after warm-up; ATR(14) ≈ 1.0 at entry


class TestStopsAndTargets:
    def test_initial_stop_hit(self):
        # Entry 100, 1.5-ATR stop ≈ 98.5; bar 2 trades down through it
        df = _flat_then([(100, 100.5, 99.5, 100), (100, 100.2, 98.0, 98.2)])
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                partial_at_r=None, trail="none")
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason.startswith("stop:initial_atr")
        assert t.exit_price == pytest.approx(t.initial_stop)
        assert t.realized_r == pytest.approx(-1.0, abs=0.01)

    def test_gap_below_stop_fills_at_open(self):
        df = _flat_then([(100, 100.5, 99.5, 100), (95.0, 95.5, 94.5, 95.0)])
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                partial_at_r=None, trail="none")
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason.startswith("stop_gap")
        assert t.exit_price == pytest.approx(95.0)
        assert t.realized_r < -1.0, "gap through the stop must lose more than 1R"

    def test_time_stop(self):
        bars = [(100, 100.5, 99.5, 100)] * 10
        df = _flat_then(bars)
        cfg = ManagedExitConfig(initial_atr_mult=3.0, breakeven_at_r=None,
                                partial_at_r=None, trail="none", time_stop_bars=5)
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason == "time_stop"
        assert t.hold_bars == 5

    def test_profit_target(self):
        df = _flat_then([(100, 100.5, 99.5, 100), (101, 106.0, 100.5, 105.0)])
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                partial_at_r=None, trail="none", profit_target_r=2.0)
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason == "profit_target"
        assert t.realized_r == pytest.approx(2.0, abs=0.05)


class TestPartialAndBreakeven:
    def test_partial_then_breakeven_then_trail_exit(self, caplog):
        # Runner: rallies through 1.5R partial and 1R breakeven, then collapses
        bars = [
            (100, 100.5, 99.8, 100.3),
            (100.5, 102.5, 100.2, 102.2),   # +1.5R hit (r_unit≈1.5) → partial + breakeven
            (102.5, 104.5, 102.0, 104.2),   # trail ratchets
            (104.0, 104.5, 103.5, 104.0),
            (99.0, 99.5, 98.5, 99.0),        # gap down through raised stop
        ]
        df = _flat_then(bars)
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=1.0,
                                partial_at_r=1.5, partial_fraction=0.5,
                                trail="atr", trail_atr_mult=2.5)
        with caplog.at_level(logging.INFO, logger="app.services.strategy.managed_exits"):
            t = manage_trade("T", df, ENTRY, cfg, log_context="TEST")
        assert t.partial_price is not None
        assert t.partial_r == pytest.approx(1.5, abs=0.05)
        assert any("PARTIAL 50%" in e for e in t.events)
        assert any("BREAKEVEN" in e for e in t.events)
        assert t.stop_moves >= 1
        # Blended result must be positive: half banked at +1.5R
        assert t.realized_r > 0
        # Logged with context
        assert any("[TEST]" in m for m in caplog.messages)

    def test_breakeven_protects_from_next_bar(self):
        # Hit 1R on bar 1, revisit entry price on bar 2 → breakeven stop exits at ~0R
        bars = [
            (100, 100.5, 99.8, 100.3),
            (100.5, 101.8, 100.2, 101.5),    # +1R (r_unit≈1.5 → 1R=101.5) hit intrabar
            (100.5, 100.8, 99.0, 99.2),      # falls back through entry
        ]
        df = _flat_then(bars)
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=1.0,
                                partial_at_r=None, trail="none")
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason == "stop:breakeven"
        assert t.realized_r == pytest.approx(0.0, abs=0.05)

    def test_stop_beats_target_same_bar(self):
        # Bar spans both the stop and the partial level → conservative: stop fills
        df = _flat_then([(100, 100.5, 99.5, 100), (100, 105.0, 98.0, 99.0)])
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                partial_at_r=1.5, trail="none")
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.exit_reason.startswith("stop:")
        assert t.partial_price is None


class TestTrails:
    def _runner(self, n: int = 30) -> pd.DataFrame:
        # Steady uptrend then a sharp break
        bars = []
        px = 100.0
        for _ in range(n):
            bars.append((px, px + 1.2, px - 0.4, px + 1.0))
            px += 1.0
        bars.append((px - 6, px - 5.5, px - 8, px - 7))   # collapse
        return _flat_then(bars)

    @pytest.mark.parametrize("trail,expected", [
        ("atr", "atr_trail"),
        ("structure", "structure_trail"),
        ("chandelier", "chandelier_trail"),
    ])
    def test_trail_variants_ratchet_and_exit(self, trail, expected):
        df = self._runner()
        cfg = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                partial_at_r=None, trail=trail, time_stop_bars=200)
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.stop_moves > 3, "trail must ratchet repeatedly in a trend"
        assert expected in t.exit_reason or t.exit_reason == "end_of_data"
        assert t.realized_r > 0, "trail must lock in trend gains"

    def test_tightest_uses_most_conservative(self):
        df = self._runner()
        base = ManagedExitConfig(initial_atr_mult=1.5, breakeven_at_r=None,
                                 partial_at_r=None, time_stop_bars=200)
        from dataclasses import replace
        t_atr = manage_trade("T", df, ENTRY, replace(base, trail="atr"))
        t_str = manage_trade("T", df, ENTRY, replace(base, trail="structure"))
        t_tight = manage_trade("T", df, ENTRY, replace(base, trail="tightest"))
        assert t_tight.final_stop >= min(t_atr.final_stop, t_str.final_stop) - 1e-6
        assert t_tight.final_stop == pytest.approx(
            max(t_atr.final_stop, t_str.final_stop), abs=0.5)

    def test_structure_initial_stop_uses_swing_low(self):
        # Swing low at 99.5 in warm-up; buffered stop should sit just below it,
        # above the 1.5-ATR stop → structure wins as the more conservative
        df = _flat_then([(100, 100.5, 99.5, 100)] * 5)
        cfg = ManagedExitConfig(initial_atr_mult=3.0, structure_initial=True,
                                swing_lookback=10, structure_buffer_atr=0.25,
                                breakeven_at_r=None, partial_at_r=None, trail="none")
        t = manage_trade("T", df, ENTRY, cfg)
        assert t.initial_stop > 100 - 3.0 * 1.5  # tighter than the wide ATR stop
        assert t.initial_stop < 99.5             # below the swing low


class TestSellSignalAndPresets:
    def test_mirrored_sell_exit(self):
        bars = [(100, 100.5, 99.5, 100)] * 8
        df = _flat_then(bars)
        sig = pd.Series("HOLD", index=df.index)
        sig.iloc[ENTRY + 3] = "SELL"
        t = manage_trade("T", df, ENTRY, EXIT_PRESETS["A_mirrored_sell"], sell_signal=sig)
        assert t.exit_reason == "sell_signal"
        assert t.hold_bars == 4  # fills at the next bar's open

    def test_presets_all_run(self):
        np.random.seed(4)
        px = 100 + np.cumsum(np.random.normal(0.3, 1.0, 120))
        df = _frame(px, px + 0.8, px - 0.8, px + 0.2)
        for name, cfg in EXIT_PRESETS.items():
            t = manage_trade("T", df, 30, cfg)
            assert t is not None, name
            assert t.exit_reason, name
            assert t.hold_bars >= 0, name

    def test_mfe_capture_recorded(self):
        df = self_runner = TestTrails()._runner()
        t = manage_trade("T", df, ENTRY, EXIT_PRESETS["B_atr_trail"])
        assert t.mfe_pct > 0
        assert t.mfe_captured_pct is not None
        assert 0 < t.mfe_captured_pct <= 110
