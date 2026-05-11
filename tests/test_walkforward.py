"""
Unit tests for walk-forward metric helpers and edge cases.
Run with:  .venv\Scripts\python -m pytest tests/test_walkforward.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.services.backtest.walkforward_engine import calc_total_return, calc_cagr, _wfe, _wfe_label

_TRADING_DAYS = 252


def _curve(start: float, end: float, n_bars: int) -> list:
    """Build a synthetic equity curve that moves linearly from start to end."""
    step = (end - start) / (n_bars - 1) if n_bars > 1 else 0
    return [{"equity": round(start + i * step, 4)} for i in range(n_bars)]


# ── calc_total_return ─────────────────────────────────────────────────────────

class TestCalcTotalReturn:
    def test_positive_return(self):
        curve = _curve(10_000, 13_000, 756)   # 30 % gain
        assert abs(calc_total_return(curve) - 30.0) < 0.01

    def test_zero_return_flat(self):
        curve = _curve(10_000, 10_000, _TRADING_DAYS)
        assert calc_total_return(curve) == 0.0

    def test_negative_return(self):
        curve = _curve(10_000, 8_000, _TRADING_DAYS)
        assert abs(calc_total_return(curve) - (-20.0)) < 0.01

    def test_empty_curve(self):
        assert calc_total_return([]) == 0.0

    def test_single_point(self):
        assert calc_total_return([{"equity": 10_000}]) == 0.0

    def test_zero_initial_equity(self):
        curve = [{"equity": 0}, {"equity": 1_000}]
        assert calc_total_return(curve) == 0.0


# ── calc_cagr ─────────────────────────────────────────────────────────────────

class TestCalcCagr:
    def test_known_example_3y(self):
        # 10k -> 13k over exactly 3 years (756 bars)
        curve = _curve(10_000, 13_000, 3 * _TRADING_DAYS)
        cagr = calc_cagr(curve)
        expected = ((13_000 / 10_000) ** (1 / 3) - 1) * 100   # ≈ 9.139%
        assert abs(cagr - expected) < 0.05

    def test_flat_equity(self):
        curve = _curve(10_000, 10_000, _TRADING_DAYS)
        assert calc_cagr(curve) == 0.0

    def test_empty_curve(self):
        assert calc_cagr([]) == 0.0

    def test_single_point(self):
        assert calc_cagr([{"equity": 10_000}]) == 0.0

    def test_fractional_year(self):
        # 6 months (126 bars): 10k -> 11k  -> CAGR ~20.9%
        curve = _curve(10_000, 11_000, 126)
        cagr = calc_cagr(curve)
        expected = ((11_000 / 10_000) ** (252 / 126) - 1) * 100
        assert abs(cagr - expected) < 0.05

    def test_strong_is_weak_oos_ratio(self):
        # IS CAGR ≈ 20%, OOS CAGR ≈ 10%  -> WFE ≈ 0.5
        is_curve  = _curve(10_000, 10_000 * (1.20 ** 3), 3 * _TRADING_DAYS)
        oos_curve = _curve(10_000, 10_000 * (1.10 ** 1), 1 * _TRADING_DAYS)
        is_cagr  = calc_cagr(is_curve)
        oos_cagr = calc_cagr(oos_curve)
        wfe = _wfe(oos_cagr, is_cagr)
        assert abs(is_cagr - 20.0) < 0.5
        assert abs(oos_cagr - 10.0) < 0.5
        assert abs(wfe - 0.5) < 0.05


# ── WFE helper ────────────────────────────────────────────────────────────────

class TestWFE:
    def test_identical_returns_ratio_one(self):
        is_cagr = oos_cagr = 12.0
        assert _wfe(oos_cagr, is_cagr) == pytest.approx(1.0)

    def test_zero_is_cagr_returns_none(self):
        assert _wfe(5.0, 0.0) is None

    def test_negative_is_cagr_returns_none(self):
        assert _wfe(5.0, -3.0) is None

    def test_no_oos_edge(self):
        assert _wfe(0.0, 10.0) == pytest.approx(0.0)

    def test_label_excellent(self):
        assert "Excellent" in _wfe_label(1.05)

    def test_label_acceptable(self):
        assert "Acceptable" in _wfe_label(0.85)

    def test_label_yellow(self):
        assert "Yellow" in _wfe_label(0.60)

    def test_label_degraded(self):
        assert "Degraded" in _wfe_label(0.40)

    def test_label_red_flag(self):
        assert "Red flag" in _wfe_label(0.20)

    def test_label_none(self):
        assert "N/A" in _wfe_label(None)


# ── Zero trades edge case ──────────────────────────────────────────────────────

class TestZeroTrades:
    def test_flat_curve_returns_zero(self):
        flat = _curve(10_000, 10_000, _TRADING_DAYS)
        assert calc_total_return(flat) == 0.0
        assert calc_cagr(flat) == 0.0
        assert _wfe(0.0, 0.0) is None
