"""Unit tests for the backtest CostModel (Phase 0)."""
from app.services.backtest.costs import CostModel, ZERO, US_DEFAULT, INDIA_DEFAULT


def test_zero_is_identity():
    assert ZERO.apply_buy(123.45) == 123.45
    assert ZERO.apply_sell(123.45) == 123.45
    assert ZERO.entry_commission(100, 12345.0) == 0.0
    assert ZERO.exit_commission(100, 12345.0) == 0.0


def test_default_constructor_is_zero():
    assert CostModel() == ZERO


def test_slippage_directions():
    cm = CostModel(slippage_bps=10.0)
    assert cm.apply_buy(100.0) > 100.0   # buys fill higher
    assert cm.apply_sell(100.0) < 100.0  # sells fill lower


def test_commission_and_min_floor():
    cm = CostModel(commission_per_share=0.01, min_commission=1.0)
    # 10 shares * 0.01 = 0.10 -> floored to min 1.0
    assert cm.entry_commission(10, 1000.0) == 1.0
    # 1000 shares * 0.01 = 10.0 -> above floor
    assert cm.entry_commission(1000, 100000.0) == 10.0


def test_taxes_only_on_exit():
    cm = CostModel(commission_bps=3.0, taxes_bps=10.0)
    notional = 10_000.0
    entry = cm.entry_commission(10, notional)
    exit_ = cm.exit_commission(10, notional)
    assert exit_ > entry  # exit adds taxes


def test_india_drag_exceeds_us():
    us = (US_DEFAULT.apply_buy(100) - 100) + (100 - US_DEFAULT.apply_sell(100))
    ind = (INDIA_DEFAULT.apply_buy(100) - 100) + (100 - INDIA_DEFAULT.apply_sell(100))
    assert ind > us


def test_liquidity_cap():
    off = CostModel()  # cap disabled
    assert off.is_liquid(1e12, 1e6) is True
    capped = CostModel(adv_cap_pct=1.0)  # 1% of ADV$
    assert capped.is_liquid(5_000.0, 1_000_000.0) is True   # 0.5% of ADV
    assert capped.is_liquid(50_000.0, 1_000_000.0) is False  # 5% of ADV
