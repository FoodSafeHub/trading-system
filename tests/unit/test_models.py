"""Unit tests for StrategySignal Phase 0 field additions."""
from app.services.strategy.models import StrategySignal


def test_new_fields_default_inert():
    s = StrategySignal(symbol="X", direction="HOLD")
    assert s.stop_price is None
    assert s.target_price is None
    assert s.confidence == 1.0


def test_existing_fields_unchanged():
    s = StrategySignal(symbol="X", direction="BUY", strength=0.7,
                       price_at_signal=10.0, indicators={"a": 1}, strategy_name="r")
    assert s.strength == 0.7
    assert s.price_at_signal == 10.0
    assert s.indicators == {"a": 1}
    assert s.strategy_name == "r"


def test_new_fields_settable():
    s = StrategySignal(symbol="X", direction="BUY", stop_price=9.0,
                       target_price=12.0, confidence=0.6)
    assert (s.stop_price, s.target_price, s.confidence) == (9.0, 12.0, 0.6)
