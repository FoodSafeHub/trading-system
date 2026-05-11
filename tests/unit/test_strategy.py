"""Unit tests for strategy rules."""
import numpy as np
import pandas as pd
import pytest

from app.services.strategy.rules import evaluate_strategy


def _rising(n: int = 80) -> pd.Series:
    return pd.Series(np.linspace(100, 200, n))


def _falling(n: int = 80) -> pd.Series:
    return pd.Series(np.linspace(200, 100, n))


def test_sma_rsi_returns_valid_direction():
    prices = _rising()
    signal = evaluate_strategy("sma_rsi", "SPY", prices, {"sma_fast": 5, "sma_slow": 20})
    assert signal.direction in ("BUY", "SELL", "HOLD")
    assert signal.symbol == "SPY"


def test_ema_crossover_returns_signal():
    prices = _rising()
    signal = evaluate_strategy("ema_crossover", "AAPL", prices, {"ema_fast": 5, "ema_slow": 20})
    assert signal.direction in ("BUY", "SELL", "HOLD")


def test_macd_strategy():
    prices = _rising(100)
    signal = evaluate_strategy("macd", "MSFT", prices, {})
    assert signal.direction in ("BUY", "SELL", "HOLD")
    assert "macd" in signal.indicators


def test_bollinger_strategy():
    prices = _falling(50)
    signal = evaluate_strategy("bollinger", "QQQ", prices, {})
    assert signal.direction in ("BUY", "SELL", "HOLD")
    assert "bb_upper" in signal.indicators


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy type"):
        evaluate_strategy("nonexistent", "SPY", _rising(), {})
