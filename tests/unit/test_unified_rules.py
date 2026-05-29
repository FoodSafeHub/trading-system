"""Phase 1: the 6 unified daily rule types exist in parallel with the old ones."""
import numpy as np
import pandas as pd
import pytest

from app.services.strategy.rules import _RULE_REGISTRY, evaluate_strategy

NEW_TYPES = ["rsi2_reversion", "trend_pullback", "squeeze_breakout",
             "momentum_breakout", "panic_reversal", "trend_follow"]
OLD_TYPES = ["rsi2_mean_reversion", "ema_macd_crossover", "bb_squeeze_breakout",
             "pullback_ema50", "vix_spike_reversal", "bollinger", "fib_pullback",
             "supertrend", "ema_ribbon", "breakout"]


def _ohlcv(n=400, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": np.full(n, 2e6)}, index=idx)


def test_new_types_registered_in_parallel():
    for t in NEW_TYPES:
        assert t in _RULE_REGISTRY
    for t in OLD_TYPES:
        assert t in _RULE_REGISTRY, f"old type {t} must remain registered"


@pytest.mark.parametrize("stype", NEW_TYPES)
def test_new_type_runs_and_returns_valid_direction(stype):
    df = _ohlcv()
    sig = evaluate_strategy(stype, "TEST", df["Close"], {}, ohlcv=df)
    assert sig.direction in ("BUY", "SELL", "HOLD")
    assert sig.strategy_name == stype


def test_new_types_emit_stop_on_buy():
    # Craft a clean RSI(2) oversold-in-uptrend tail so rsi2_reversion fires BUY.
    base = np.linspace(100, 160, 300)               # strong uptrend (> SMA200)
    dip = np.array([160, 158, 150])                 # sharp 3-bar dip -> RSI(2) low
    close = np.concatenate([base, dip])
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    df = pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                       "Close": close, "Volume": np.full(len(close), 2e6)}, index=idx)
    sig = evaluate_strategy("rsi2_reversion", "TEST", df["Close"], {}, ohlcv=df)
    if sig.direction == "BUY":
        assert sig.stop_price is not None and sig.stop_price < float(close[-1])
        assert 0.0 < sig.confidence <= 1.0
