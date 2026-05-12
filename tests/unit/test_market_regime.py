"""Tests for market regime detection and risk cap configuration."""

import numpy as np
import pandas as pd

from app.services.market_regime import MarketRegime, detect_market_regime, get_regime_risk_caps


def _build_close_series(values: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"Close": values}, index=dates)


def test_detect_market_regime_identifies_bull():
    close = np.concatenate([
        np.linspace(100, 150, 200),
        np.linspace(150, 220, 100),
    ])
    df = _build_close_series(close)

    regime = detect_market_regime(df, lookback_sma_long=200, lookback_sma_mid=50, deep_bear_drawdown=0.20)
    assert regime == MarketRegime.BULL


def test_detect_market_regime_identifies_bear():
    close = np.concatenate([
        np.linspace(150, 200, 200),
        np.linspace(200, 170, 100),
    ])
    df = _build_close_series(close)

    regime = detect_market_regime(df, lookback_sma_long=200, lookback_sma_mid=50, deep_bear_drawdown=0.20)
    assert regime == MarketRegime.BEAR


def test_detect_market_regime_identifies_deep_bear():
    close = np.concatenate([
        np.linspace(150, 200, 200),
        np.linspace(200, 155, 100),
    ])
    df = _build_close_series(close)

    regime = detect_market_regime(df, lookback_sma_long=200, lookback_sma_mid=50, deep_bear_drawdown=0.20)
    assert regime == MarketRegime.DEEP_BEAR


def test_get_regime_risk_caps_returns_expected_keys():
    caps = get_regime_risk_caps(MarketRegime.BULL)
    assert set(caps.keys()) == {"max_positions", "risk_pct_per_trade", "max_account_risk_pct"}
    assert caps["risk_pct_per_trade"] > 0
