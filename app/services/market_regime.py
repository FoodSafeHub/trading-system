from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

from app.config import get_settings
from app.services.market_data.provider import get_ohlcv


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    DEEP_BEAR = "deep_bear"


def detect_market_regime(
    spy_ohlcv: pd.DataFrame,
    lookback_sma_long: int = 200,
    lookback_sma_mid: int = 50,
    deep_bear_drawdown: float = 0.20,
) -> MarketRegime:
    """
    Determine current regime from SPY daily data:
      - BULL: close > SMA(200) AND SMA(50) >= SMA(200)
      - BEAR: close < SMA(200)
      - DEEP_BEAR: close < SMA(200) AND drawdown from 52-week high <= -deep_bear_drawdown
    Returns one of MarketRegime.BULL / BEAR / DEEP_BEAR.
    """
    if spy_ohlcv.empty or len(spy_ohlcv) < lookback_sma_long:
        raise ValueError("Not enough SPY data for regime detection")

    close = spy_ohlcv["Close"]
    sma_long = close.rolling(lookback_sma_long).mean()
    sma_mid = close.rolling(lookback_sma_mid).mean()
    high_52 = close.rolling(252, min_periods=1).max()

    c_now = float(close.iloc[-1])
    l_now = float(sma_long.iloc[-1])
    m_now = float(sma_mid.iloc[-1])
    dd = (c_now / float(high_52.iloc[-1])) - 1 if high_52.iloc[-1] > 0 else 0.0

    if c_now < l_now and dd <= -deep_bear_drawdown:
        return MarketRegime.DEEP_BEAR
    if c_now < l_now:
        return MarketRegime.BEAR
    if c_now > l_now and m_now >= l_now:
        return MarketRegime.BULL
    return MarketRegime.BEAR


def detect_regime_series(
    spy_ohlcv: pd.DataFrame,
    lookback_sma_long: int = 200,
    lookback_sma_mid: int = 50,
    deep_bear_drawdown: float = 0.20,
) -> pd.Series:
    """Return a series of market regimes indexed by SPY date."""
    if spy_ohlcv.empty:
        return pd.Series(dtype=object)

    close = spy_ohlcv["Close"]
    sma_long = close.rolling(lookback_sma_long).mean()
    sma_mid = close.rolling(lookback_sma_mid).mean()
    high_52 = close.rolling(252, min_periods=1).max()

    regimes = []
    for idx in close.index:
        c_now = float(close.loc[idx])
        l_now = float(sma_long.loc[idx]) if not pd.isna(sma_long.loc[idx]) else float("nan")
        m_now = float(sma_mid.loc[idx]) if not pd.isna(sma_mid.loc[idx]) else float("nan")
        h52 = float(high_52.loc[idx]) if not pd.isna(high_52.loc[idx]) else float("nan")
        if pd.isna(l_now) or pd.isna(m_now) or pd.isna(h52):
            regimes.append(MarketRegime.BEAR)
            continue
        dd = (c_now / h52) - 1 if h52 > 0 else 0.0
        if c_now < l_now and dd <= -deep_bear_drawdown:
            regimes.append(MarketRegime.DEEP_BEAR)
        elif c_now < l_now:
            regimes.append(MarketRegime.BEAR)
        elif c_now > l_now and m_now >= l_now:
            regimes.append(MarketRegime.BULL)
        else:
            regimes.append(MarketRegime.BEAR)

    return pd.Series(regimes, index=spy_ohlcv.index)


def get_current_regime(
    date: pd.Timestamp,
    benchmark_symbol: str | None = None,
) -> MarketRegime:
    """Load benchmark data up to `date` and return the current market regime."""
    settings = get_settings()
    symbol = benchmark_symbol or getattr(settings, "regime_benchmark_symbol", "SPY")
    df = get_ohlcv(symbol, period="2y", interval="1d")
    if df.empty:
        raise ValueError(f"No benchmark data available for {symbol}")
    ts = pd.Timestamp(date)
    # Align tz-awareness with the benchmark index, else the comparison raises
    # "Cannot compare tz-naive and tz-aware". Callers that pass df.index[-1]
    # are already aware; a plain datetime (e.g. the scheduler's utcnow()) is
    # naive and gets localized/converted to the index tz here.
    idx_tz = getattr(df.index, "tz", None)
    if idx_tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize("UTC").tz_convert(idx_tz)
    elif idx_tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    history = df.loc[df.index <= ts]
    if history.empty:
        raise ValueError(f"Benchmark data for {symbol} does not include {ts.date()}")
    lookback_sma_long = getattr(settings, "regime_sma_long", 200)
    lookback_sma_mid = getattr(settings, "regime_sma_mid", 50)
    deep_bear_drawdown = getattr(settings, "regime_deep_bear_drawdown", 0.20)
    return detect_market_regime(history, lookback_sma_long, lookback_sma_mid, deep_bear_drawdown)


def get_regime_risk_caps(regime: MarketRegime) -> dict[str, Any]:
    settings = get_settings()
    return {
        MarketRegime.BULL: {
            "max_positions": getattr(settings, "regime_max_positions_bull", 10),
            "risk_pct_per_trade": getattr(settings, "regime_risk_pct_bull", 0.01),
            "max_account_risk_pct": getattr(settings, "regime_max_account_risk_bull", 0.20),
        },
        MarketRegime.BEAR: {
            "max_positions": getattr(settings, "regime_max_positions_bear", 3),
            "risk_pct_per_trade": getattr(settings, "regime_risk_pct_bear", 0.003),
            "max_account_risk_pct": getattr(settings, "regime_max_account_risk_bear", 0.08),
        },
        MarketRegime.DEEP_BEAR: {
            "max_positions": getattr(settings, "regime_max_positions_deep_bear", 1),
            "risk_pct_per_trade": getattr(settings, "regime_risk_pct_deep_bear", 0.002),
            "max_account_risk_pct": getattr(settings, "regime_max_account_risk_deep_bear", 0.04),
        },
    }[regime]
