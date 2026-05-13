from __future__ import annotations

"""
Shared SPY-based market regime filter used by all 5 new strategies.

BULL  = SPY close > SMA(200)   → all 5 strategies active, full position sizing
BEAR  = SPY close <= SMA(200)  → only Strategy 4 & 5 allowed, 50% position size

Usage:
    spy_df = yf.download("SPY", start="2015-01-01", end="2025-12-31")
    regime_series = get_market_regime(spy_df)   # pd.Series of "BULL" / "BEAR"
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


# ── Shared indicator helpers ──────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bb(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Returns (upper, middle, lower, bandwidth, percent_b)."""
    middle = _sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle.replace(0, 1e-9)
    percent_b = (series - lower) / (upper - lower).replace(0, 1e-9)
    return upper, middle, lower, bandwidth, percent_b


# ── Regime detection ─────────────────────────────────────────────────────────

def get_market_regime(spy_df: pd.DataFrame) -> pd.Series:
    """
    Returns a daily Series indexed by date: 'BULL' or 'BEAR'.

    BULL = SPY close > SMA(200)
    BEAR = SPY close <= SMA(200)

    Requires at least 200 rows of SPY daily OHLCV.
    Dates before warmup (< 200 bars) are marked 'BEAR' conservatively.
    """
    close = spy_df["Close"].squeeze()
    sma200 = _sma(close, 200)
    regime = pd.Series(
        ["BULL" if (not pd.isna(sma200.iloc[i]) and close.iloc[i] > sma200.iloc[i]) else "BEAR"
         for i in range(len(close))],
        index=spy_df.index,
        name="regime",
    )
    return regime


def get_spy_sma200_pct(spy_df: pd.DataFrame) -> pd.Series:
    """Returns (SPY_close / SMA200 - 1) * 100 as percentage above/below SMA200."""
    close = spy_df["Close"].squeeze()
    sma200 = _sma(close, 200)
    return ((close - sma200) / sma200.replace(0, 1e-9) * 100).rename("spy_sma200_pct")


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    Unified signal returned by all 5 new strategies.

    side    : "BUY" or "SELL"
    date    : bar date (pd.Timestamp or str)
    price   : signal price (typically current close)
    stop_loss   : hard stop price
    take_profit : hard target price
    confidence  : 0.0–1.0
    reason  : human-readable explanation
    """
    date: object
    symbol: str
    strategy_name: str
    side: str            # "BUY" | "SELL"
    price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    regime: str = "BULL"
    hold_bars: int = 0   # suggested max hold (strategy default)


# ── Base class ────────────────────────────────────────────────────────────────

class NewStrategy:
    """
    Base class for the 5 new strategies.

    Subclasses implement:
        name            : str
        default_config  : dict
        generate_signals(df, symbol, config, spy_df) -> list[Signal]

    df      : target symbol daily OHLCV with columns Open/High/Low/Close/Volume
    spy_df  : SPY daily OHLCV aligned to the same calendar (used for regime)
    """

    name: str = "base"
    default_config: dict = {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        config: Optional[dict] = None,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> list:
        raise NotImplementedError

    def _get_regime_series(
        self,
        df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Align regime to df's index.
        Falls back to 'BULL' for every bar if SPY data is unavailable.
        """
        if spy_df is None or spy_df.empty:
            return pd.Series("BULL", index=df.index)
        regime = get_market_regime(spy_df)
        # Reindex to target symbol dates; forward-fill missing SPY dates
        regime = regime.reindex(df.index, method="ffill")
        # Any still-missing dates default to BULL (conservative for entry suppression)
        return regime.fillna("BULL")

    def _spy_pct_below_sma200(
        self,
        spy_df: Optional[pd.DataFrame],
        date: object,
    ) -> float:
        """Returns how far SPY is below SMA200 on a given date (positive = below, negative = above)."""
        if spy_df is None or spy_df.empty:
            return 0.0
        pct = get_spy_sma200_pct(spy_df)
        pct = pct.reindex(pd.DatetimeIndex([date]), method="ffill")
        if pct.empty or pd.isna(pct.iloc[0]):
            return 0.0
        # Return how far BELOW sma200 (negative of pct means below)
        return float(-pct.iloc[0])
