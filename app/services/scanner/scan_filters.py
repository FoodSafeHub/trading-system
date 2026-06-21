from __future__ import annotations

"""
Liquidity and price filters applied before running strategies.
Rejects symbols that are too cheap, too thinly traded, or missing data.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    passed: bool
    symbol: str
    price: float | None = None
    avg_volume: float | None = None
    reason: str = ""


def apply_filters(
    symbol: str,
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_volume: float = 500_000.0,
    min_bars: int = 60,
    max_price: float = 0.0,
) -> FilterResult:
    """
    Run all pre-strategy filters on an OHLCV DataFrame.
    Returns FilterResult indicating pass/fail and why.

    ``max_price`` of 0 (or any value <= ``min_price``) disables the ceiling.
    The shares-float filter is applied separately in the scanner service
    because it needs an external (slow) fetch and should only run for symbols
    that already cleared these cheap price/volume gates.
    """
    if df is None or df.empty:
        return FilterResult(passed=False, symbol=symbol, reason="No price data returned")

    if len(df) < min_bars:
        return FilterResult(
            passed=False, symbol=symbol,
            reason=f"Insufficient history: {len(df)} bars (need {min_bars})",
        )

    last_price = float(df["Close"].iloc[-1])
    if last_price < min_price:
        return FilterResult(
            passed=False, symbol=symbol, price=last_price,
            reason=f"Price ${last_price:.2f} below minimum ${min_price:.2f}",
        )

    if max_price and max_price > min_price and last_price > max_price:
        return FilterResult(
            passed=False, symbol=symbol, price=last_price,
            reason=f"Price ${last_price:.2f} above maximum ${max_price:.2f}",
        )

    # Average daily volume over last 20 trading days
    vol_col = "Volume" if "Volume" in df.columns else None
    avg_vol: float | None = None
    if vol_col:
        avg_vol = float(df[vol_col].iloc[-20:].mean())
        if avg_vol < min_avg_volume:
            return FilterResult(
                passed=False, symbol=symbol, price=last_price, avg_volume=avg_vol,
                reason=f"Avg volume {avg_vol:,.0f} below minimum {min_avg_volume:,.0f}",
            )

    return FilterResult(passed=True, symbol=symbol, price=last_price, avg_volume=avg_vol)


def fetch_float_shares(symbol: str) -> float:
    """Return the shares float (or shares outstanding) for ``symbol``, else 0.0.

    Mirrors the day-trading scanner's proven approach:
      1. yfinance ``fast_info.shares`` first — light, fast, and resilient to the
         crumb-401 / rate-limit storms that the heavier ``.info`` endpoint hits.
         This is shares outstanding, a close upper bound on float for most names.
      2. ``.info["floatShares"]`` (then ``sharesOutstanding``) as a fallback for
         the exact figure when ``fast_info`` is unavailable.
      3. 0.0 on total failure — the caller treats 0 as "unknown".
    """
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        try:
            shares = getattr(t.fast_info, "shares", None)
            if shares and shares > 0:
                return float(shares)
        except Exception:
            pass
        info = t.info or {}
        val = info.get("floatShares") or info.get("sharesOutstanding")
        return float(val) if val else 0.0
    except Exception:
        return 0.0


def passes_float_filter(
    float_shares: float,
    min_float: float = 0.0,
    max_float: float = 0.0,
) -> tuple[bool, str]:
    """Apply the min/max shares-float band. Returns (passed, reason).

    A float of 0 means "unknown" — when a float band is active, unknown floats
    are rejected (we can't prove they're inside the band). Both 0 ⇒ no filter.
    """
    if not min_float and not max_float:
        return True, ""
    if float_shares <= 0:
        return False, "Float unknown (no data); float filter active"
    if min_float and float_shares < min_float:
        return False, f"Float {float_shares/1e6:,.1f}M below minimum {min_float/1e6:,.1f}M"
    if max_float and float_shares > max_float:
        return False, f"Float {float_shares/1e6:,.1f}M above maximum {max_float/1e6:,.1f}M"
    return True, ""
