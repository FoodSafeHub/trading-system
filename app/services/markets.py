from __future__ import annotations

"""Single source of truth for "which market does this symbol belong to?"

US symbols trade on Schwab/Webull during US hours; India (NSE) symbols trade on
Zerodha during India hours. Routing, the data provider, the risk engine's
market-hours check, and the dashboard all import from here so the India-vs-US
decision lives in exactly one place.

Convention for India symbols (any of):
  * appears in NIFTY_50
  * carries an explicit exchange prefix  "NSE:RELIANCE" / "BSE:..."
  * carries a yfinance suffix            "RELIANCE.NS" / "...BO"

Everything else is treated as US.
"""

from typing import Literal

Market = Literal["india", "us"]

# Nifty 50 constituents (NSE tradingsymbols, no suffix). This is the first
# India universe wired into the scanner/backtest. Expand to Nifty 200/500 later
# by adding lists here — is_india_symbol() unions them all.
NIFTY_50: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM",
    "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "DIVISLAB", "HINDALCO", "BPCL", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
]

_NIFTY_50_SET = {s.upper() for s in NIFTY_50}

# Exchange prefixes / suffixes that mark a symbol as Indian.
_INDIA_PREFIXES = ("NSE:", "BSE:")
_INDIA_SUFFIXES = (".NS", ".BO")


def normalize(symbol: str) -> str:
    """Bare tradingsymbol, upper-cased, stripped of any India prefix/suffix.

    "NSE:RELIANCE" -> "RELIANCE", "RELIANCE.NS" -> "RELIANCE", "aapl" -> "AAPL".
    """
    s = (symbol or "").upper().strip()
    for p in _INDIA_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    for suf in _INDIA_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def is_india_symbol(symbol: str) -> bool:
    """True if the symbol trades on an Indian exchange (NSE/BSE)."""
    s = (symbol or "").upper().strip()
    if s.startswith(_INDIA_PREFIXES) or s.endswith(_INDIA_SUFFIXES):
        return True
    return normalize(s) in _NIFTY_50_SET


def market_for(symbol: str) -> Market:
    """Which market a symbol belongs to: 'india' or 'us'."""
    return "india" if is_india_symbol(symbol) else "us"


def broker_for_market(market: Market, *, us_default: str = "schwab") -> str:
    """Default broker for a market. India -> zerodha; US -> us_default."""
    return "zerodha" if market == "india" else us_default


def yf_symbol(symbol: str) -> str:
    """Yahoo-Finance ticker for a symbol. India symbols get the '.NS' suffix.

    Used as a free fallback data source for India bars when Upstox isn't
    configured yet. US symbols pass through unchanged.
    """
    s = normalize(symbol)
    return f"{s}.NS" if is_india_symbol(symbol) else s
