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

# Nifty Next 50 — ranks 51–100 by market cap. Nifty 100 = NIFTY_50 + this.
NIFTY_NEXT_50: list[str] = [
    "ADANIGREEN", "ADANIPOWER", "ADANIENSOL", "AMBUJACEM", "BAJAJHLDNG",
    "BANKBARODA", "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "DMART", "GAIL", "GODREJCP", "HAVELLS",
    "HAL", "ICICIGI", "ICICIPRULI", "IOC", "INDIGO", "IRCTC", "JINDALSTEL",
    "JIOFIN", "LICI", "MARICO", "MUTHOOTFIN", "NAUKRI", "PIDILITIND",
    "PFC", "PNB", "RECLTD", "SBICARD", "SIEMENS", "SRF", "SHREECEM",
    "TATAPOWER", "TORNTPHARM", "TRENT", "TVSMOTOR", "UNITDSPR", "VBL",
    "VEDL", "ZOMATO", "ZYDUSLIFE", "IDEA", "INDUSTOWER", "MOTHERSON",
]

# Nifty Midcap 150 selection — adds liquid mid-caps (ranks ~101–250) so the
# 200/500 tiers have breadth. Curated; not every Nifty index member, but the
# liquid, backtestable ones. Anything missing can still be free-typed (the
# Upstox instrument map resolves ~2,466 NSE equities regardless of this list).
NIFTY_MIDSMALL_EXTRA: list[str] = [
    "AUBANK", "ABCAPITAL", "ABFRL", "ALKEM", "APLAPOLLO", "ASHOKLEY",
    "ASTRAL", "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BHARATFORG",
    "BHEL", "BIOCON", "CGPOWER", "COFORGE", "CONCOR", "CUMMINSIND",
    "DALBHARAT", "DEEPAKNTR", "DIXON", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "FORTIS", "GMRINFRA", "GODREJPROP", "GUJGASLTD", "HDFCAMC", "HINDPETRO",
    "IDFCFIRSTB", "INDHOTEL", "INDUSINDBK", "IRFC", "JUBLFOOD", "KPITTECH",
    "LTF", "LTTS", "LAURUSLABS", "LUPIN", "MFSL", "MPHASIS", "MRF",
    "NMDC", "NHPC", "OBEROIRLTY", "OFSS", "PAGEIND", "PATANJALI",
    "PAYTM", "PERSISTENT", "PETRONET", "PHOENIXLTD", "PIIND", "POLYCAB",
    "POONAWALLA", "PRESTIGE", "RVNL", "SAIL", "SBFC", "SOLARINDS",
    "SONACOMS", "SUNTV", "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM",
    "TATAELXSI", "TIINDIA", "TORNTPOWER", "UBL", "UNIONBANK", "UPL",
    "VOLTAS", "YESBANK", "ZEEL", "ABBOTINDIA", "ACC", "BHARTIHEXA",
    "CDSL", "CAMS", "MAXHEALTH", "MAZDOCK", "KALYANKJIL", "KEI",
]

NIFTY_100: list[str] = NIFTY_50 + NIFTY_NEXT_50
# Nifty 200/500 are approximated as 100 + the mid/small extras above. They are
# NOT the exact index constituents — they're a liquid, backtestable superset.
# For exact or exotic names, free-type the ticker (resolved via Upstox).
NIFTY_200: list[str] = NIFTY_100 + NIFTY_MIDSMALL_EXTRA
NIFTY_500: list[str] = NIFTY_200  # same curated pool today; widen via nse_all for the full market

_NIFTY_50_SET = {s.upper() for s in NIFTY_50}
# Union of every curated India ticker — used by is_india_symbol so a midcap typed
# without a prefix/suffix is still recognized as Indian and routed to Zerodha.
_INDIA_CURATED_SET = {s.upper() for s in NIFTY_200}


def nse_all_symbols() -> list[str]:
    """Every NSE equity Upstox can resolve (~2,466). Falls back to the curated
    Nifty 200 pool if the Upstox instrument map is unavailable."""
    try:
        from app.services.marketdata import upstox_instruments as instr
        instr._ensure_map()
        syms = sorted(
            k.split(":", 1)[1] for k in instr._MAP if k.startswith("NSE:")
        )
        if syms:
            return syms
    except Exception:
        pass
    return list(NIFTY_200)

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
    """True if the symbol trades on an Indian exchange (NSE/BSE).

    Resolution order:
      1. Explicit NSE:/BSE: prefix or .NS/.BO suffix -> India.
      2. Curated Nifty 200 pool -> India (works with no Upstox token).
      3. Upstox NSE instrument map (2400+ symbols) -> India. This catches
         mid/small-caps outside the curated list (e.g. AIIL) that Upstox can
         still fetch. US majors (AAPL, NVDA, SPY, ...) are absent from the NSE
         map, so this does not hijack US tickers. Falls back silently if the
         map can't be loaded.
    """
    s = (symbol or "").upper().strip()
    if s.startswith(_INDIA_PREFIXES) or s.endswith(_INDIA_SUFFIXES):
        return True
    norm = normalize(s)
    if norm in _INDIA_CURATED_SET:
        return True
    try:
        from app.services.marketdata import upstox_instruments as _instr
        return _instr.resolve(norm) is not None
    except Exception:
        return False


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
