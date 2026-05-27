from __future__ import annotations

"""
Universe service — builds the candidate symbol list for the scanner.

Supported universes:
  watchlist  — symbols already in strategies.json + any active assignments
  sp500      — S&P 500 constituents fetched from Wikipedia via pandas
  nasdaq100  — NASDAQ 100 constituents fetched from Wikipedia via pandas
  nifty50    — India: Nifty 50 NSE constituents (orders route to Zerodha)
  custom     — caller-supplied list
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Hardcoded fallbacks used when the web fetch fails
_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","MA","PG","COST","HD","MRK","ABBV","CVX",
    "KO","PEP","ADBE","WMT","CRM","BAC","TMO","MCD","CSCO","ACN","ABT","LIN",
    "DHR","TXN","NKE","ORCL","PM","NEE","QCOM","WFC","AMD","UPS","MS","INTU",
    "BMY","RTX","AMGN","T","HON","COP","SPGI","GS","LOW","CAT","SBUX","AMAT",
    "DE","AXP","BLK","ISRG","GILD","ADI","VRTX","MDLZ","SYK","REGN","ZTS",
    "CI","PLD","DUK","ETN","SO","MO","BDX","BSX","NOC","MMC","AON","CME",
    "HUM","USB","TJX","ICE","EMR","NSC","ITW","ELV","APD","ECL","GD","PGR",
    "TGT","MCO","FCX","SHW","WM","KLAC","LRCX","MCHP","ON","SNPS","CDNS",
    "SPY","QQQ","IWM",
]

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "ADBE","PEP","CSCO","INTU","QCOM","TXN","AMD","AMAT","SBUX","ISRG",
    "VRTX","REGN","MU","LRCX","ADI","KLAC","MELI","CDNS","SNPS","ORLY",
    "MNST","CTAS","NFLX","PAYX","FTNT","KHC","ROST","IDXX","DXCM","ILMN",
    "EXC","ODFL","SGEN","BIIB","DLTR","CHTR","FAST","VRSK","CEG","FANG",
    "AEP","XEL","WBA","ZS","PANW","CRWD","TEAM","WDAY","OKTA","DDOG",
    "ABNB","ALGN","ENPH","MRNA","LCID","PYPL","EBAY","INTC","BKNG",
    "QQQ",
]


def get_watchlist_symbols() -> List[str]:
    """Return all symbols currently in strategies.json + active DB assignments."""
    symbols: set[str] = set()
    try:
        import json, os
        if os.path.exists("strategies.json"):
            with open("strategies.json") as f:
                configs = json.load(f)
            symbols.update(c["symbol"].upper() for c in configs if c.get("enabled", True))
    except Exception as e:
        logger.warning("[universe] Could not load strategies.json: %s", e)

    try:
        from app.db import SessionLocal
        from app.models.assignments import SymbolStrategyAssignment
        with SessionLocal() as db:
            rows = db.query(SymbolStrategyAssignment).filter_by(enabled=True).all()
            symbols.update(r.symbol.upper() for r in rows)
    except Exception as e:
        logger.warning("[universe] Could not load assignments: %s", e)

    return sorted(symbols) or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD"]


def get_sp500_symbols() -> List[str]:
    """Fetch S&P 500 symbols from Wikipedia. Falls back to hardcoded list."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=10)
        df = tables[0]
        symbols = df["Symbol"].str.replace(".", "-", regex=False).str.upper().tolist()
        logger.info("[universe] Loaded %d S&P 500 symbols from Wikipedia", len(symbols))
        return symbols
    except Exception as e:
        logger.warning("[universe] Wikipedia S&P 500 fetch failed (%s) — using fallback", e)
        return _SP500_FALLBACK


def get_nasdaq100_symbols() -> List[str]:
    """Fetch NASDAQ 100 symbols from Wikipedia. Falls back to hardcoded list."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", timeout=10)
        # Find the table with a 'Ticker' or 'Symbol' column
        for df in tables:
            for col in df.columns:
                if str(col).lower() in ("ticker", "symbol"):
                    symbols = df[col].str.upper().tolist()
                    logger.info("[universe] Loaded %d NASDAQ 100 symbols from Wikipedia", len(symbols))
                    return symbols
    except Exception as e:
        logger.warning("[universe] Wikipedia NASDAQ 100 fetch failed (%s) — using fallback", e)
    return _NASDAQ100_FALLBACK


def get_nifty50_symbols() -> List[str]:
    """India: Nifty 50 NSE tradingsymbols. Sourced from app.services.markets."""
    from app.services.markets import NIFTY_50
    return list(NIFTY_50)


# Wider India tiers. Curated supersets live in app.services.markets; nse_all
# pulls the full ~2,466-symbol Upstox instrument map.
_INDIA_UNIVERSES = {
    "nifty50":  "NIFTY_50",
    "nifty100": "NIFTY_100",
    "nifty200": "NIFTY_200",
    "nifty500": "NIFTY_500",
}


def get_india_universe(universe: str) -> List[str]:
    """Return an India tier's symbol list (nifty50/100/200/500 or nse_all)."""
    import app.services.markets as mk
    if universe == "nse_all":
        return mk.nse_all_symbols()
    attr = _INDIA_UNIVERSES.get(universe, "NIFTY_50")
    return list(getattr(mk, attr))


def get_universe(universe: str, custom_symbols: list[str] | None = None) -> List[str]:
    """Return the symbol list for the requested universe."""
    if universe == "watchlist":
        return get_watchlist_symbols()
    elif universe == "sp500":
        return get_sp500_symbols()
    elif universe == "nasdaq100":
        return get_nasdaq100_symbols()
    elif universe in ("nifty50", "nifty100", "nifty200", "nifty500", "nse_all"):
        return get_india_universe(universe)
    elif universe == "custom":
        return [s.upper().strip() for s in (custom_symbols or []) if s.strip()]
    else:
        logger.warning("[universe] Unknown universe '%s' — defaulting to watchlist", universe)
        return get_watchlist_symbols()
