from __future__ import annotations

"""
Universe service — builds the candidate symbol list for the scanner.

Supported universes:
  watchlist  — symbols already in strategies.json + any active assignments
  sp500      — S&P 500 constituents fetched from Wikipedia via pandas
  nasdaq100  — NASDAQ 100 constituents fetched from Wikipedia via pandas
  sp400      — S&P MidCap 400 constituents (Wikipedia)
  sp600      — S&P SmallCap 600 constituents (Wikipedia)
  sp1500     — S&P Composite 1500 = 500 + 400 + 600 (~1500 symbols)
  nifty50    — India: Nifty 50 NSE constituents (orders route to Zerodha)
  custom     — caller-supplied list
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trading-scanner/1.0)"}


def _fetch_wikipedia_symbols(url: str, label: str) -> List[str]:
    """Scrape an index's constituent tickers from a Wikipedia 'List of …' page.

    Parses the HTML with BeautifulSoup's built-in ``html.parser`` rather than
    ``pandas.read_html`` — read_html requires lxml/html5lib (not installed here)
    and its older ``timeout=`` kwarg was removed in pandas 3.x, so the previous
    approach silently fell back to hardcoded lists. We fetch with requests (which
    DOES honour a timeout), locate the constituents table (``id='constituents'``
    or the first ``wikitable``), find the Symbol/Ticker column, and read it down.

    Dots in tickers (BRK.B) are normalised to dashes (BRK-B) for yfinance.
    Returns [] on any failure so the caller can fall back / combine partials.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        r = requests.get(url, timeout=20, headers=_WIKI_HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", id="constituents") or soup.find("table", class_="wikitable")
        if table is None:
            logger.warning("[universe] %s: no constituents table found", label)
            return []

        header_cells = table.find("tr").find_all(["th", "td"])
        headers = [c.get_text(strip=True).lower() for c in header_cells]
        idx = next((i for i, h in enumerate(headers) if h in ("symbol", "ticker")), 0)

        symbols: List[str] = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= idx:
                continue
            sym = cells[idx].get_text(strip=True).replace(".", "-").upper()
            if sym and sym.replace("-", "").isalnum():
                symbols.append(sym)
        logger.info("[universe] Loaded %d %s symbols from Wikipedia", len(symbols), label)
        return symbols
    except Exception as e:
        logger.warning("[universe] Wikipedia %s fetch failed (%s)", label, e)
        return []

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
    symbols = _fetch_wikipedia_symbols(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"
    )
    return symbols or _SP500_FALLBACK


def get_sp400_symbols() -> List[str]:
    """Fetch S&P MidCap 400 constituents from Wikipedia (no hardcoded fallback)."""
    return _fetch_wikipedia_symbols(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400"
    )


def get_sp600_symbols() -> List[str]:
    """Fetch S&P SmallCap 600 constituents from Wikipedia (no hardcoded fallback)."""
    return _fetch_wikipedia_symbols(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600"
    )


def get_sp1500_symbols() -> List[str]:
    """S&P Composite 1500 = S&P 500 + S&P MidCap 400 + S&P SmallCap 600.

    Combines all three tiers (deduplicated, order-preserving). If the 400/600
    fetches fail we still return whatever did resolve — at minimum the S&P 500
    (which has its own hardcoded fallback), so a scan never comes back empty.
    """
    seen: set[str] = set()
    out: List[str] = []
    for sym in get_sp500_symbols() + get_sp400_symbols() + get_sp600_symbols():
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    logger.info("[universe] S&P Composite 1500 assembled: %d unique symbols", len(out))
    return out


def get_nasdaq100_symbols() -> List[str]:
    """Fetch NASDAQ 100 symbols from Wikipedia. Falls back to hardcoded list."""
    symbols = _fetch_wikipedia_symbols("https://en.wikipedia.org/wiki/Nasdaq-100", "NASDAQ 100")
    return symbols or _NASDAQ100_FALLBACK


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
    """Return an India tier's symbol list (nifty50/100/200/500 or nse_all).

    ``nifty500`` maps to the full Upstox NSE list (~2,466) by user decision —
    the repo has no verified 500-constituent list, so the widest accurate
    coverage is the full instrument map (same source as ``nse_all``).
    """
    import app.services.markets as mk
    if universe in ("nse_all", "nifty500"):
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
    elif universe == "sp400":
        return get_sp400_symbols()
    elif universe == "sp600":
        return get_sp600_symbols()
    elif universe == "sp1500":
        return get_sp1500_symbols()
    elif universe in ("nifty50", "nifty100", "nifty200", "nifty500", "nse_all"):
        return get_india_universe(universe)
    elif universe == "custom":
        return [s.upper().strip() for s in (custom_symbols or []) if s.strip()]
    else:
        logger.warning("[universe] Unknown universe '%s' — defaulting to watchlist", universe)
        return get_watchlist_symbols()
