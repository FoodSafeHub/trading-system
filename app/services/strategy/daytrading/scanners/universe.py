"""Symbol universe loader for the day-trading scanner.

Loads the full US-listed common-stock universe (~7,400 names across NYSE,
NASDAQ, and AMEX) from the official NASDAQ Trader symbol directory files.
The iShares IWV holdings endpoint we tried earlier now serves an anti-bot
HTML page instead of CSV, so NASDAQ Trader is the working alternative —
it's free, updated nightly, and covers a wider universe than Russell 3000.

Sources (in order):
  1. Local cache file (refreshed daily).
  2. NASDAQ Trader symbol files (nasdaqlisted.txt + otherlisted.txt).
  3. Hard-coded fallback (the original 22 mega-caps — keeps the scanner
     working if both the network and cache are unavailable).

The cache lives in ``data/scanner_cache/us_listed.json`` and is auto-refreshed
once per day. Float data is cached separately in ``us_listed_float.json`` and
also refreshed daily — float doesn't change intraday so daily granularity is
correct.
"""
from __future__ import annotations

import io
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Cache files live next to the rest of the project data (project root / data /)
_CACHE_DIR = Path(__file__).resolve().parents[5] / "data" / "scanner_cache"
_UNIVERSE_CACHE = _CACHE_DIR / "us_listed.json"
_FLOAT_CACHE = _CACHE_DIR / "us_listed_float.json"

# NASDAQ Trader publishes the canonical US-listed security directories.
# Both files are pipe-delimited and refreshed nightly. The final line is a
# "File Creation Time" footer that we strip before parsing.
_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Hard-coded fallback — original 22 mega-caps. Keeps the scanner functional
# when both the cache and the network are unreachable.
_FALLBACK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    "JPM", "BAC", "GS", "MS",
    "SPY", "QQQ", "IWM", "XLK", "XLF",
    "PLTR", "COIN", "MSTR", "HOOD", "SOFI",
]


def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_is_fresh(path: Path, *, max_age_days: int = 1) -> bool:
    """True if the cache file exists and was written within max_age_days."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached_at = payload.get("cached_at", "")
        cached_date = datetime.fromisoformat(cached_at).date()
        return (date.today() - cached_date).days < max_age_days
    except Exception:
        return False


def _read_cache(path: Path) -> Optional[dict]:
    """Return the parsed cache payload, or None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read cache %s: %s", path, e)
        return None


def _write_cache(path: Path, payload: dict) -> None:
    _ensure_cache_dir()
    payload = {**payload, "cached_at": datetime.now().isoformat()}
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write cache %s: %s", path, e)


def _parse_nasdaq_trader_file(text: str) -> pd.DataFrame:
    """Strip the 'File Creation Time' footer and parse the pipe-delimited body."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("File Creation")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


# Security-name keywords that indicate a non-common-stock instrument we don't
# want to scan: warrants/rights/units (SPAC mechanics), preferred shares, notes,
# depositary receipts wrappers. Common stock and ADR ordinary shares pass.
_NON_COMMON_NAME_PATTERNS = (
    " Warrant",
    " Warrants",
    " Right",
    " Rights",
    " Unit",
    " Units",
    " Preferred",
    " Pref ",
    " Notes",
    " Debenture",
    " Trust Preferred",
    "% Note",
)


def _is_common_stock_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    upper = " " + name
    return not any(p in upper for p in _NON_COMMON_NAME_PATTERNS)


def _fetch_us_listed_from_nasdaq_trader() -> list[str]:
    """Download NASDAQ Trader symbol directories and return common stock tickers.

    Combines nasdaqlisted.txt (NASDAQ-listed) and otherlisted.txt (NYSE / AMEX /
    NYSE Arca via the ACT Symbol column). Filters out:
      - Test issues and ETFs (per the file's own flags).
      - Warrants, rights, units, preferreds, notes (per Security Name keywords).
      - Tickers with '.', '$', '^', etc. — yfinance commonly can't quote these.

    Returns an empty list on any parse/network failure — caller falls back
    to cache or the hard-coded list.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    symbols: list[str] = []

    try:
        r = requests.get(_NASDAQ_LISTED_URL, timeout=30, headers=headers)
        r.raise_for_status()
        df = _parse_nasdaq_trader_file(r.text)
        mask = (
            (df["Test Issue"] == "N")
            & (df["ETF"] == "N")
            & df["Security Name"].apply(_is_common_stock_name)
        )
        symbols.extend(df.loc[mask, "Symbol"].dropna().astype(str).tolist())
        logger.info("NASDAQ listed: %d common stocks", int(mask.sum()))
    except Exception as e:
        logger.warning("Failed to fetch NASDAQ listed: %s", e)

    try:
        r = requests.get(_OTHER_LISTED_URL, timeout=30, headers=headers)
        r.raise_for_status()
        df = _parse_nasdaq_trader_file(r.text)
        mask = (
            (df["Test Issue"] == "N")
            & (df["ETF"] == "N")
            & df["Security Name"].apply(_is_common_stock_name)
        )
        symbols.extend(df.loc[mask, "ACT Symbol"].dropna().astype(str).tolist())
        logger.info("Other listed (NYSE/AMEX): %d common stocks", int(mask.sum()))
    except Exception as e:
        logger.warning("Failed to fetch other listed: %s", e)

    # Normalize and filter. We drop tickers with non-alphanumeric characters —
    # preferred shares (".PR"), warrants ("W"), units ("U") often don't have
    # the price/volume data we need to scan, and yfinance commonly fails on them.
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        t = raw.strip().upper()
        if not t or not t.isascii():
            continue
        if any(ch in t for ch in (".", "$", "^", "/", " ")):
            continue
        if not t.replace("-", "").isalnum():
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)

    logger.info("Combined US-listed universe: %d unique common stocks", len(out))
    return out


def load_universe(*, force_refresh: bool = False) -> list[str]:
    """Return the US-listed common-stock universe, refreshed once per day.

    ``force_refresh=True`` bypasses the cache and re-downloads. Use this if
    the cache appears stale or after a major listing/delisting event.
    """
    _ensure_cache_dir()

    if not force_refresh and _cache_is_fresh(_UNIVERSE_CACHE):
        payload = _read_cache(_UNIVERSE_CACHE)
        if payload and isinstance(payload.get("tickers"), list) and payload["tickers"]:
            return payload["tickers"]

    tickers = _fetch_us_listed_from_nasdaq_trader()
    if tickers:
        _write_cache(_UNIVERSE_CACHE, {"tickers": tickers, "source": "nasdaq_trader"})
        return tickers

    # Network failure — fall through to whatever's in the cache, even if stale
    payload = _read_cache(_UNIVERSE_CACHE)
    if payload and isinstance(payload.get("tickers"), list) and payload["tickers"]:
        logger.warning("Using stale US-listed cache (network fetch failed)")
        return payload["tickers"]

    logger.error("No US-listed source available — falling back to 22 mega-caps")
    return list(_FALLBACK_UNIVERSE)


# ── India universe ───────────────────────────────────────────────────────────

def load_india_universe() -> list[str]:
    """Return the curated NSE universe (Nifty 200) as bare symbols.

    Returns plain NSE trading symbols (e.g. "RELIANCE", "TATAMOTORS").
    The India scanner path uses Upstox for all data fetching — no .NS
    suffix manipulation needed.
    """
    from app.services.markets import NIFTY_200
    return list(NIFTY_200)


# ── Float cache ──────────────────────────────────────────────────────────────


def load_float_cache() -> dict[str, float]:
    """Return the cached float data ``{symbol: shares_float}``.

    Float data is fetched lazily on the first scan of the day and reused for
    subsequent scans (see the scanner's float-cache update path).
    """
    payload = _read_cache(_FLOAT_CACHE)
    if not payload:
        return {}
    floats = payload.get("floats", {})
    return floats if isinstance(floats, dict) else {}


def save_float_cache(floats: dict[str, float]) -> None:
    """Persist the float cache."""
    _write_cache(_FLOAT_CACHE, {"floats": floats})


def float_cache_is_fresh() -> bool:
    """True if the float cache was written today."""
    return _cache_is_fresh(_FLOAT_CACHE, max_age_days=1)
