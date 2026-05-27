from __future__ import annotations

"""Upstox market-data client — India quotes + historical bars.

Used as the India data source while orders execute on Zerodha. Standard OAuth2
(authorization code grant). The access token expires daily (~03:30 IST) and is
stored in the shared ``broker_tokens`` table under ``broker="upstox"``; re-login
via GET /upstox/login.

Two public surfaces:
  * ``get_ltp(symbols)``      → {ticker: last_price}     (for quotes / unrealized P/L)
  * ``fetch_bars(symbol, …)`` → OHLCV DataFrame, ET-naive→IST tz, df.attrs["source"]="upstox"

Everything degrades to empty/None on failure so callers fall back cleanly.

Endpoints (Upstox v2/v3):
  auth dialog : https://api.upstox.com/v2/login/authorization/dialog
  token       : https://api.upstox.com/v2/login/authorization/token
  ltp         : GET https://api.upstox.com/v2/market-quote/ltp?instrument_key=...
  historical  : GET https://api.upstox.com/v3/historical-candle/{key}/{unit}/{interval}/{to}/{from}
"""

import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.config import get_settings
from app.services.marketdata import upstox_instruments as instr

logger = logging.getLogger(__name__)

UPSTOX_API_BASE = "https://api.upstox.com"
AUTH_DIALOG = f"{UPSTOX_API_BASE}/v2/login/authorization/dialog"
TOKEN_URL = f"{UPSTOX_API_BASE}/v2/login/authorization/token"
IST = ZoneInfo("Asia/Kolkata")

# yfinance-style interval → Upstox v3 (unit, interval) path components.
_INTERVAL_MAP = {
    "1m":  ("minutes", "1"),
    "5m":  ("minutes", "5"),
    "15m": ("minutes", "15"),
    "30m": ("minutes", "30"),
    "1h":  ("hours", "1"),
    "1d":  ("days", "1"),
    "1wk": ("weeks", "1"),
}

# Upstox v3 caps the date span of a SINGLE historical request for intraday
# intervals — a one-shot 90-day 5m request returns an empty candle list. The
# deeper history exists; it just has to be pulled in chunks and stitched. Map
# each interval to the largest span (days) we'll request per call. `None` = no
# chunking (daily/weekly have effectively unlimited span).
_MAX_SPAN_DAYS = {
    "1m": 28, "5m": 28, "15m": 28, "30m": 28, "1h": 90,
    "1d": None, "1wk": None,
}

# period string → how many days of history to request.
# NOTE: every period the app can request must be listed. A missing key falls
# back to 366 days, which silently truncates long windows — e.g. a "10y"
# request would return only ~1y of bars, starving backtests/calibration.
_PERIOD_DAYS = {
    "1d": 1, "2d": 2, "5d": 5, "1mo": 31, "3mo": 93,
    "6mo": 186, "1y": 366, "2y": 731, "5y": 1827,
    "10y": 3653, "max": 3653,
    # Day-trading page period strings (intraday windows).
    "30d": 30, "60d": 60, "90d": 90, "180d": 180, "730d": 730,
}


def is_configured() -> bool:
    s = get_settings()
    return bool(s.upstox_api_key and s.upstox_api_secret)


# ── Auth ────────────────────────────────────────────────────────────────────

def get_login_url(state: str | None = None) -> str:
    s = get_settings()
    params = {
        "response_type": "code",
        "client_id": s.upstox_api_key,
        "redirect_uri": s.upstox_redirect_uri,
    }
    if state:
        params["state"] = state
    return f"{AUTH_DIALOG}?{urllib.parse.urlencode(params)}"


async def exchange_code(code: str) -> None:
    """Exchange the OAuth code for a daily access token; persist it."""
    s = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_URL,
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code": code,
                "client_id": s.upstox_api_key,
                "client_secret": s.upstox_api_secret,
                "redirect_uri": s.upstox_redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("Upstox token exchange returned no access_token")
        _save_token(token)
        logger.info("[upstox] Access token stored (valid until ~03:30 IST tomorrow)")


def _next_expiry_utc() -> datetime:
    """Upstox tokens die ~03:30 IST. Next such instant, in UTC."""
    now = datetime.now(tz=IST)
    exp = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if now >= exp:
        exp += timedelta(days=1)
    return exp.astimezone(ZoneInfo("UTC"))


def _save_token(token: str) -> None:
    try:
        from app.db import SessionLocal
        from app.models.broker_tokens import BrokerToken
        with SessionLocal() as db:
            row = db.query(BrokerToken).filter_by(broker="upstox").first()
            if not row:
                row = BrokerToken(broker="upstox")
                db.add(row)
            row.access_token = token
            row.refresh_token = None
            row.token_expiry = _next_expiry_utc()
            db.commit()
    except Exception as exc:
        logger.error("[upstox] failed to save token: %s", exc)


def _load_token() -> str:
    """Stored token from DB, else the .env bootstrap value, else ''."""
    try:
        from app.db import SessionLocal
        from app.models.broker_tokens import BrokerToken
        with SessionLocal() as db:
            row = db.query(BrokerToken).filter_by(broker="upstox").first()
            if row and row.access_token:
                return row.access_token
    except Exception as exc:
        logger.debug("[upstox] token load failed: %s", exc)
    return get_settings().upstox_access_token or ""


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_load_token()}", "Accept": "application/json"}


# ── Quotes ──────────────────────────────────────────────────────────────────

def get_ltp(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Last traded price per ticker. Missing/failed symbols map to None."""
    out: Dict[str, Optional[float]] = {s.upper().strip(): None for s in symbols}
    if not is_configured() or not _load_token():
        return out
    key_map = instr.resolve_many(symbols)            # ticker -> instrument_key
    if not key_map:
        return out
    inst_to_ticker = {v: k for k, v in key_map.items()}
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{UPSTOX_API_BASE}/v2/market-quote/ltp",
                headers=_headers(),
                params={"instrument_key": ",".join(key_map.values())},
            )
            if resp.status_code in (401, 403):
                logger.warning("[upstox] LTP auth rejected (%s) — re-login at /upstox/login", resp.status_code)
                return out
            resp.raise_for_status()
            data = resp.json().get("data", {}) or {}
    except Exception as exc:
        logger.debug("[upstox] LTP fetch failed: %s", exc)
        return out

    # Upstox keys the response by a normalized "EXCH_SEG:TRADINGSYMBOL" string,
    # not the instrument_key we sent — match on the instrument_key field inside.
    for _resp_key, info in data.items():
        ikey = info.get("instrument_token") or info.get("instrument_key")
        ltp = info.get("last_price")
        ticker = inst_to_ticker.get(ikey)
        if ticker is not None and ltp is not None:
            out[ticker] = float(ltp)
    return out


# ── Historical bars ───────────────────────────────────────────────────────────

def fetch_bars(symbol: str, interval: str = "1d", period: str = "1y") -> pd.DataFrame:
    """OHLCV history for an India ticker. Empty DataFrame on any failure.

    Shape matches the other providers: tz-aware (IST) DatetimeIndex, columns
    Open/High/Low/Close/Volume, df.attrs["source"] = "upstox".
    """
    if not is_configured() or not _load_token():
        return pd.DataFrame()
    key = instr.resolve(symbol)
    if not key:
        logger.debug("[upstox] no instrument_key for %s", symbol)
        return pd.DataFrame()
    unit_interval = _INTERVAL_MAP.get(interval)
    if not unit_interval:
        return pd.DataFrame()
    unit, ivl = unit_interval

    to_date = datetime.now(tz=IST).date()
    start_date = to_date - timedelta(days=_PERIOD_DAYS.get(period, 366))

    # Upstox caps the span of a single intraday request. Walk backwards from
    # `to_date` in <= max_span chunks and stitch — a one-shot 90d 5m request
    # otherwise returns an empty list even though the deeper history exists.
    max_span = _MAX_SPAN_DAYS.get(interval)
    candles: list = []
    with httpx.Client(timeout=30) as client:
        chunk_to = to_date
        while chunk_to >= start_date:
            chunk_from = start_date if max_span is None else max(start_date, chunk_to - timedelta(days=max_span))
            # Path order per v3 docs: /{key}/{unit}/{interval}/{to_date}/{from_date}
            path = (
                f"/v3/historical-candle/{urllib.parse.quote(key, safe='')}"
                f"/{unit}/{ivl}/{chunk_to.isoformat()}/{chunk_from.isoformat()}"
            )
            try:
                resp = client.get(f"{UPSTOX_API_BASE}{path}", headers=_headers())
                if resp.status_code in (401, 403):
                    logger.warning("[upstox] historical auth rejected (%s) — re-login at /upstox/login", resp.status_code)
                    return pd.DataFrame()
                resp.raise_for_status()
                chunk = (resp.json().get("data", {}) or {}).get("candles", []) or []
            except Exception as exc:
                logger.debug("[upstox] historical fetch failed %s %s %s..%s: %s",
                             symbol, interval, chunk_from, chunk_to, exc)
                chunk = []
            candles.extend(chunk)
            if max_span is None or chunk_from <= start_date:
                break
            # Next chunk ends the day before this chunk started (no overlap).
            chunk_to = chunk_from - timedelta(days=1)

    if not candles:
        return pd.DataFrame()
    # Each candle: [ts, open, high, low, close, volume, open_interest]
    df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume", "OI"])
    df = df.drop_duplicates(subset=["ts"])
    df["datetime"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    df = df.set_index("datetime").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    df.attrs["source"] = "upstox"
    return df
