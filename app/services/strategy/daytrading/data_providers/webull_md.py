"""Webull market-data client — first fallback after Twelve Data.

Two-tier strategy:
  1. Public ``quotes-gw.webullbroker.com`` endpoint (no auth). Used by Webull's
     own web app; works without you enabling anything on developer.webull.com.
     Returns full-depth bars for daily ('d1'), only the most-recent live bar
     for intraday — so it serves daily fallback well but not intraday history.
  2. Signed Webull OpenAPI at ``api.webull.com/market-data/bars`` — HMAC-SHA1
     signed per Webull SDK convention. Used only if tier 1 returns empty AND
     ``webull_app_key`` + ``webull_app_secret`` are configured. This is where
     proper intraday history comes from when your developer account has the
     market-data product enabled.

Both tiers return an empty DataFrame on any failure. Callers
(``runner.fetch_intraday`` and ``market_open._td_fetch``) treat empty as
"try the next provider in the chain" — so a region-gated developer account
or a Webull endpoint change degrades gracefully into yfinance.

Returned DataFrame shape matches the other providers: ET-tz index, columns
``Open / High / Low / Close / Volume``, ``df.attrs["source"] = "webull"``.

Signed-tier signature recipe (extracted verbatim from the official
webull-inc/openapi-python-sdk Python SDK — sha_hmac1.py + default_signature
_composer.py at commit on main, 2026-05):

  1. Build sign_headers: x-app-key, x-timestamp (ISO-8601 UTC), x-version,
     x-signature-algorithm=HMAC-SHA1, x-signature-version=1.0,
     x-signature-nonce (UUID5), plus an internal Host pseudo-header.
  2. Lowercase header keys, merge query params; sort by key; join
     k=v with '&'.
  3. Prefix with the request URI ("/market-data/bars") + '&'.
  4. URL-encode the whole string_to_sign with quote(safe='').
  5. HMAC-SHA1(string_to_sign, secret=app_secret + "&"); base64-encode the
     digest; that is the value of the x-signature header.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# yfinance-style interval → Webull *public* quotes-gw "type" parameter
_WB_INTERVAL_MAP = {
    "1m":  "m1",
    "5m":  "m5",
    "15m": "m15",
    "30m": "m30",
    "1h":  "h1",
    "1d":  "d1",
}

# yfinance-style interval → Webull *OpenAPI* signed-endpoint "timespan" enum.
# Values are the EasyEnum NAMES used by the official SDK (Timespan.M5.name etc).
_WB_SIGNED_INTERVAL_MAP = {
    "1m":  "M1",
    "5m":  "M5",
    "15m": "M15",
    "30m": "M30",
    "1h":  "M60",
    "1d":  "D",
}

# period → bar count to request (Webull returns most recent N bars)
_WB_COUNT_MAP = {
    "1d":  390,
    "2d":  780,
    "5d":  500,
    "60d": 800,
    "730d": 1000,
}

# 6-minute symbol→tickerId cache. Webull's public quotes endpoint needs an
# internal tickerId, not a symbol. The lookup is cheap (1 small HTTP call)
# but doing it on every bar fetch would double request volume.
_TICKER_ID_CACHE: dict[str, tuple[int, float]] = {}
_TICKER_ID_TTL_S = 6 * 60


def _lookup_ticker_id(symbol: str) -> int | None:
    """Resolve a US equity symbol to Webull's internal tickerId. Cached."""
    now = time.time()
    cached = _TICKER_ID_CACHE.get(symbol)
    if cached and now - cached[1] < _TICKER_ID_TTL_S:
        return cached[0]
    try:
        resp = requests.get(
            "https://quotes-gw.webullbroker.com/api/search/pc/tickers",
            params={"keyword": symbol, "pageIndex": 1, "pageSize": 5, "regionId": 6},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        for row in data.get("data", []):
            if (row.get("symbol") or "").upper() == symbol.upper():
                tid = int(row["tickerId"])
                _TICKER_ID_CACHE[symbol] = (tid, now)
                return tid
    except Exception as e:
        logger.debug("[webull] tickerId lookup failed for %s: %s", symbol, e)
    return None


def _normalise(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Common normaliser for both Webull endpoints. Returns ET-tz OHLCV."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Webull bars come in two shapes; handle both.
    if "tradeTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["tradeTime"])
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        return pd.DataFrame()
    df = df.set_index("datetime").sort_index()
    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df.dropna(subset=["Close"])


def _fetch_public(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Tier 1: public quotes-gw endpoint, no auth required."""
    timespan = _WB_INTERVAL_MAP.get(interval)
    count = _WB_COUNT_MAP.get(period, 500)
    if not timespan:
        return pd.DataFrame()
    tid = _lookup_ticker_id(symbol)
    if tid is None:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"https://quotes-gw.webullbroker.com/api/quote/charts/query",
            params={"tickerIds": tid, "type": timespan, "count": count,
                    "extendTrading": 0},
            timeout=12,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame()
        first = payload[0] or {}
        raw_bars = first.get("data") or []
        # Webull's older variant returns CSV strings like "ts,o,c,h,l,v,vwap"
        records: list[dict[str, Any]] = []
        for row in raw_bars:
            if isinstance(row, str):
                # Webull CSV row: ts, open, close, high, low, prev_close, volume, vwap
                parts = row.split(",")
                if len(parts) < 7:
                    continue
                try:
                    records.append({
                        "timestamp": int(parts[0]),
                        "open": parts[1],
                        "close": parts[2],
                        "high": parts[3],
                        "low": parts[4],
                        "volume": parts[6],
                    })
                except (ValueError, IndexError):
                    continue
            elif isinstance(row, dict):
                records.append(row)
        return _normalise(records)
    except Exception as e:
        logger.debug("[webull] public fetch failed %s %s: %s", symbol, interval, e)
        return pd.DataFrame()


_SIGNED_HOST = "api.webull.com"
_SIGNED_URI = "/market-data/bars"
_SIGNED_VERSION = "v1"


def _webull_uuid() -> str:
    """UUID5 nonce — matches the SDK's get_uuid() helper."""
    name = socket.gethostname() + str(uuid.uuid1())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _iso8601_utc_now() -> str:
    """ISO-8601 UTC timestamp without microseconds (matches SDK FORMAT_ISO_8601)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_signed_headers(app_key: str, app_secret: str, query: dict[str, str]) -> dict[str, str]:
    """Construct Webull OpenAPI signed headers per the official Python SDK.

    Signature recipe (see module docstring for the full recipe):
      string_to_sign = quote(URI + "&" + sorted_kv_join(lower(headers) | query))
      signature      = base64(HMAC-SHA1(string_to_sign, app_secret + "&"))
    """
    sign_headers = {
        "x-app-key": app_key,
        "x-timestamp": _iso8601_utc_now(),
        "x-signature-version": "1.0",
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-nonce": _webull_uuid(),
        "Host": _SIGNED_HOST,
    }

    # Merge headers (already lowercase) + query params; collisions concat with '&'
    merged: dict[str, str] = {}
    for k, v in sign_headers.items():
        merged[k.lower()] = v
    for k, v in query.items():
        existing = merged.get(k)
        merged[k] = (str(existing) + "&" + str(v)) if existing is not None else str(v)

    # Sorted k=v joined by &
    sorted_kv = "&".join(f"{k}={merged[k]}" for k in sorted(merged.keys()))
    string_to_sign = _SIGNED_URI + "&" + sorted_kv
    # GET has no body → no body_string suffix
    encoded = quote(string_to_sign, safe="")

    sig = hmac.new(
        (app_secret + "&").encode("utf-8"),
        encoded.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.standard_b64encode(sig).decode("ascii").strip()

    # Return the wire headers — drop the synthetic Host (requests sets it),
    # and add x-version + x-signature.
    wire = {k: v for k, v in sign_headers.items() if k != "Host"}
    wire["x-version"] = _SIGNED_VERSION
    wire["x-signature"] = signature
    return wire


def _parse_signed_bars(payload: Any) -> pd.DataFrame:
    """Parse the OpenAPI /market-data/bars response into our standard OHLCV shape.

    Confirmed wire format from a live US Market Data response:
      [ {"tickerId":"913256135", "time":"2026-05-21T02:20:00.000+0000",
         "open":"301.45","close":"301.36","high":"301.45","low":"301.24",
         "volume":"675"}, ... ]

    Defensive: also accept dict wrappers with "bars"/"data" keys, since the
    documented v1 schema mentions an envelope and Webull may add one later.
    """
    if isinstance(payload, dict):
        bars = payload.get("bars") or payload.get("data") or []
    elif isinstance(payload, list):
        bars = payload
    else:
        return pd.DataFrame()
    if not isinstance(bars, list) or not bars:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        ts = b.get("time") or b.get("timestamp") or b.get("tradeTime")
        if ts is None:
            continue
        rows.append({
            "timestamp": ts,
            "open":  b.get("open"),
            "high":  b.get("high"),
            "low":   b.get("low"),
            "close": b.get("close"),
            "volume": b.get("volume"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 'time' from OpenAPI may be either epoch-ms int or ISO string
    if df["timestamp"].dtype.kind in ("i", "u", "f"):
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = df.index.tz_convert(ET)
    return df.dropna(subset=["Close"])


def _fetch_signed(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Tier 2: signed Webull OpenAPI /market-data/bars. Uses app_key/app_secret."""
    try:
        from app.config import get_settings
        s = get_settings()
        app_key = s.webull_app_key
        app_secret = s.webull_app_secret
    except Exception:
        return pd.DataFrame()
    if not app_key or not app_secret:
        return pd.DataFrame()

    timespan = _WB_SIGNED_INTERVAL_MAP.get(interval)
    if not timespan:
        return pd.DataFrame()
    count = _WB_COUNT_MAP.get(period, 200)
    # OpenAPI caps count at 1200 historically; keep below that.
    count = min(count, 1200)

    query = {
        "symbol":   symbol,
        "category": "US_STOCK",
        "timespan": timespan,
        "count":    str(count),
    }
    try:
        headers = _build_signed_headers(app_key, app_secret, query)
    except Exception as e:
        logger.debug("[webull] signed header build failed %s %s: %s", symbol, interval, e)
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"https://{_SIGNED_HOST}{_SIGNED_URI}",
            headers=headers,
            params=query,
            timeout=12,
        )
    except Exception as e:
        logger.debug("[webull] signed network failed %s %s: %s", symbol, interval, e)
        return pd.DataFrame()

    if resp.status_code in (401, 403):
        # One WARNING line per process is enough — the scanner hits this in a loop.
        if not getattr(_fetch_signed, "_warned_auth", False):
            logger.warning(
                "[webull] signed endpoint auth rejected (%s). "
                "Check WEBULL_APP_KEY/WEBULL_APP_SECRET, region, and that "
                "the Market Data product is enabled for your developer app.",
                resp.status_code,
            )
            _fetch_signed._warned_auth = True  # type: ignore[attr-defined]
        return pd.DataFrame()
    if resp.status_code == 429:
        logger.warning("[webull] signed endpoint rate-limited (429) for %s — falling back",
                       symbol)
        return pd.DataFrame()
    if resp.status_code >= 400:
        logger.debug("[webull] signed HTTP %s for %s %s: %s",
                     resp.status_code, symbol, interval, resp.text[:200])
        return pd.DataFrame()

    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame()
    return _parse_signed_bars(payload)


# Minimum bars before we accept a result without escalating to the signed tier.
# Webull's public quotes-gw returns full depth for daily but only the latest
# live bar for intraday — so 1-bar intraday results are treated as too thin.
_MIN_USEFUL_BARS_INTRADAY = 5


def fetch_webull(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Public entry point. Tries public endpoint, then signed if creds present.

    Returns an empty DataFrame on every failure path — never raises. The
    caller is expected to fall back to yfinance when the result is empty.

    Routing:
      - Daily (1d): public returns full depth -> use it directly.
      - Intraday (1m/5m/15m/30m/1h): public usually returns 1 bar -> if the
        signed tier is configured, try it for real history. If signed also
        empty (auth/region/quota), fall back to whatever public did return,
        so the next provider (yfinance) gets a fair shot.
    """
    pub = _fetch_public(symbol, interval, period)
    is_intraday = interval != "1d"

    # Daily: public depth is good, return early.
    if not pub.empty and not is_intraday:
        pub.attrs["source"] = "webull"
        pub.attrs["webull_tier"] = "public"
        return pub

    # Intraday with enough bars (rare on public, but possible): accept it.
    if not pub.empty and len(pub) >= _MIN_USEFUL_BARS_INTRADAY:
        pub.attrs["source"] = "webull"
        pub.attrs["webull_tier"] = "public"
        return pub

    # Public empty or too thin → try signed tier (only does work if creds set).
    signed = _fetch_signed(symbol, interval, period)
    if not signed.empty:
        signed.attrs["source"] = "webull"
        signed.attrs["webull_tier"] = "signed"
        return signed

    # Signed also empty. If public had at least 1 bar (intraday latest tick),
    # return it rather than empty — better than nothing, and the runner will
    # decide whether the depth is sufficient for downstream consumers.
    if not pub.empty:
        pub.attrs["source"] = "webull"
        pub.attrs["webull_tier"] = "public"
        return pub

    return pd.DataFrame()
