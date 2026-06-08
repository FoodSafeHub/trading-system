"""
Market hours utilities and pre-market regime detection for day trading.

Regime is evaluated at 9:45 AM ET (after first 15 minutes complete).
  BULL_OPEN  = SPY opened above prior close AND current price > VWAP
  BEAR_OPEN  = SPY opened below prior close AND current price < VWAP
  CHOPPY     = neither condition clearly met
  PRE_MARKET = before 9:30 AM ET
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Tuple

_log = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _fetch_twelvedata_raw(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Twelve Data tier only — empty DF on any failure. Internal helper."""
    _TD_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "1d": "1day"}
    _SIZE_MAP = {"1d": 390, "2d": 780, "5d": 500, "60d": 800}
    from app.services.strategy.daytrading.data_providers import td_breaker
    if td_breaker.is_tripped():
        return pd.DataFrame()
    try:
        from app.config import get_settings
        api_key = get_settings().twelve_data_api_key
        if not api_key:
            return pd.DataFrame()
        td_interval = _TD_MAP.get(interval)
        if not td_interval:
            return pd.DataFrame()
        outputsize = _SIZE_MAP.get(period, 500)
        import requests
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": symbol, "interval": td_interval,
                    "outputsize": outputsize, "timezone": "America/New_York",
                    "apikey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            msg = data.get("message", "") or ""
            td_breaker.note_response_text(msg)
            _log.debug("[twelvedata] %s %s: %s", symbol, interval, msg)
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(ET)
        else:
            df.index = df.index.tz_convert(ET)
        return df
    except Exception as e:
        _log.debug("[twelvedata] fetch failed %s %s: %s", symbol, interval, e)
        return pd.DataFrame()


def _td_fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Provider-routed bar fetch: Twelve Data -> Webull -> yfinance.

    Name kept as ``_td_fetch`` for backwards compatibility with the scanner
    and premarket-volume helpers that import it. Returns an OHLCV DataFrame
    in ET tz with ``df.attrs["source"]`` set to the provider that served it,
    or an empty DataFrame if all three providers fail.

    interval: "1m","5m","15m","1d"   period: "1d","2d","5d","60d"
    """
    # 1) Twelve Data
    df = _fetch_twelvedata_raw(symbol, interval, period)
    if not df.empty:
        df.attrs["source"] = "twelvedata"
        return df

    # 2) Webull
    try:
        from app.services.strategy.daytrading.data_providers.webull_md import fetch_webull
        wb = fetch_webull(symbol, interval, period)
    except Exception as e:
        _log.debug("[webull] import/fetch failed %s %s: %s", symbol, interval, e)
        wb = pd.DataFrame()
    if not wb.empty:
        _log.info("[fallback] TD -> Webull for %s %s %s (%d bars)",
                  symbol, interval, period, len(wb))
        wb.attrs["source"] = "webull"
        return wb

    # 3) yfinance
    try:
        import yfinance as _yf
        yf_df = _yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        _log.debug("[yfinance] download failed %s %s: %s", symbol, interval, e)
        return pd.DataFrame()
    if yf_df is None or yf_df.empty:
        return pd.DataFrame()
    yf_df = _normalise_yf(yf_df)
    _log.info("[fallback] Webull -> yfinance for %s %s %s (%d bars)",
              symbol, interval, period, len(yf_df))
    yf_df.attrs["source"] = "yfinance"
    return yf_df


def _normalise_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and convert index to ET timezone."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        idx = idx.tz_convert(ET)
    df.index = idx
    return df

MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(15, 45)
REGIME_EVAL_TIME = time(9, 45)
GAP_FADE_CUTOFF = time(11, 0)
LAST_ENTRY_TIME = time(15, 15)

# ── Market-aware sessions ────────────────────────────────────────────────────
# Day-trading strategies anchor on the cash session (opening range, last-entry
# cutoff, EOD flat). Those clock times differ by market, so strategies must use
# the session for the symbol they're evaluating instead of hardcoding US ET.
IST = ZoneInfo("Asia/Kolkata")


class MarketSession:
    """Session clock for one market. All times are naive local times in `tz`."""
    __slots__ = ("tz", "open_time", "close_time", "regime_eval_time",
                 "gap_cutoff", "last_entry_time")

    def __init__(self, tz, open_time, close_time, regime_eval_time,
                 gap_cutoff, last_entry_time):
        self.tz = tz
        self.open_time = open_time
        self.close_time = close_time
        self.regime_eval_time = regime_eval_time
        self.gap_cutoff = gap_cutoff
        self.last_entry_time = last_entry_time

    def after_open(self, minutes: int) -> time:
        """Clock time `minutes` after the session open (session-local, no date)."""
        base = datetime(2000, 1, 1, self.open_time.hour, self.open_time.minute)
        return (base + timedelta(minutes=minutes)).time()

    def before_close(self, minutes: int) -> time:
        """Clock time `minutes` before the session close."""
        base = datetime(2000, 1, 1, self.close_time.hour, self.close_time.minute)
        return (base - timedelta(minutes=minutes)).time()


# US equities (NYSE/Nasdaq), times in America/New_York.
_SESSION_US = MarketSession(
    tz=ET, open_time=MARKET_OPEN_TIME, close_time=MARKET_CLOSE_TIME,
    regime_eval_time=REGIME_EVAL_TIME, gap_cutoff=GAP_FADE_CUTOFF,
    last_entry_time=LAST_ENTRY_TIME,
)
# India (NSE) cash session 09:15–15:30 IST. Mirror the US offsets from the open:
# regime eval +15m, gap cutoff +75m, last entry 15m before close, flatten 15:30.
_SESSION_IN = MarketSession(
    tz=IST, open_time=time(9, 15), close_time=time(15, 30),
    regime_eval_time=time(9, 30), gap_cutoff=time(10, 30),
    last_entry_time=time(15, 15),
)


def market_session(symbol: str) -> MarketSession:
    """Return the cash-session clock for `symbol`'s market (NSE for India, else US)."""
    try:
        from app.services.markets import is_india_symbol
        if is_india_symbol(symbol):
            return _SESSION_IN
    except Exception:
        pass
    return _SESSION_US


def localize_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Index → the symbol's market timezone (IST for India, ET for US).

    Replaces the per-strategy `tz_localize(ET)` helpers so session-relative logic
    (opening range, entry cutoffs) lines up with the correct market.
    """
    if df.empty:
        return df
    sess = market_session(symbol)
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize(sess.tz)
    else:
        idx = idx.tz_convert(sess.tz)
    df = df.copy()
    df.index = idx
    return df


def now_et() -> datetime:
    return datetime.now(ET)


def is_market_open(symbol: str = "") -> bool:
    """True when the cash session for `symbol` is open. Defaults to US ET."""
    sess = market_session(symbol) if symbol else _SESSION_US
    t = datetime.now(sess.tz).time()
    return sess.open_time <= t <= sess.close_time


def is_pre_market(symbol: str = "") -> bool:
    """True before the cash open for `symbol`'s market."""
    sess = market_session(symbol) if symbol else _SESSION_US
    t = datetime.now(sess.tz).time()
    return t < sess.open_time


def is_past_last_entry(symbol: str = "") -> bool:
    """True after the last-entry cutoff for `symbol`'s market."""
    sess = market_session(symbol) if symbol else _SESSION_US
    t = datetime.now(sess.tz).time()
    return t >= sess.last_entry_time


def market_status() -> dict:
    now = now_et()
    t = now.time()
    if t < MARKET_OPEN_TIME:
        status = "PRE_MARKET"
    elif t > time(16, 0):
        status = "CLOSED"
    elif t > MARKET_CLOSE_TIME:
        status = "CLOSING"
    else:
        status = "OPEN"
    return {
        "status": status,
        "is_open": status == "OPEN",
        "time_et": now.strftime("%H:%M:%S %Z"),
        "date": now.strftime("%Y-%m-%d"),
    }


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP that resets at 9:30 AM ET every trading day.
    Grouped by calendar date so overnight data never bleeds into today's VWAP.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = typical * df["Volume"]
    # group by date so VWAP resets each session
    dates = df.index.normalize()
    cum_tpv = tpv.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    return cum_tpv / cum_vol


def validate_vwap_resets(df: pd.DataFrame) -> dict:
    """
    Verify that VWAP resets correctly at each session boundary.

    Returns a dict with:
      ok: bool — True if all dates show a proper reset
      dates_checked: int
      issues: list of (date, reason) for any anomalies
      open_vwap_by_date: {date: vwap_at_9:35} — sanity snapshot
    """
    if df.empty:
        return {"ok": False, "dates_checked": 0, "issues": ["Empty dataframe"], "open_vwap_by_date": {}}

    vwap = compute_vwap(df)
    dates = sorted(set(df.index.date))
    issues: list[str] = []
    open_vwap: dict[str, float] = {}

    for date in dates:
        day = df[df.index.date == date]
        day_vwap = vwap[df.index.date == date]
        if day.empty:
            continue

        first_bar_vwap = float(day_vwap.iloc[0])
        first_bar_close = float(day["Close"].iloc[0])

        # VWAP at open should equal typical price of first bar (cumsum of 1 bar)
        expected_vwap_open = (float(day["High"].iloc[0]) + float(day["Low"].iloc[0]) + first_bar_close) / 3
        diff = abs(first_bar_vwap - expected_vwap_open)

        if diff > expected_vwap_open * 0.001:   # >0.1% off means bleed from prior day
            issues.append(f"{date}: VWAP at open={first_bar_vwap:.4f} expected={expected_vwap_open:.4f} — possible daily reset failure")
            _log.warning("VWAP reset issue on %s: got %.4f expected %.4f", date, first_bar_vwap, expected_vwap_open)
        else:
            _log.debug("VWAP reset OK on %s: %.4f", date, first_bar_vwap)

        # Log VWAP at ~9:35 (second bar) and last bar
        if len(day_vwap) >= 2:
            vwap_935 = float(day_vwap.iloc[1])
            open_vwap[str(date)] = round(vwap_935, 4)
            _log.debug("VWAP at 9:35 on %s: %.4f | EOD: %.4f", date, vwap_935, float(day_vwap.iloc[-1]))

    return {
        "ok": len(issues) == 0,
        "dates_checked": len(dates),
        "issues": issues,
        "open_vwap_by_date": open_vwap,
    }


def get_spy_regime(df_spy_5m: pd.DataFrame | None = None) -> Tuple[str, float, float, str]:
    """
    Return (regime, spy_vs_vwap_pct).
    Downloads SPY 5m data if not provided.
    """
    if df_spy_5m is None:
        try:
            df_spy_5m = _td_fetch("SPY", "5m", "2d")
            if df_spy_5m.empty:
                df_spy_5m = yf.download("SPY", period="2d", interval="5m", progress=False)
                if df_spy_5m.empty:
                    return "CHOPPY", 0.0, 0.0, "unknown"
                df_spy_5m = _normalise_yf(df_spy_5m)
        except Exception:
            return "CHOPPY", 0.0, 0.0, "unknown"

    today = now_et().date()
    today_bars = df_spy_5m[df_spy_5m.index.date == today]
    if today_bars.empty:
        return "CHOPPY", 0.0, 0.0, "unknown"

    # prior close = last bar from previous session
    prev_bars = df_spy_5m[df_spy_5m.index.date < today]
    if prev_bars.empty:
        return "CHOPPY", 0.0, 0.0, "unknown"
    prior_close = float(prev_bars["Close"].iloc[-1])

    open_price = float(today_bars["Open"].iloc[0])
    current_price = float(today_bars["Close"].iloc[-1])

    vwap_series = compute_vwap(today_bars)
    current_vwap = float(vwap_series.iloc[-1])
    spy_vs_vwap_pct = round((current_price - current_vwap) / current_vwap * 100, 3)

    opened_above = open_price > prior_close
    above_vwap = current_price > current_vwap

    if opened_above and above_vwap:
        regime = "BULL_OPEN"
    elif (not opened_above) and (not above_vwap):
        regime = "BEAR_OPEN"
    else:
        regime = "CHOPPY"

    # Compute gap using prior close already computed above
    gap_pct = round((open_price - prior_close) / prior_close * 100, 3) if prior_close > 0 else 0.0
    if gap_pct > 1.0:
        gap_type = "up_gap"
    elif gap_pct < -1.0:
        gap_type = "down_gap"
    else:
        gap_type = "normal"

    return regime, spy_vs_vwap_pct, gap_pct, gap_type


def get_prior_close(symbol: str) -> float | None:
    """
    Fetch the most recent prior session's closing price from daily bars.
    Uses yfinance daily data so it correctly handles weekends and holidays.
    Returns None if data is unavailable.
    """
    try:
        df = _td_fetch(symbol, "1d", "5d")
        if df.empty:
            df = yf.download(symbol, period="5d", interval="1d", progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
        today = now_et().date()
        prior = df[df.index.date < today]
        if prior.empty:
            return None
        return float(prior["Close"].iloc[-1])
    except Exception:
        return None


def get_premarket_gap(symbol: str, open_price: float | None = None) -> dict:
    """
    Compute the pre-market gap for a symbol.

    Gap % = (today_open - prior_close) / prior_close * 100
    Classification:
      > +1.0% -> "up_gap"
      < -1.0% -> "down_gap"
      else    -> "normal"

    If open_price is None, fetches today's first 5m bar open from yfinance.
    Returns dict: {gap_pct, gap_type, open_price, prior_close, symbol}
    """
    prior_close = get_prior_close(symbol)
    if prior_close is None or prior_close <= 0:
        return {"gap_pct": 0.0, "gap_type": "unknown", "open_price": None, "prior_close": None, "symbol": symbol}

    if open_price is None:
        try:
            df = _td_fetch(symbol, "5m", "1d")
            if df.empty:
                df = yf.download(symbol, period="1d", interval="5m", progress=False)
                if not df.empty:
                    df = _normalise_yf(df)
            if not df.empty:
                today = now_et().date()
                today_bars = df[df.index.date == today]
                if not today_bars.empty:
                    open_price = float(today_bars["Open"].iloc[0])
        except Exception:
            pass

    if open_price is None:
        return {"gap_pct": 0.0, "gap_type": "unknown", "open_price": None, "prior_close": round(prior_close, 4), "symbol": symbol}

    gap_pct = (open_price - prior_close) / prior_close * 100

    if gap_pct > 1.0:
        gap_type = "up_gap"
    elif gap_pct < -1.0:
        gap_type = "down_gap"
    else:
        gap_type = "normal"

    _log.info("%s gap: %.2f%% (%s) open=%.2f prior_close=%.2f", symbol, gap_pct, gap_type, open_price, prior_close)

    return {
        "gap_pct": round(gap_pct, 3),
        "gap_type": gap_type,
        "open_price": round(open_price, 4),
        "prior_close": round(prior_close, 4),
        "symbol": symbol,
    }


def regime_allows_strategy(regime: str, strategy_name: str) -> bool:
    """
    BULL_OPEN / CHOPPY: all strategies are allowed (each self-filters inside
    generate_signals via internal regime checks and returns [] when not applicable).

    BEAR_OPEN: only strategies with an explicit short/sell path pass through.
    Strategies that are long-only (ORBBreakout) safely return [] on BEAR_OPEN,
    so including them here is harmless and avoids masking them in backtests.
    """
    bear_open_allowed = {
        # Active core strategies (6)
        "ORBBreakout",           # long-only; returns [] on BEAR_OPEN — safe to include
        "VWAPMeanReversion",     # short side added for BEAR_OPEN
        "EMAMomentum",           # bearish crossover path on BEAR_OPEN
        "OpeningGapFade",        # gap-up fades short on BEAR_OPEN
        "SupertrendTrend",       # 15m ST bearish → shorts; self-filters direction
        "NRSqueezeBreakout",     # close below lower band on BEAR_OPEN
        # Retired strategies — kept for backtest compatibility
        "VolumeSpikeReversal",
        "BollingerMomentum",
        "NarrowRangeBreakout",
        "EngulfingVolumeSurge",
        "ThreeBarPush",
        "HammerShootingStar",
    }
    if regime == "BEAR_OPEN" and strategy_name not in bear_open_allowed:
        return False
    return True


def apply_choppy_penalty(confidence: float, regime: str) -> float:
    if regime == "CHOPPY":
        return max(0.0, confidence - 0.2)
    return confidence
