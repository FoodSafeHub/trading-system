"""Advanced market-regime gate for momentum strategies.

Layers on top of the existing :mod:`app.services.market_regime` model. The
existing model classifies BULL/BEAR/DEEP_BEAR off SPY's 200-DMA. Momentum
strategies need a tighter filter so they don't fire into deteriorating tape:

    BULL_MOMENTUM     SPY > SMA200 AND SPY > SMA50 AND VIX < 25 AND breadth >= 50%
    BULL_CAUTION      SPY > SMA200 but one of (SMA50, VIX, breadth) is off
    BEAR_MOMENTUM     SPY < SMA200 AND SPY < SMA50 AND VIX > 25 (short setups OK)
    NO_TRADE          everything else (mean-reversion only, no momentum)

Breadth is a soft check — when the data is unavailable we degrade to a 2-of-3
filter (SPY structure + VIX). The fixture is small and cached so this doesn't
hammer the network on every signal evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import logging

import pandas as pd

from app.services.market_data.provider import get_ohlcv

logger = logging.getLogger(__name__)


class MomentumRegime(str, Enum):
    BULL_MOMENTUM = "bull_momentum"   # all 3 momentum filters pass — fire longs aggressively
    BULL_CAUTION  = "bull_caution"    # SPY > SMA200 but VIX hot OR breadth weak — half-size
    BEAR_MOMENTUM = "bear_momentum"   # SPY < SMA200 AND VIX > 25 — shorts only
    NO_TRADE      = "no_trade"        # whippy / transitional — momentum strategies stand down


@dataclass
class RegimeSnapshot:
    regime: MomentumRegime
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    vix: Optional[float]
    breadth_pct: Optional[float]
    reasons: list[str]

    @property
    def allows_long(self) -> bool:
        return self.regime in (MomentumRegime.BULL_MOMENTUM, MomentumRegime.BULL_CAUTION)

    @property
    def allows_short(self) -> bool:
        return self.regime == MomentumRegime.BEAR_MOMENTUM

    @property
    def size_multiplier(self) -> float:
        # Half-size in caution regime, full size when all filters align.
        return {
            MomentumRegime.BULL_MOMENTUM: 1.0,
            MomentumRegime.BULL_CAUTION:  0.5,
            MomentumRegime.BEAR_MOMENTUM: 0.75,
            MomentumRegime.NO_TRADE:      0.0,
        }[self.regime]

    def as_dict(self) -> dict:
        return {
            "regime":         self.regime.value,
            "spy_close":      round(self.spy_close, 2),
            "spy_sma50":      round(self.spy_sma50, 2),
            "spy_sma200":     round(self.spy_sma200, 2),
            "vix":            round(self.vix, 2) if self.vix is not None else None,
            "breadth_pct":    round(self.breadth_pct, 1) if self.breadth_pct is not None else None,
            "allows_long":    self.allows_long,
            "allows_short":   self.allows_short,
            "size_multiplier": self.size_multiplier,
            "reasons":        self.reasons,
        }


# ── Cache (5 minutes) ─────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[datetime, RegimeSnapshot]] = {}
_CACHE_TTL = timedelta(hours=1)  # daily-bar regime — no need to refresh every 5 min


def _cached(key: str) -> Optional[RegimeSnapshot]:
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, snap = hit
    if datetime.utcnow() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return snap


def _sma(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return float("nan")
    return float(series.rolling(window).mean().iloc[-1])


def _fetch_vix(ticker: str = "^VIX") -> Optional[float]:
    try:
        df = get_ohlcv(ticker, period="1mo", interval="1d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("VIX fetch (%s) failed: %s", ticker, exc)
        return None


# ── Market benchmark config ───────────────────────────────────────────────────
# Each market maps to (benchmark index, VIX ticker, VIX hot threshold, breadth
# sample). India uses the Nifty 50 index and India VIX rather than SPY/^VIX so
# India momentum strategies gate on Indian tape, not the US tape.
_MARKET_CFG = {
    "us": {
        "index": "SPY", "vix": "^VIX", "vix_hot": 25.0,
        "breadth": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "UNH", "V"],
    },
    "india": {
        "index": "^NSEI", "vix": "^INDIAVIX", "vix_hot": 20.0,
        "breadth": ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
                    "ITC", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK"],
    },
}


_breadth_cache: tuple[datetime, Optional[float]] | None = None


_breadth_cache_by_market: dict[str, tuple[datetime, Optional[float]]] = {}


def _fetch_breadth_pct(sample: list[str] | None = None, market: str = "us") -> Optional[float]:
    """% of a mega-cap sample above their 50-DMA — a breadth proxy.

    The official ``^SPXA50R`` ticker was previously tried first but yfinance no
    longer serves it — every cold call wasted ~1s on a 404, then ran the same
    fallback anyway.

    Result cached for 1 hour per market. Daily-bar strategies don't need fresher
    than that, and the surrounding RegimeSnapshot already has its own cache —
    this second layer keeps backtests fast even when the snapshot cache is
    bypassed.
    """
    hit = _breadth_cache_by_market.get(market)
    if hit is not None:
        ts, val = hit
        if datetime.utcnow() - ts < timedelta(hours=1):
            return val

    if sample is None:
        sample = _MARKET_CFG["us"]["breadth"]
    above = total = 0
    for sym in sample:
        try:
            df = get_ohlcv(sym, period="6mo", interval="1d")
            if df.empty or len(df) < 50:
                continue
            sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
            close = float(df["Close"].iloc[-1])
            total += 1
            if close > sma50:
                above += 1
        except Exception:
            continue
    result = (above / total) * 100.0 if total > 0 else None
    _breadth_cache_by_market[market] = (datetime.utcnow(), result)
    return result


def get_momentum_regime(*, refresh: bool = False, market: str = "us") -> RegimeSnapshot:
    """Return the current momentum regime snapshot for a market.

    ``market="us"`` gates on SPY / ^VIX; ``market="india"`` gates on the Nifty 50
    (^NSEI) / India VIX (^INDIAVIX) so Indian momentum strategies read Indian
    tape, not the US tape.

    Cached per-market for 1 hour — momentum strategies fire on bar close, not on
    every indicator evaluation, so daily-bar staleness is fine and keeps yfinance
    quiet.
    """
    market = market if market in _MARKET_CFG else "us"
    cfg = _MARKET_CFG[market]
    idx_ticker = cfg["index"]
    label = idx_ticker

    if not refresh:
        cached = _cached(f"momentum:{market}")
        if cached:
            return cached

    reasons: list[str] = []
    try:
        idx = get_ohlcv(idx_ticker, period="1y", interval="1d")
    except Exception as exc:
        logger.warning("%s regime fetch failed: %s", label, exc)
        snap = RegimeSnapshot(
            regime=MomentumRegime.NO_TRADE, spy_close=0.0, spy_sma50=0.0, spy_sma200=0.0,
            vix=None, breadth_pct=None,
            reasons=[f"{label} fetch failed: {exc}"],
        )
        _CACHE[f"momentum:{market}"] = (datetime.utcnow(), snap)
        return snap

    if idx.empty or len(idx) < 200:
        snap = RegimeSnapshot(
            regime=MomentumRegime.NO_TRADE, spy_close=0.0, spy_sma50=0.0, spy_sma200=0.0,
            vix=None, breadth_pct=None,
            reasons=[f"Not enough {label} history for 200-DMA"],
        )
        _CACHE[f"momentum:{market}"] = (datetime.utcnow(), snap)
        return snap

    close = float(idx["Close"].iloc[-1])
    sma50 = _sma(idx["Close"], 50)
    sma200 = _sma(idx["Close"], 200)
    vix = _fetch_vix(cfg["vix"])
    vix_hot = cfg["vix_hot"]
    breadth = _fetch_breadth_pct(cfg["breadth"], market=market)

    spy_above_200 = close > sma200
    spy_above_50 = close > sma50
    vix_ok = (vix is None) or (vix < vix_hot)     # missing VIX → pass
    breadth_ok = (breadth is None) or (breadth >= 50.0)

    # Classify.
    if spy_above_200 and spy_above_50 and vix_ok and breadth_ok:
        regime = MomentumRegime.BULL_MOMENTUM
        reasons.append(f"{label} {close:.2f} > SMA50 {sma50:.2f} > SMA200 {sma200:.2f}")
        if vix is not None:
            reasons.append(f"VIX {vix:.1f} < {vix_hot:.0f}")
        if breadth is not None:
            reasons.append(f"Breadth {breadth:.0f}% above 50DMA")
    elif spy_above_200 and (not spy_above_50 or not vix_ok or not breadth_ok):
        regime = MomentumRegime.BULL_CAUTION
        reasons.append(f"{label} > SMA200 but ")
        if not spy_above_50:
            reasons.append(f"{label} below SMA50 ({sma50:.2f})")
        if not vix_ok:
            reasons.append(f"VIX hot ({vix:.1f})")
        if not breadth_ok:
            reasons.append(f"Breadth weak ({breadth:.0f}%)")
    elif (not spy_above_200) and (not spy_above_50) and (vix is not None and vix > vix_hot):
        regime = MomentumRegime.BEAR_MOMENTUM
        reasons.append(f"{label} {close:.2f} < SMA200 {sma200:.2f}, VIX {vix:.1f} > {vix_hot:.0f}")
    else:
        regime = MomentumRegime.NO_TRADE
        reasons.append("Transitional tape — no clean momentum setup")
        if not spy_above_200:
            reasons.append(f"{label} {close:.2f} < SMA200 {sma200:.2f}")

    snap = RegimeSnapshot(
        regime=regime,
        spy_close=close, spy_sma50=sma50, spy_sma200=sma200,
        vix=vix, breadth_pct=breadth,
        reasons=reasons,
    )
    _CACHE[f"momentum:{market}"] = (datetime.utcnow(), snap)
    return snap


def clear_cache() -> None:
    _CACHE.clear()
    _breadth_cache_by_market.clear()


# ── Point-in-time historical regime (for backtests) ──────────────────────────
#
# The default get_momentum_regime() fetches LIVE data and is cached for the
# current session — fine for live signal generation, fatal in a backtest loop
# where every historical bar would see the same live snapshot (a lookahead /
# leak). get_momentum_regime_at() takes pre-sliced point-in-time inputs and
# runs the same classifier, so backtests can compute one snapshot per bar
# without any reference to "today".
#
# Inputs are deliberately series-shaped, not single floats: the classifier
# needs SMA(50) and SMA(200), and the caller controls the slice so we cannot
# accidentally peek beyond as_of_date.


def _classify(
    label: str,
    close: float,
    sma50: float,
    sma200: float,
    vix: Optional[float],
    vix_hot: float,
    breadth: Optional[float],
) -> tuple[MomentumRegime, list[str]]:
    """Pure classifier — same logic as get_momentum_regime, no I/O.

    Returns (regime, reasons). Used by both the live path and the
    historical point-in-time path so behaviour cannot diverge.
    """
    import math
    if any(map(math.isnan, (close, sma50, sma200))):
        return MomentumRegime.NO_TRADE, [f"{label} insufficient history for SMAs"]

    spy_above_200 = close > sma200
    spy_above_50 = close > sma50
    vix_ok = (vix is None) or (vix < vix_hot)
    breadth_ok = (breadth is None) or (breadth >= 50.0)

    reasons: list[str] = []
    if spy_above_200 and spy_above_50 and vix_ok and breadth_ok:
        regime = MomentumRegime.BULL_MOMENTUM
        reasons.append(f"{label} {close:.2f} > SMA50 {sma50:.2f} > SMA200 {sma200:.2f}")
        if vix is not None:
            reasons.append(f"VIX {vix:.1f} < {vix_hot:.0f}")
        if breadth is not None:
            reasons.append(f"Breadth {breadth:.0f}% above 50DMA")
    elif spy_above_200 and (not spy_above_50 or not vix_ok or not breadth_ok):
        regime = MomentumRegime.BULL_CAUTION
        reasons.append(f"{label} > SMA200 but ")
        if not spy_above_50:
            reasons.append(f"{label} below SMA50 ({sma50:.2f})")
        if not vix_ok and vix is not None:
            reasons.append(f"VIX hot ({vix:.1f})")
        if not breadth_ok and breadth is not None:
            reasons.append(f"Breadth weak ({breadth:.0f}%)")
    elif (not spy_above_200) and (not spy_above_50) and (vix is not None and vix > vix_hot):
        regime = MomentumRegime.BEAR_MOMENTUM
        reasons.append(f"{label} {close:.2f} < SMA200 {sma200:.2f}, VIX {vix:.1f} > {vix_hot:.0f}")
    else:
        regime = MomentumRegime.NO_TRADE
        reasons.append("Transitional tape — no clean momentum setup")
        if not spy_above_200:
            reasons.append(f"{label} {close:.2f} < SMA200 {sma200:.2f}")

    return regime, reasons


def get_momentum_regime_at(
    as_of_date: pd.Timestamp,
    *,
    index_close: pd.Series,
    vix_close: Optional[pd.Series] = None,
    breadth_close_by_symbol: Optional[dict[str, pd.Series]] = None,
    market: str = "us",
) -> RegimeSnapshot:
    """Compute a momentum regime snapshot AS-OF a historical date.

    Used by the backtest engine to avoid the live-data leak that
    get_momentum_regime() introduces when called per historical bar.

    All series are sliced to ``as_of_date`` internally — the caller can pass
    the full series without thinking about it. Breadth is computed from any
    symbol series supplied: for each, whether its close is above its own
    50-bar SMA. Missing VIX / breadth degrade gracefully (same as live).

    Parameters
    ----------
    as_of_date         : the bar timestamp to evaluate
    index_close        : full daily Close series for the market index
                         (SPY / ^NSEI) — the function slices internally
    vix_close          : full daily Close series for the relevant VIX
                         ticker (^VIX / ^INDIAVIX). Optional.
    breadth_close_by_symbol : dict of {symbol: Close series} for the breadth
                         sample. Optional; missing → breadth gate degrades
                         to None (same as live).
    market             : "us" or "india" — picks the VIX threshold/labels.
    """
    cfg = _MARKET_CFG.get(market, _MARKET_CFG["us"])
    label = cfg["index"]
    vix_hot = cfg["vix_hot"]

    idx = index_close.loc[index_close.index <= as_of_date]
    if len(idx) < 200:
        return RegimeSnapshot(
            regime=MomentumRegime.NO_TRADE,
            spy_close=0.0, spy_sma50=0.0, spy_sma200=0.0,
            vix=None, breadth_pct=None,
            reasons=[f"Not enough {label} history at {as_of_date.date()} for 200-DMA"],
        )

    close = float(idx.iloc[-1])
    sma50 = _sma(idx, 50)
    sma200 = _sma(idx, 200)

    vix_val: Optional[float] = None
    if vix_close is not None and not vix_close.empty:
        v = vix_close.loc[vix_close.index <= as_of_date]
        if not v.empty:
            vix_val = float(v.iloc[-1])

    breadth_pct: Optional[float] = None
    if breadth_close_by_symbol:
        above = total = 0
        for sym, series in breadth_close_by_symbol.items():
            s = series.loc[series.index <= as_of_date]
            if len(s) < 50:
                continue
            sma_sym = float(s.rolling(50).mean().iloc[-1])
            cs = float(s.iloc[-1])
            total += 1
            if cs > sma_sym:
                above += 1
        if total > 0:
            breadth_pct = (above / total) * 100.0

    regime, reasons = _classify(label, close, sma50, sma200, vix_val, vix_hot, breadth_pct)
    return RegimeSnapshot(
        regime=regime,
        spy_close=close, spy_sma50=sma50, spy_sma200=sma200,
        vix=vix_val, breadth_pct=breadth_pct,
        reasons=reasons,
    )
