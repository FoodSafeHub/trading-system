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
_CACHE_TTL = timedelta(minutes=5)


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


def _fetch_vix() -> Optional[float]:
    try:
        df = get_ohlcv("^VIX", period="1mo", interval="1d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("VIX fetch failed: %s", exc)
        return None


def _fetch_breadth_pct() -> Optional[float]:
    """% of S&P 500 stocks above their 50-DMA.

    Proxy via ``$SPXA50R`` (S&P 500 stocks above 50-DMA). yfinance carries it
    as ``%5ESPXA50R`` on some mirrors and not others — fall back to comparing
    a handful of mega-caps to their own 50-DMAs if the index isn't reachable.
    """
    try:
        df = get_ohlcv("^SPXA50R", period="2mo", interval="1d")
        if not df.empty and not pd.isna(df["Close"].iloc[-1]):
            return float(df["Close"].iloc[-1])
    except Exception:
        pass

    # Fallback breadth proxy — sample 10 mega-caps and count how many are above
    # their own 50-DMA. Crude but correlated with the official figure.
    sample = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "UNH", "V"]
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
    if total == 0:
        return None
    return (above / total) * 100.0


def get_momentum_regime(*, refresh: bool = False) -> RegimeSnapshot:
    """Return the current momentum regime snapshot.

    Cached for 5 minutes — momentum strategies fire on bar close, not on every
    indicator evaluation, so a few-minute staleness is fine and keeps yfinance
    quiet.
    """
    if not refresh:
        cached = _cached("momentum")
        if cached:
            return cached

    reasons: list[str] = []
    try:
        spy = get_ohlcv("SPY", period="1y", interval="1d")
    except Exception as exc:
        logger.warning("SPY regime fetch failed: %s", exc)
        snap = RegimeSnapshot(
            regime=MomentumRegime.NO_TRADE, spy_close=0.0, spy_sma50=0.0, spy_sma200=0.0,
            vix=None, breadth_pct=None,
            reasons=[f"SPY fetch failed: {exc}"],
        )
        _CACHE["momentum"] = (datetime.utcnow(), snap)
        return snap

    if spy.empty or len(spy) < 200:
        snap = RegimeSnapshot(
            regime=MomentumRegime.NO_TRADE, spy_close=0.0, spy_sma50=0.0, spy_sma200=0.0,
            vix=None, breadth_pct=None,
            reasons=["Not enough SPY history for 200-DMA"],
        )
        _CACHE["momentum"] = (datetime.utcnow(), snap)
        return snap

    close = float(spy["Close"].iloc[-1])
    sma50 = _sma(spy["Close"], 50)
    sma200 = _sma(spy["Close"], 200)
    vix = _fetch_vix()
    breadth = _fetch_breadth_pct()

    spy_above_200 = close > sma200
    spy_above_50 = close > sma50
    vix_ok = (vix is None) or (vix < 25.0)     # missing VIX → pass
    breadth_ok = (breadth is None) or (breadth >= 50.0)

    # Classify.
    if spy_above_200 and spy_above_50 and vix_ok and breadth_ok:
        regime = MomentumRegime.BULL_MOMENTUM
        reasons.append(f"SPY {close:.2f} > SMA50 {sma50:.2f} > SMA200 {sma200:.2f}")
        if vix is not None:
            reasons.append(f"VIX {vix:.1f} < 25")
        if breadth is not None:
            reasons.append(f"Breadth {breadth:.0f}% above 50DMA")
    elif spy_above_200 and (not spy_above_50 or not vix_ok or not breadth_ok):
        regime = MomentumRegime.BULL_CAUTION
        reasons.append(f"SPY > SMA200 but ")
        if not spy_above_50:
            reasons.append(f"SPY below SMA50 ({sma50:.2f})")
        if not vix_ok:
            reasons.append(f"VIX hot ({vix:.1f})")
        if not breadth_ok:
            reasons.append(f"Breadth weak ({breadth:.0f}%)")
    elif (not spy_above_200) and (not spy_above_50) and (vix is not None and vix > 25.0):
        regime = MomentumRegime.BEAR_MOMENTUM
        reasons.append(f"SPY {close:.2f} < SMA200 {sma200:.2f}, VIX {vix:.1f} > 25")
    else:
        regime = MomentumRegime.NO_TRADE
        reasons.append("Transitional tape — no clean momentum setup")
        if not spy_above_200:
            reasons.append(f"SPY {close:.2f} < SMA200 {sma200:.2f}")

    snap = RegimeSnapshot(
        regime=regime,
        spy_close=close, spy_sma50=sma50, spy_sma200=sma200,
        vix=vix, breadth_pct=breadth,
        reasons=reasons,
    )
    _CACHE["momentum"] = (datetime.utcnow(), snap)
    return snap


def clear_cache() -> None:
    _CACHE.clear()
