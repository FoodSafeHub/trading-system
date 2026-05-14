"""
Market State Classifier — rule-based, fully explainable.

Classifies the current intraday session into one of 5 states using SPY data:
  TREND_UP   — price above VWAP, ORB broken upward, ATR% normal
  TREND_DOWN — price below VWAP, ORB broken downward, bearish internals
  CHOPPY     — price oscillating around VWAP, narrow range, low follow-through
  HIGH_VOL   — ATR% significantly above 20-day avg (panic / news reaction)
  NEWS_RISK  — volume surge in first 15 min suggesting catalyst (avoid new trades)

Each classification comes with a confidence score 0.0–1.0 and a reasons list
so the brain (and the trader) can understand exactly why the state was assigned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, compute_vwap

# States
TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
CHOPPY = "CHOPPY"
HIGH_VOL = "HIGH_VOL"
NEWS_RISK = "NEWS_RISK"
UNKNOWN = "UNKNOWN"

ALL_STATES = [TREND_UP, TREND_DOWN, CHOPPY, HIGH_VOL, NEWS_RISK]

# Thresholds
_ATR_HIGH_VOL_MULT = 1.8       # ATR% > 1.8× 20d avg → HIGH_VOL
_NEWS_VOL_MULT = 3.0           # opening 15m volume > 3× avg → NEWS_RISK
_VWAP_TREND_THRESHOLD = 0.15  # % distance from VWAP to confirm trend alignment
_ORB_CONFIRM_BARS = 3          # bars after ORB window to confirm breakout holds


@dataclass
class MarketStateResult:
    state: str
    confidence: float               # 0.0–1.0
    reasons: list[str] = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)
    # Sub-scores that fed into the final state
    trend_up_score: float = 0.0
    trend_down_score: float = 0.0
    choppy_score: float = 0.0
    high_vol_score: float = 0.0
    news_risk_score: float = 0.0


def classify_market_state(
    df_5m: pd.DataFrame,
    df_spy_5m: pd.DataFrame | None = None,
) -> MarketStateResult:
    """
    Classify the current intraday market state from 5m bars.

    If df_spy_5m is provided it is used for macro context (SPY VWAP, ORB).
    If not, df_5m itself is used (suitable when the symbol IS SPY).
    """
    ref = df_spy_5m if df_spy_5m is not None and not df_spy_5m.empty else df_5m
    ref = _today_bars(ref)
    df = _today_bars(df_5m)

    if ref.empty or len(ref) < 6:
        return MarketStateResult(state=UNKNOWN, confidence=0.0, reasons=["Insufficient data"])

    reasons: list[str] = []
    indicators: dict[str, Any] = {}

    # ── Core indicators ───────────────────────────────────────────────────────
    ref = ref.copy()
    ref["vwap"] = compute_vwap(ref)
    atr_ind = tav.AverageTrueRange(ref["High"], ref["Low"], ref["Close"], window=14)
    ref["atr"] = atr_ind.average_true_range()
    ref["rsi"] = tam.RSIIndicator(ref["Close"], window=14).rsi()

    current_close = float(ref["Close"].iloc[-1])
    current_vwap = float(ref["vwap"].iloc[-1])
    current_atr = float(ref["atr"].iloc[-1]) if not pd.isna(ref["atr"].iloc[-1]) else 0.0
    current_rsi = float(ref["rsi"].iloc[-1]) if not pd.isna(ref["rsi"].iloc[-1]) else 50.0

    vwap_dist_pct = (current_close - current_vwap) / current_vwap * 100
    atr_pct = current_atr / current_close * 100 if current_close > 0 else 0.0

    # 20-bar rolling ATR average for comparison
    atr_20_avg = float(ref["atr"].rolling(20).mean().iloc[-1]) if len(ref) >= 20 else current_atr
    atr_ratio = current_atr / atr_20_avg if atr_20_avg > 0 else 1.0

    indicators["vwap"] = round(current_vwap, 4)
    indicators["vwap_dist_pct"] = round(vwap_dist_pct, 3)
    indicators["atr_pct"] = round(atr_pct, 4)
    indicators["atr_ratio"] = round(atr_ratio, 2)
    indicators["rsi"] = round(current_rsi, 2)
    indicators["current_close"] = round(current_close, 4)

    # ── Opening range (first 15 min = 3 bars) ────────────────────────────────
    orb_df = ref.iloc[:3]
    orb_high = float(orb_df["High"].max())
    orb_low = float(orb_df["Low"].min())
    orb_height = orb_high - orb_low
    orb_broken_up = current_close > orb_high
    orb_broken_down = current_close < orb_low
    orb_intact = not orb_broken_up and not orb_broken_down

    indicators["orb_high"] = round(orb_high, 4)
    indicators["orb_low"] = round(orb_low, 4)
    indicators["orb_height"] = round(orb_height, 4)
    indicators["orb_broken_up"] = orb_broken_up
    indicators["orb_broken_down"] = orb_broken_down

    # ── Volume analysis ───────────────────────────────────────────────────────
    vol_avg_20 = float(ref["Volume"].rolling(20).mean().iloc[-1]) if len(ref) >= 20 else float(ref["Volume"].mean())
    opening_vol = float(ref.iloc[:3]["Volume"].sum())
    typical_opening_vol = vol_avg_20 * 3
    opening_vol_ratio = opening_vol / typical_opening_vol if typical_opening_vol > 0 else 1.0
    current_vol_ratio = float(ref["Volume"].iloc[-1]) / vol_avg_20 if vol_avg_20 > 0 else 1.0

    indicators["opening_vol_ratio"] = round(opening_vol_ratio, 2)
    indicators["current_vol_ratio"] = round(current_vol_ratio, 2)

    # ── Price structure: % of bars above VWAP ─────────────────────────────────
    recent_bars = ref.iloc[-min(12, len(ref)):]  # last 12 bars (~1 hour)
    pct_above_vwap = (recent_bars["Close"] > recent_bars["vwap"]).mean()
    indicators["pct_bars_above_vwap"] = round(float(pct_above_vwap), 2)

    # ── Higher highs / lower lows structure ──────────────────────────────────
    if len(ref) >= 6:
        closes = ref["Close"].iloc[-6:].values
        hh_hl = all(closes[i] > closes[i - 2] for i in range(2, 6, 2))   # higher highs
        lh_ll = all(closes[i] < closes[i - 2] for i in range(2, 6, 2))   # lower lows
    else:
        hh_hl = lh_ll = False
    indicators["higher_highs"] = hh_hl
    indicators["lower_lows"] = lh_ll

    # ─────────────────────────────────────────────────────────────────────────
    # Score each state (0.0–1.0)
    # ─────────────────────────────────────────────────────────────────────────

    # NEWS_RISK — check first: overrides everything
    news_score = 0.0
    if opening_vol_ratio > _NEWS_VOL_MULT:
        news_score += 0.6
        reasons.append(f"Opening volume {opening_vol_ratio:.1f}× avg — possible news catalyst.")
    if atr_ratio > 2.5:
        news_score += 0.3
        reasons.append(f"ATR {atr_ratio:.1f}× above 20-bar avg — extreme volatility.")
    if current_vol_ratio > 4.0:
        news_score += 0.2
        reasons.append(f"Current bar volume {current_vol_ratio:.1f}× avg.")
    news_score = min(news_score, 1.0)

    # HIGH_VOL — elevated but not necessarily news
    high_vol_score = 0.0
    if atr_ratio > _ATR_HIGH_VOL_MULT:
        high_vol_score += 0.5
        reasons.append(f"ATR {atr_ratio:.1f}× above normal — elevated volatility session.")
    if opening_vol_ratio > 1.8:
        high_vol_score += 0.3
    if atr_pct > 0.3:   # SPY moves >0.3% per 5m bar on avg
        high_vol_score += 0.2
    high_vol_score = min(high_vol_score, 1.0)

    # TREND_UP
    trend_up_score = 0.0
    if vwap_dist_pct > _VWAP_TREND_THRESHOLD:
        trend_up_score += 0.3
        reasons.append(f"Price {vwap_dist_pct:+.2f}% above VWAP — bullish.")
    if orb_broken_up:
        trend_up_score += 0.25
        reasons.append("ORB high broken — bullish breakout.")
    if pct_above_vwap > 0.7:
        trend_up_score += 0.2
        reasons.append(f"{pct_above_vwap:.0%} of recent bars above VWAP.")
    if hh_hl:
        trend_up_score += 0.15
        reasons.append("Higher highs and higher lows structure.")
    if current_rsi > 55:
        trend_up_score += 0.1
    trend_up_score = min(trend_up_score, 1.0)

    # TREND_DOWN
    trend_down_score = 0.0
    if vwap_dist_pct < -_VWAP_TREND_THRESHOLD:
        trend_down_score += 0.3
        reasons.append(f"Price {vwap_dist_pct:+.2f}% below VWAP — bearish.")
    if orb_broken_down:
        trend_down_score += 0.25
        reasons.append("ORB low broken — bearish breakdown.")
    if pct_above_vwap < 0.3:
        trend_down_score += 0.2
        reasons.append(f"Only {pct_above_vwap:.0%} of recent bars above VWAP.")
    if lh_ll:
        trend_down_score += 0.15
        reasons.append("Lower highs and lower lows structure.")
    if current_rsi < 45:
        trend_down_score += 0.1
    trend_down_score = min(trend_down_score, 1.0)

    # CHOPPY — price hugging VWAP, narrow range, no directional conviction
    choppy_score = 0.0
    if abs(vwap_dist_pct) < _VWAP_TREND_THRESHOLD:
        choppy_score += 0.35
        reasons.append(f"Price within {_VWAP_TREND_THRESHOLD}% of VWAP — no directional bias.")
    if orb_intact:
        choppy_score += 0.25
        reasons.append("ORB intact — no breakout or breakdown yet.")
    if 0.35 < pct_above_vwap < 0.65:
        choppy_score += 0.2
        reasons.append("Price oscillating around VWAP.")
    if atr_ratio < 0.8:
        choppy_score += 0.2
        reasons.append(f"ATR {atr_ratio:.1f}× below average — compressed, low momentum.")
    choppy_score = min(choppy_score, 1.0)

    # ── Pick winner ───────────────────────────────────────────────────────────
    # NEWS_RISK and HIGH_VOL have priority gates
    if news_score >= 0.6:
        state = NEWS_RISK
        confidence = news_score
    elif high_vol_score >= 0.5 and high_vol_score > max(trend_up_score, trend_down_score):
        state = HIGH_VOL
        confidence = high_vol_score
    else:
        scores = {
            TREND_UP: trend_up_score,
            TREND_DOWN: trend_down_score,
            CHOPPY: choppy_score,
        }
        state = max(scores, key=scores.get)
        confidence = scores[state]

        # Require meaningful confidence — fall back to CHOPPY if unclear
        if confidence < 0.3:
            state = CHOPPY
            confidence = max(choppy_score, 0.3)
            reasons.append("No dominant state — defaulting to CHOPPY.")

    # Deduplicate reasons
    seen: set[str] = set()
    unique_reasons = [r for r in reasons if not (r in seen or seen.add(r))]

    return MarketStateResult(
        state=state,
        confidence=round(confidence, 3),
        reasons=unique_reasons,
        indicators=indicators,
        trend_up_score=round(trend_up_score, 3),
        trend_down_score=round(trend_down_score, 3),
        choppy_score=round(choppy_score, 3),
        high_vol_score=round(high_vol_score, 3),
        news_risk_score=round(news_score, 3),
    )


def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    df = df.copy()
    df.index = idx
    today = idx[-1].date()
    return df[idx.date == today]
