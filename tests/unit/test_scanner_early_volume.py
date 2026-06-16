"""
Tests for the scanner's early-activity volume fallback and bucket classification.

Root problem these guard against: data providers frequently return pre-market
bars with zero volume (or none at all). When the pre-market window summed to 0,
rel_vol was 0 for every symbol, which (a) zeroed the 25% rel-vol score weight and
(b) collapsed every symbol into the VWAP bucket because all the breakout-bucket
tags required rel_vol above a floor. The fix:
  - _early_activity_volume falls back to opening-range (first 30 min RTH) volume,
  - score renormalizes when rel_vol is unavailable,
  - bucket tagging uses gap+ATR when rel_vol is unknown.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.services.strategy.daytrading.market_open import ET
from app.services.strategy.daytrading.scanners.daytrading_scanner import (
    DayTradingScanner,
    DayTradingScannerConfig,
    SymbolScanMetrics,
    _early_activity_volume,
)


def _bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """rows = [(ET HH:MM, volume), ...] on a fixed weekday -> 1m OHLCV frame."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-10 {hm}", tz=ET) for hm, _ in rows]
    )
    vol = [v for _, v in rows]
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": vol},
        index=idx,
    )


def test_uses_premarket_volume_when_present():
    df = _bars([("08:00", 1000), ("09:00", 2000), ("09:45", 9999)])
    # pre-market (04:00–09:30) = 3000; opening-range ignored because PM > 0
    assert _early_activity_volume(df) == 3000.0


def test_falls_back_to_opening_range_when_premarket_zero():
    df = _bars([("08:00", 0), ("09:00", 0), ("09:35", 5000), ("09:55", 1000), ("10:30", 9999)])
    # PM sums to 0 -> fall back to 09:30–10:00 window = 6000 (10:30 excluded)
    assert _early_activity_volume(df) == 6000.0


def test_utc_indexed_frame_is_converted_before_masking():
    # A UTC-indexed frame: 13:30–14:00 UTC == 09:30–10:00 ET (opening range).
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-06-10 13:35", tz="UTC"),
        pd.Timestamp("2026-06-10 13:50", tz="UTC"),
    ])
    df = pd.DataFrame({"Volume": [4000, 1000]}, index=idx)
    # No pre-market bars -> opening-range fallback should still find these.
    assert _early_activity_volume(df) == 5000.0


def test_empty_or_missing_volume_returns_zero():
    assert _early_activity_volume(pd.DataFrame()) == 0.0
    assert _early_activity_volume(None) == 0.0


def _metrics(**kw) -> SymbolScanMetrics:
    base = dict(
        symbol="X", last_price=20.0, avg_daily_volume_30d=5_000_000,
        atr_14=0.6, atr_pct=3.0, premarket_volume=0.0, premarket_rel_vol=0.0,
        premarket_gap_pct=0.0, gap_direction="flat", gap_size="none",
        has_catalyst=False, catalyst_tags=[],
    )
    base.update(kw)
    return SymbolScanMetrics(**base)


def test_large_gap_classifies_as_gap_even_when_relvol_unknown():
    s = DayTradingScanner(config=DayTradingScannerConfig(), universe=["X"])
    m = _metrics(premarket_gap_pct=4.0, gap_size="large", atr_pct=4.0, premarket_rel_vol=0.0)
    r = s.score_symbol(m)
    assert "gap_and_go" in r.tags
    assert r.recommended_strategy_bucket == "gap"


def test_relvol_available_keeps_strict_gating():
    # Small gap + LOW (but present) rel_vol must NOT be promoted to gap_and_go.
    s = DayTradingScanner(config=DayTradingScannerConfig(), universe=["X"])
    m = _metrics(premarket_gap_pct=1.0, gap_size="small", premarket_rel_vol=0.01)
    r = s.score_symbol(m)
    assert "gap_and_go" not in r.tags
    assert "gap_fade_candidate" in r.tags


def test_score_renormalizes_when_relvol_unavailable():
    # Two identical metrics except rel_vol availability. The unavailable-rv symbol
    # should not be penalized by the full rel-vol weight (its score should be at
    # least as high as if rel_vol scored a low-but-present value).
    s = DayTradingScanner(config=DayTradingScannerConfig(), universe=["X"])
    unknown = s.score_symbol(_metrics(gap_size="medium", premarket_gap_pct=2.0, premarket_rel_vol=0.0))
    low_present = s.score_symbol(_metrics(gap_size="medium", premarket_gap_pct=2.0, premarket_rel_vol=0.001))
    assert unknown.score >= low_present.score
