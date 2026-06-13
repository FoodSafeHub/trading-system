"""Tilt-math guards for the M1 portfolio analyzer (advisory only).

These pin the contribution-tilt behavior independent of live market data:
  - aggressive: only BUY-signal names are funded; weights sum to 1.0.
  - HOLD/SELL names get 0 dollars in aggressive mode.
  - data-failed holdings (error set) never receive tilt.
  - all-HOLD portfolio in aggressive mode funds nothing (no BUY → total 0).
"""
from __future__ import annotations

import pandas as pd

from app.services.m1.analyzer import (
    HoldingSignal, PieSlice, compute_tilt, _pie_conviction, _dip_metrics,
)


def _sig(symbol, direction, conviction=50.0, value=100.0, error=None,
         dip_score=0.0, laggard=False):
    return HoldingSignal(
        symbol=symbol, name=symbol, value=value, price=10.0,
        direction=direction, conviction=conviction,
        votes_buy=0, votes_sell=0, votes_hold=0, error=error,
        dip_score=dip_score, laggard=laggard,
    )


def test_aggressive_funds_only_buys_and_sums_to_one():
    sigs = [
        _sig("AAA", "BUY", conviction=80),
        _sig("BBB", "BUY", conviction=20),
        _sig("CCC", "HOLD"),
        _sig("DDD", "SELL"),
    ]
    compute_tilt(sigs, contribution=1000.0, mode="aggressive")

    by = {s.symbol: s for s in sigs}
    # Only BUYs funded
    assert by["CCC"].suggested_dollars == 0.0
    assert by["DDD"].suggested_dollars == 0.0
    assert by["AAA"].suggested_dollars > 0
    assert by["BBB"].suggested_dollars > 0
    # Higher conviction BUY gets more
    assert by["AAA"].suggested_dollars > by["BBB"].suggested_dollars
    # Weights of funded names sum to ~1.0
    assert abs(by["AAA"].tilt_weight + by["BBB"].tilt_weight - 1.0) < 1e-9
    # Dollars sum to the contribution (within rounding)
    assert abs(by["AAA"].suggested_dollars + by["BBB"].suggested_dollars - 1000.0) < 0.05


def test_errored_holding_never_funded():
    sigs = [
        _sig("AAA", "BUY", conviction=50),
        _sig("ZZZ", "BUY", conviction=99, error="no data"),
    ]
    compute_tilt(sigs, contribution=500.0, mode="aggressive")
    by = {s.symbol: s for s in sigs}
    assert by["ZZZ"].tilt_weight == 0.0
    assert by["ZZZ"].suggested_dollars == 0.0
    # The only usable BUY takes the whole contribution
    assert abs(by["AAA"].suggested_dollars - 500.0) < 0.05


def test_all_hold_aggressive_funds_nothing():
    sigs = [_sig("AAA", "HOLD"), _sig("BBB", "HOLD")]
    compute_tilt(sigs, contribution=1000.0, mode="aggressive")
    assert all(s.suggested_dollars == 0.0 for s in sigs)
    assert all(s.tilt_weight == 0.0 for s in sigs)


def test_gentle_mode_funds_holds_too():
    sigs = [_sig("AAA", "BUY"), _sig("BBB", "HOLD"), _sig("CCC", "SELL")]
    compute_tilt(sigs, contribution=300.0, mode="gentle")
    by = {s.symbol: s for s in sigs}
    # Gentle keeps everything in play (SELL down-weighted, not zeroed)
    assert by["AAA"].suggested_dollars > by["BBB"].suggested_dollars > by["CCC"].suggested_dollars > 0


def _slice(symbol, direction, conviction=50.0, value=100.0, error=None):
    return PieSlice(symbol=symbol, value=value, direction=direction,
                    conviction=conviction, error=error)


def test_pie_conviction_value_weighted_net_buy():
    # Big BUY slice + small SELL slice → positive, value-weighted toward the BUY.
    slices = [
        _slice("BIG", "BUY", conviction=80, value=900),
        _slice("SMALL", "SELL", conviction=80, value=100),
    ]
    conv = _pie_conviction(slices)
    # (900*80 - 100*80) / 1000 = 64
    assert abs(conv - 64.0) < 1e-6


def test_pie_conviction_all_sell_clamps_to_zero():
    slices = [_slice("A", "SELL", conviction=90, value=100),
              _slice("B", "SELL", conviction=70, value=100)]
    assert _pie_conviction(slices) == 0.0


def test_pie_conviction_ignores_errored_slices():
    slices = [_slice("OK", "BUY", conviction=50, value=100),
              _slice("BAD", "BUY", conviction=99, value=100, error="no data")]
    # Errored slice excluded → conviction == the one good BUY's conviction.
    assert abs(_pie_conviction(slices) - 50.0) < 1e-6


# ── Dip-timing (SIP) ──────────────────────────────────────────────────────────


def test_dip_score_higher_when_oversold_pullback_in_uptrend():
    # Long uptrend (above SMA200) that recently pulled back hard → high dip_score.
    up = pd.Series([100 + i for i in range(220)], dtype=float)          # steady uptrend
    pulled = pd.concat([up, pd.Series([320, 300, 280, 265, 255], dtype=float)],
                       ignore_index=True)  # sharp recent drop from the highs
    rsi, pct_high, above200, dip = _dip_metrics(pulled)
    assert above200 is True                 # still structurally up
    assert pct_high is not None and pct_high < 0   # below the trailing high
    assert dip > 40                          # registers as a dip-buy

    # A name making new highs (no pullback) should score low.
    _, _, _, dip_flat = _dip_metrics(up)
    assert dip_flat < dip


def test_dip_score_damped_below_sma200():
    # A steady downtrend (below its 200-day): oversold, but NOT a dip-buy.
    down = pd.Series([300 - i for i in range(220)], dtype=float)
    rsi, pct_high, above200, dip = _dip_metrics(down)
    assert above200 is False
    # The 0.35 damp factor keeps a falling knife from scoring like a real dip.
    assert dip < 50


def test_dip_tilt_skips_laggards_and_funds_dips():
    sigs = [
        _sig("DIP", "HOLD", dip_score=80, value=100),     # great dip, no signal yet
        _sig("BUY", "BUY", dip_score=30, value=100),      # trend buy
        _sig("LAG", "SELL", dip_score=90, laggard=True),  # oversold but broken down
    ]
    compute_tilt(sigs, contribution=600.0, mode="dip")
    by = {s.symbol: s for s in sigs}
    assert by["LAG"].suggested_dollars == 0.0   # laggard never funded
    assert by["DIP"].suggested_dollars > 0       # dip funded even without BUY
    assert by["BUY"].suggested_dollars > 0       # trend buy funded
    # Conservation across the two funded names.
    assert abs(by["DIP"].suggested_dollars + by["BUY"].suggested_dollars - 600.0) < 0.05
