"""
Tests for the bucket-aware risk-based sizing wired into the day-trading
backtest runner.

Pins:
    * tighter stop → larger size (per fixed account risk budget)
    * wider stop  → smaller size
    * zero / negative stop distance → 0 (skip)
    * bucket changes the risk budget (NSE_MID_CAP gets a larger budget than US_ETF)
    * notional cap clamps unreasonably-large risk sizes
"""
from __future__ import annotations

from app.services.strategy.daytrading.runner import _risk_based_size


# ── stop distance drives size ────────────────────────────────────────────────


def test_tighter_stop_yields_larger_size_than_wider_stop():
    equity = 10_000.0
    bucket = "US_LARGE_CAP"
    tight = _risk_based_size(equity, bucket, entry=100.0, stop=99.0)   # $1 risk
    wide  = _risk_based_size(equity, bucket, entry=100.0, stop=95.0)   # $5 risk
    assert tight > wide
    # Sanity: $50 budget / $1 = 50 shares; / $5 = 10 shares
    assert tight == 50
    assert wide == 10


def test_zero_risk_stop_returns_zero():
    # entry == stop (zero risk) must be skipped, not produce a divide-by-zero.
    assert _risk_based_size(10_000.0, "US_LARGE_CAP", entry=100.0, stop=100.0) == 0


def test_negative_entry_or_equity_returns_zero():
    assert _risk_based_size(0.0,     "US_LARGE_CAP", entry=100.0, stop=99.0) == 0
    assert _risk_based_size(10_000., "US_LARGE_CAP", entry=0.0,   stop=99.0) == 0


# ── bucket differentiation ───────────────────────────────────────────────────


def test_us_mid_small_gets_larger_budget_than_us_etf():
    # Same trade geometry; only the bucket changes. risk_per_trade is
    # 0.40% for US_ETF vs 0.60% for US_MID_SMALL → mid_small gets more shares.
    equity = 100_000.0
    etf   = _risk_based_size(equity, "US_ETF",       entry=400.0, stop=398.0)
    mid_s = _risk_based_size(equity, "US_MID_SMALL", entry=400.0, stop=398.0)
    assert mid_s > etf


def test_nse_buckets_vs_us_buckets():
    # Use a wider stop so the risk budget — not the notional cap — is binding.
    # Cap at entry 2500 / equity 100k = 38 shares; with $50 stop distance, risk
    # math gives NSE_LARGE 0.5%/50 = 10 shares, NSE_MID 0.6%/50 = 12 shares.
    equity = 100_000.0
    nse_large = _risk_based_size(equity, "NSE_LARGE_CAP", entry=2_500.0, stop=2_450.0)
    nse_mid   = _risk_based_size(equity, "NSE_MID_CAP",   entry=2_500.0, stop=2_450.0)
    # NSE_MID_CAP has a larger risk budget than NSE_LARGE_CAP.
    assert nse_mid > nse_large


# ── notional cap ─────────────────────────────────────────────────────────────


def test_notional_cap_clamps_when_stop_is_unusually_tight():
    # A 1-cent stop on a $5 stock would compute thousands of shares by risk;
    # the notional cap (default 0.95) prevents that from exceeding equity.
    equity = 10_000.0
    n = _risk_based_size(equity, "US_MID_SMALL", entry=5.0, stop=4.99,
                         notional_cap_pct=0.95)
    # Cap = (10000 * 0.95) / 5 = 1900 shares
    assert n == 1900


def test_notional_cap_does_not_inflate_risk_sized_result():
    # If the risk-sized result is well under the cap, the cap must not change it.
    equity = 10_000.0
    n = _risk_based_size(equity, "US_LARGE_CAP", entry=100.0, stop=98.0,
                         notional_cap_pct=0.95)
    # $50 / $2 = 25 shares (well under cap of 95 shares).
    assert n == 25
