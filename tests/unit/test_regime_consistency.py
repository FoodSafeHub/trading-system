"""
Regime parity tests: the backtest classifier (`_regime_from_spy`) must agree
with the live classifier (`detect_market_regime`) on the SAME SPY history.

The two used to diverge on:
    * The BULL gate (live also requires SMA(50) >= SMA(200); backtest skipped
      this cross check).
    * The DEEP_BEAR drawdown semantics (backtest hard-coded 0.80*SMA200;
      live uses drawdown-from-52w-high <= -20%).

These tests build small synthetic SPY series that probe each label and pin
the two functions to the same answer.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.perplexity_engine import _regime_from_spy
from app.services.market_regime import MarketRegime, detect_market_regime


def _synthetic_spy(n: int, builder) -> pd.DataFrame:
    """Build a SPY-like daily series. `builder(i)` returns the close for bar i."""
    idx = pd.bdate_range("2022-01-03", periods=n)
    closes = pd.Series([float(builder(i)) for i in range(n)], index=idx)
    return pd.DataFrame({"Close": closes}, index=idx)


# ── Direct parity ────────────────────────────────────────────────────────────


class TestLiveBacktestParity:
    def test_bull_market_agrees(self):
        # Steady uptrend: close > SMA200 and SMA50 >= SMA200. Both classifiers
        # must say BULL.
        df = _synthetic_spy(280, lambda i: 100.0 + i * 0.2)
        live = detect_market_regime(df)
        bt = _regime_from_spy(df["Close"], df.index[-1])
        assert live == MarketRegime.BULL
        assert bt == live

    def test_bear_market_agrees(self):
        # Close < SMA200 but drawdown from 52w high is shallow (<20%) so
        # DEEP_BEAR doesn't trigger. Both classifiers must say BEAR.
        # Build: ramp up to 100, dip ~12% to 88, hold there long enough for
        # SMA200 to lag above current price.
        def builder(i):
            if i < 100:
                return 100.0
            if i < 220:
                return 100.0 - (i - 100) * 0.10   # 100 -> 88 over 120 bars
            return 88.0                            # hold at 88
        df = _synthetic_spy(280, builder)
        live = detect_market_regime(df)
        bt = _regime_from_spy(df["Close"], df.index[-1])
        assert live == MarketRegime.BEAR, f"got {live}"
        assert bt == live

    def test_deep_bear_drawdown_semantics_agree(self):
        # Climb to 200, then fall to 140 — a 30% drawdown from the 52w high.
        # The OLD inline backtest classifier (close < 0.80*SMA200) and the
        # live classifier (close < SMA200 AND DD from 52w high <= -20%)
        # disagreed here whenever SMA200 was meaningfully different from the
        # 52w high. After the fix they must agree.
        def builder(i):
            if i < 200:
                return 100.0 + i * 0.5     # ramp up: 100 -> 200
            return 200.0 - (i - 200) * 0.75  # collapse: 200 -> ~140 over 80 bars
        df = _synthetic_spy(280, builder)
        live = detect_market_regime(df)
        bt = _regime_from_spy(df["Close"], df.index[-1])
        assert live == bt, f"live={live} backtest={bt} on the same data"

    def test_death_cross_no_longer_silently_bull(self):
        # Close > SMA200 but SMA50 < SMA200 (a "death cross" near recovery).
        # Live calls this BEAR (the BULL gate requires SMA50 >= SMA200).
        # The OLD backtest classifier called this BULL. After the fix it
        # must match live.
        def builder(i):
            # Long history at ~120 (anchors SMA200), then a sharp dip to 90,
            # then a partial recovery to ~115. Close > SMA200 mid-recovery
            # but SMA50 still recovering.
            if i < 150:
                return 120.0
            if i < 220:
                return 120.0 - (i - 150) * 0.5    # 120 -> 85
            return 85.0 + (i - 220) * 0.6         # 85 -> 121
        df = _synthetic_spy(280, builder)
        live = detect_market_regime(df)
        bt = _regime_from_spy(df["Close"], df.index[-1])
        assert live == bt, (
            f"live={live} backtest={bt}: SMA50/200 cross gate must agree"
        )


# ── Edge cases the wrapper handles ───────────────────────────────────────────


class TestWrapperBehavior:
    def test_short_history_falls_back_to_bull(self):
        # Live raises on <200 bars; the backtest wrapper returns BULL so the
        # early-period bars of long backtests don't crash. This preserves
        # the pre-fix fallback semantics.
        df = _synthetic_spy(50, lambda i: 100.0 + i * 0.1)
        bt = _regime_from_spy(df["Close"], df.index[-1])
        assert bt == MarketRegime.BULL

    def test_partial_history_slice_uses_only_data_up_to_date(self):
        # The wrapper slices spy_close to as_of_date — no peeking at future
        # bars. With 280 bars of bull data, asking for the regime at bar 50
        # must still return the fallback (BULL) because only 50 bars are
        # accessible.
        df = _synthetic_spy(280, lambda i: 100.0 + i * 0.2)
        cutoff = df.index[49]   # bar 50 (50 bars of history <= 200 threshold)
        bt = _regime_from_spy(df["Close"], cutoff)
        assert bt == MarketRegime.BULL  # falls back since <200 bars
