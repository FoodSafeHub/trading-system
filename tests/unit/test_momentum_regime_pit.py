"""
Point-in-time momentum regime tests.

Pin the contract:
    * get_momentum_regime_at() uses only data <= as_of_date (no lookahead).
    * Synthetic bear+hot-VIX history classifies as BEAR_MOMENTUM at the right
      historical date.
    * The strategy gate (_momentum_snapshot) honours an injected snapshot and
      does NOT call the live helper when one is provided.
    * Asking for the snapshot at an early date with < 200 bars degrades to
      NO_TRADE rather than blowing up.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.market_regime_advanced import (
    MomentumRegime,
    get_momentum_regime_at,
    RegimeSnapshot,
)
from app.services.strategy.perplexity.momentum_strategies import (
    _momentum_snapshot,
    _regime_allows_long,
    _regime_allows_short,
)


def _series(n: int, builder, start: str = "2020-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([float(builder(i)) for i in range(n)], index=idx)


# ── get_momentum_regime_at() ────────────────────────────────────────────────


class TestPointInTimeRegime:
    def test_bear_momentum_on_synthetic_decline_plus_hot_vix(self):
        # 260 bars sliding from 100 down to ~61: close < SMA50 < SMA200,
        # VIX held at 35 (> us threshold 25). Must be BEAR_MOMENTUM.
        idx_close = _series(260, lambda i: 100.0 - 0.15 * i)
        vix = _series(260, lambda i: 35.0)
        snap = get_momentum_regime_at(
            as_of_date=idx_close.index[-1],
            index_close=idx_close, vix_close=vix, market="us",
        )
        assert snap.regime == MomentumRegime.BEAR_MOMENTUM, snap.regime
        assert snap.allows_short is True

    def test_bull_momentum_on_steady_uptrend_with_calm_vix(self):
        idx_close = _series(260, lambda i: 100.0 + 0.2 * i)
        vix = _series(260, lambda i: 15.0)
        snap = get_momentum_regime_at(
            as_of_date=idx_close.index[-1],
            index_close=idx_close, vix_close=vix, market="us",
        )
        assert snap.regime == MomentumRegime.BULL_MOMENTUM
        assert snap.allows_long is True
        assert snap.allows_short is False

    def test_no_trade_when_history_under_200_bars(self):
        # Only 100 bars of history → can't compute SMA200 → NO_TRADE.
        idx_close = _series(100, lambda i: 100.0 + 0.1 * i)
        snap = get_momentum_regime_at(
            as_of_date=idx_close.index[-1],
            index_close=idx_close, vix_close=None, market="us",
        )
        assert snap.regime == MomentumRegime.NO_TRADE
        assert not snap.allows_long and not snap.allows_short

    def test_slice_respects_as_of_date(self):
        # 260 bars: first 200 are bear, last 60 are bull. Asking at the
        # midpoint (bar 200) must still see BEAR — no peeking at the future
        # 60 bull bars.
        idx_close = _series(
            260,
            lambda i: 100.0 - 0.15 * i if i < 200 else 70.0 + 0.5 * (i - 200),
        )
        vix = _series(260, lambda i: 35.0 if i < 200 else 15.0)
        as_of = idx_close.index[199]
        snap = get_momentum_regime_at(
            as_of_date=as_of,
            index_close=idx_close, vix_close=vix, market="us",
        )
        # At the midpoint we should be in BEAR_MOMENTUM (the late bull data
        # must not bleed into the as-of snapshot).
        assert snap.regime == MomentumRegime.BEAR_MOMENTUM
        # Sanity: same series, asked at the end → ought to flip away from BEAR.
        snap_end = get_momentum_regime_at(
            as_of_date=idx_close.index[-1],
            index_close=idx_close, vix_close=vix, market="us",
        )
        assert snap_end.regime != MomentumRegime.BEAR_MOMENTUM


# ── _momentum_snapshot honours injection (no live leak) ─────────────────────


class TestStrategyGateNoLiveLeak:
    def test_injected_snapshot_is_used_verbatim(self):
        """When the engine injects a snapshot, the gate must NOT call the
        live helper. We verify by checking identity."""
        injected = RegimeSnapshot(
            regime=MomentumRegime.BEAR_MOMENTUM,
            spy_close=350.0, spy_sma50=400.0, spy_sma200=440.0,
            vix=30.0, breadth_pct=40.0, reasons=["synthetic bear"],
        )
        out = _momentum_snapshot("AAPL", injected=injected)
        assert out is injected, "injected snapshot must be returned unchanged"
        assert out.allows_short is True

    def test_no_injection_falls_back_to_live_path(self, monkeypatch):
        """Without injection, the function falls back to the live helper.
        We stub get_momentum_regime so the test is deterministic offline."""
        sentinel = RegimeSnapshot(
            regime=MomentumRegime.BULL_MOMENTUM,
            spy_close=500.0, spy_sma50=480.0, spy_sma200=460.0,
            vix=15.0, breadth_pct=70.0, reasons=["sentinel live"],
        )
        from app.services.strategy.perplexity import momentum_strategies as ms

        def _stub(market="us"):
            return sentinel
        monkeypatch.setattr(ms, "get_momentum_regime", _stub)
        out = _momentum_snapshot("AAPL", injected=None)
        assert out is sentinel

    def test_bearish_snapshot_unblocks_short_gate(self):
        """The whole point of the fix: when the historical regime is bearish,
        _regime_allows_short returns True (so a strategy's bearish-pattern
        branch can fire). Previously every backtest bar saw today's live
        regime, and short was usually disabled."""
        bearish = RegimeSnapshot(
            regime=MomentumRegime.BEAR_MOMENTUM,
            spy_close=350.0, spy_sma50=400.0, spy_sma200=440.0,
            vix=30.0, breadth_pct=40.0, reasons=["bear"],
        )
        snap = _momentum_snapshot("AAPL", injected=bearish)
        assert _regime_allows_short(snap) is True
        assert _regime_allows_long(snap) is False
