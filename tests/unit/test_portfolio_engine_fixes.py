"""
Targeted tests for the two portfolio_engine fixes:

1) Honor strategy.config["max_hold_bars"] — time-exit closes a position when
   its budget elapses, regardless of strategy SELL.
2) CostModel wiring — passing a non-zero CostModel produces strictly worse
   P&L than `cost_model=None` on a winning scenario; ZERO CostModel is
   identity with None.

Synthetic OHLCV + a minimal PerplexityStrategy so the tests don't depend on
real markets.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.costs import CostModel
from app.services.backtest.portfolio_engine import run_portfolio_backtest
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _padded(post_bars, warmup_close=100.0, warmup_n=220):
    """Pad with flat warmup bars so the engine's 210-bar lookback is satisfied."""
    flat = [(warmup_close,) * 4] * warmup_n
    rows = flat + post_bars
    idx = pd.bdate_range("2023-01-02", periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1_000_000
    return df


class _OneShotBuy(PerplexityStrategy):
    """Emits BUY on bar at `entry_idx`, HOLD otherwise.

    The portfolio engine calls strategy.run(sym, df_slice) where df_slice has
    length == iloc (engine line 213). So firing on len(df) == entry_idx matches
    bar `entry_idx`. Stops/targets are wide enough not to trip the no-cost
    tests' time-exit assertion.
    """
    name = "portfolio_one_shot_buy"

    def __init__(self, entry_idx, stop, target, max_hold_bars=5):
        self._entry_idx = entry_idx
        self._stop = stop
        self._target = target
        self.config = {"max_hold_bars": max_hold_bars}

    def run(self, symbol, df, **kw):
        if len(df) == self._entry_idx:
            price = float(df["Close"].iloc[-1])
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=price, stop_price=self._stop,
                target_price=self._target, confidence=1.0,
            )
        return self._hold(symbol)


def _patch_data(monkeypatch, dfs: dict):
    """Make get_ohlcv return the test data instead of fetching."""
    from app.services.backtest import portfolio_engine as pe

    def _stub(symbol, period=None, interval=None):
        return dfs.get(symbol, pd.DataFrame())

    monkeypatch.setattr(pe, "get_ohlcv", _stub)


# ── 1. max_hold_bars ────────────────────────────────────────────────────────


class TestPortfolioMaxHoldBars:
    def test_time_exit_fires_when_budget_exceeded(self, monkeypatch):
        # 220 flat warmup bars at 100; then a quiet rise that never touches
        # the wide stop (95) or wide target (120). With max_hold_bars=5,
        # the position must time-exit on bar 225.
        df = _padded(
            post_bars=[(101.0, 101.5, 100.5, 101.0)] * 12,
            warmup_close=100.0, warmup_n=220,
        )
        _patch_data(monkeypatch, {"TEST": df})
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=120.0, max_hold_bars=5)
        r = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
        )
        sells = [t for t in r.trades if "SELL" in t["side"]]
        assert sells, f"expected a sell leg; got {r.trades}"
        assert "time" in sells[0]["side"].lower(), (
            f"expected SELL (time); got side={sells[0]['side']}"
        )

    def test_no_time_exit_when_target_hits_first(self, monkeypatch):
        # Target hits on the next bar — time-exit must NOT fire.
        df = _padded(
            post_bars=[
                (101.0, 105.0, 100.5, 104.5),    # entry day
                (104.5, 121.0, 104.0, 120.5),    # target 120 hits
                (120.5, 121.0, 120.0, 120.0),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        _patch_data(monkeypatch, {"TEST": df})
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=120.0, max_hold_bars=5)
        r = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
        )
        sells = [t for t in r.trades if "SELL" in t["side"]]
        assert sells and "target" in sells[0]["side"].lower()


# ── 2. CostModel ─────────────────────────────────────────────────────────────


class TestPortfolioCostModel:
    def test_costed_pnl_strictly_worse_on_winner(self, monkeypatch):
        df = _padded(
            post_bars=[
                (101.0, 102.0, 100.5, 101.0),
                (101.0, 106.0, 100.8, 105.5),    # target 105 hits
                (105.5, 105.6, 105.0, 105.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        _patch_data(monkeypatch, {"TEST": df})
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=105.0, max_hold_bars=10)

        clean = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
            cost_model=None,
        )
        costed = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
            cost_model=CostModel(slippage_bps=10.0, half_spread_bps=5.0,
                                 commission_per_share=0.005, taxes_bps=5.0),
        )
        assert clean.total_pnl > 0
        assert costed.total_pnl < clean.total_pnl, (
            f"costed ({costed.total_pnl}) must be strictly less than "
            f"clean ({clean.total_pnl}) on a winner"
        )

    def test_zero_cost_model_is_identity_with_none(self, monkeypatch):
        df = _padded(
            post_bars=[
                (101.0, 102.0, 100.5, 101.0),
                (101.0, 106.0, 100.8, 105.5),
                (105.5, 105.6, 105.0, 105.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        _patch_data(monkeypatch, {"TEST": df})
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=105.0, max_hold_bars=10)
        r_none = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
            cost_model=None,
        )
        r_zero = run_portfolio_backtest(
            strat, symbols=["TEST"], period="2y",
            initial_capital=100_000.0, position_pct=0.50, max_open_positions=1,
            cost_model=CostModel(),
        )
        assert r_none.total_pnl == r_zero.total_pnl
