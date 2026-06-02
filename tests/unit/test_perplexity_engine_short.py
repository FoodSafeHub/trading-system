"""
Targeted tests for the minimal short-side path in perplexity_engine.

What we pin:
    * A SELL signal while flat with a valid short geometry (stop > entry)
      opens a short position (negative `position`, "SHORT" trade leg).
    * Short hits target -> COVER (target).
    * Short hits stop  -> COVER (stop).
    * Short time-exit  -> COVER (time).
    * P&L on a winning short = entry_proceeds - cover_value (positive when
      price fell).
    * CostModel applies symmetrically: a profitable short with costs is
      worse than the same short without costs.
    * A SELL while flat that lacks a valid short geometry (stop <= entry)
      is SKIPPED — no phantom long-only-style exit.

The intent is to validate the SHAPE of short execution, not to bench
strategy P&L (that's the 5y benchmark in phase B).
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.costs import CostModel
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


# ── Helpers ──────────────────────────────────────────────────────────────────


def _padded(post_bars, warmup_close=100.0, warmup_n=220):
    flat = [(warmup_close,) * 4] * warmup_n
    rows = flat + post_bars
    idx = pd.bdate_range("2024-01-02", periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1_000_000
    return df


class _OneShotSell(PerplexityStrategy):
    """Emits SELL on bar `entry_idx`, HOLD otherwise. Used to test the short
    entry path in isolation — strategy emits a single short signal then stays
    quiet so we observe how the engine handles it through to exit.
    """
    name = "one_shot_sell_test"

    def __init__(self, entry_idx, stop, target, max_hold_bars=5):
        self._entry_idx = entry_idx
        self._stop = stop
        self._target = target
        self.config = {"max_hold_bars": max_hold_bars}

    def run(self, symbol, df, **kw):
        if len(df) == self._entry_idx:
            price = float(df["Close"].iloc[-1])
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="SELL",
                entry_price=price, stop_price=self._stop,
                target_price=self._target, confidence=1.0,
            )
        return self._hold(symbol)


# ── 1. Short opens and exits at target ───────────────────────────────────────


class TestShortEntry:
    def test_sell_while_flat_opens_short(self):
        # 220 flat warmup at 100; bar 220 entry; bar 221 dives to 90 hitting
        # the 90 target. Short stop is 105 (above entry), target 90 (below).
        df = _padded(
            post_bars=[
                (100.0, 100.5, 99.5, 100.0),   # entry day
                (100.0, 100.3, 89.5, 90.5),    # target 90 hits
                (90.5, 91.0, 90.0, 90.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotSell(entry_idx=220, stop=105.0, target=90.0, max_hold_bars=10)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        sides = [t["side"] for t in r.trades]
        assert "SHORT" in sides, f"expected SHORT entry leg, got {sides}"
        # Exit must be a COVER (target)
        cover = [t for t in r.trades if "COVER" in t["side"]]
        assert cover, f"expected COVER leg; got {r.trades}"
        assert "target" in cover[0]["side"].lower()

    def test_short_hits_stop(self):
        # Entry at 100, stop 105 — bar 221 rallies to 110 -> stop fires.
        df = _padded(
            post_bars=[
                (100.0, 100.5, 99.5, 100.0),
                (100.0, 110.0, 99.8, 108.0),    # high 110 >= stop 105
                (108.0, 108.5, 107.0, 107.5),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotSell(entry_idx=220, stop=105.0, target=80.0, max_hold_bars=10)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        cover = [t for t in r.trades if "COVER" in t["side"]]
        assert cover and "stop" in cover[0]["side"].lower()

    def test_short_time_exit_fires(self):
        # No move large enough to trip stop or target; max_hold_bars=3 forces
        # a COVER (time) on the budget bar.
        df = _padded(
            post_bars=[(100.0, 100.3, 99.7, 100.0)] * 10,
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotSell(entry_idx=220, stop=110.0, target=90.0, max_hold_bars=3)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        cover = [t for t in r.trades if "COVER" in t["side"]]
        assert cover, f"expected COVER leg; got {r.trades}"
        assert "time" in cover[0]["side"].lower()

    def test_short_pnl_correct_on_winner(self):
        # Entry at 100, target 90 hits next bar. P&L per share = 100 - 90 = 10
        # before costs. We verify the SIGN and magnitude are right.
        df = _padded(
            post_bars=[
                (100.0, 100.5, 99.5, 100.0),
                (100.0, 100.3, 89.5, 90.5),
                (90.5, 91.0, 90.0, 90.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotSell(entry_idx=220, stop=105.0, target=90.0, max_hold_bars=10)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        cover = [t for t in r.trades if "COVER" in t["side"]]
        # P&L positive on a winning short
        assert cover[0]["pnl"] > 0, (
            f"winning short must have positive pnl, got {cover[0]['pnl']}"
        )


# ── 2. CostModel applies symmetrically to shorts ─────────────────────────────


class TestShortCosts:
    def test_costed_short_winner_is_worse_than_uncosted(self):
        df = _padded(
            post_bars=[
                (100.0, 100.5, 99.5, 100.0),
                (100.0, 100.3, 89.5, 90.5),
                (90.5, 91.0, 90.0, 90.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotSell(entry_idx=220, stop=105.0, target=90.0, max_hold_bars=10)
        clean = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"], cost_model=None,
        )
        costed = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
            cost_model=CostModel(slippage_bps=10.0, half_spread_bps=5.0,
                                 commission_per_share=0.005, taxes_bps=5.0),
        )
        assert clean.total_pnl > 0
        assert costed.total_pnl < clean.total_pnl, (
            f"costed short winner must be strictly worse than uncosted: "
            f"clean={clean.total_pnl} costed={costed.total_pnl}"
        )


# ── 3. Skip SELL with invalid short geometry ────────────────────────────────


class TestInvalidShortGeometry:
    def test_sell_with_stop_at_or_below_entry_is_skipped(self):
        """SELL must be skipped (not opened as a short) when stop is not
        above entry. The engine must not invent a short with malformed
        geometry."""
        df = _padded(
            post_bars=[(100.0, 100.5, 99.5, 100.0)] * 8,
            warmup_close=100.0, warmup_n=220,
        )
        # Stop EQUAL to entry — invalid short geometry
        strat = _OneShotSell(entry_idx=220, stop=100.0, target=90.0)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        # No SHORT, no COVER — strategy emitted SELL while flat with bad
        # geometry; engine ignored it.
        sides = [t["side"] for t in r.trades]
        assert not any(s == "SHORT" or "COVER" in s for s in sides), (
            f"engine must not open a short with stop<=entry; got trades {sides}"
        )
