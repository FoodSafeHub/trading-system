"""
Targeted tests for the three perplexity_engine fixes:

1) Honor strategy.config["max_hold_bars"] — a time-exit branch should fire
   when the budget is hit and stop/target/SELL have not triggered.
2) CostModel wiring — passing a non-zero CostModel produces strictly worse
   net P&L than the same scenario with cost_model=None, on a winner.
3) Same-bar trailing-stop ratchet — the trail uses the PRIOR bar's high,
   not the current bar's high. A bar that punches a new high then reverses
   into a deep stop must not exit at the "high-water minus 0.5R" stop.

These tests build synthetic OHLCV deterministically and ride a minimal
PerplexityStrategy subclass — no live data, no randomness.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.costs import CostModel
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy


# ── Synthetic fixtures ───────────────────────────────────────────────────────


def _bars(rows: list[tuple[float, float, float, float]],
          start: str = "2024-01-02") -> pd.DataFrame:
    """rows = list of (Open, High, Low, Close) per day. Volume is constant."""
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(
        rows, columns=["Open", "High", "Low", "Close"], index=idx,
    )
    df["Volume"] = 1_000_000
    return df


def _padded(post_bars: list[tuple[float, float, float, float]],
            warmup_close: float = 100.0,
            warmup_n: int = 220) -> pd.DataFrame:
    """Pad with `warmup_n` flat bars so engine's 210-bar lookback is satisfied."""
    flat = [(warmup_close, warmup_close, warmup_close, warmup_close)] * warmup_n
    return _bars(flat + post_bars)


class _OneShotBuy(PerplexityStrategy):
    """Emits BUY on bar at `entry_idx` (relative to df), HOLD otherwise.

    The fixed-stop/target are wide so they don't fire during the test windows
    — we want to study time exits and trailing dynamics in isolation.
    """
    name = "one_shot_buy_strategy"

    def __init__(self, entry_idx: int, stop: float, target: float,
                 max_hold_bars: int = 5):
        self._entry_idx = entry_idx
        self._stop = stop
        self._target = target
        # Engine reads this for time-exit budget
        self.config = {"max_hold_bars": max_hold_bars}

    def run(self, symbol, df, **kwargs):  # noqa: ANN001
        # The engine slices df_slice = df_full.iloc[:i] so the last bar in df
        # represents bar (i-1). We want to fire when the engine is about to
        # execute on bar `_entry_idx`, i.e. df_slice length == _entry_idx.
        if len(df) == self._entry_idx:
            price = float(df["Close"].iloc[-1])
            return PerplexitySignal(
                symbol=symbol, strategy_name=self.name, direction="BUY",
                entry_price=price, stop_price=self._stop,
                target_price=self._target, confidence=1.0,
            )
        return self._hold(symbol)


# ── 1. max_hold_bars time exit ───────────────────────────────────────────────


class TestMaxHoldBars:
    def test_time_exit_fires_when_budget_exceeded(self):
        # 220 flat warmup bars at 100, then 10 quiet bars at 101 — neither
        # the wide stop (95) nor the wide target (120) trips. With
        # max_hold_bars=5, the position must time-exit on bar 226 (entry on
        # 220, +5 holding bars).
        df = _padded(
            post_bars=[(101.0, 101.5, 100.5, 101.0)] * 10,
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=120.0, max_hold_bars=5)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],   # bull regime (SPY above its mean)
        )
        assert r.total_trades == 1
        # The exit trade should be the time-exit variant the engine emits
        sell_legs = [t for t in r.trades if "SELL" in t["side"]]
        assert sell_legs, f"expected a sell leg; got {r.trades}"
        assert "time" in sell_legs[0]["side"].lower(), (
            f"expected a time-exit leg; got side={sell_legs[0]['side']}"
        )

    def test_no_time_exit_when_target_hits_first(self):
        # Same setup, but bar 1 punches the target — the time-exit branch must
        # never fire because the target closes the position first.
        df = _padded(
            post_bars=[(101.0, 125.0, 100.5, 121.0)] + [(121.0, 121.5, 120.5, 121.0)] * 9,
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=120.0, max_hold_bars=5)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        sell_legs = [t for t in r.trades if "SELL" in t["side"]]
        assert sell_legs and "target" in sell_legs[0]["side"].lower()


# ── 2. CostModel wiring ──────────────────────────────────────────────────────


class TestCostModel:
    def test_costed_pnl_is_strictly_worse_than_uncosted_on_a_winner(self):
        # Entry on bar 220 around 101, target 105 hit a couple of bars later.
        # With cost_model=None we get a clean winner; with a non-trivial cost
        # model (slippage + commission) net P&L must be strictly less.
        df = _padded(
            post_bars=[
                (101.0, 102.0, 100.5, 101.0),   # entry day
                (101.0, 106.0, 100.8, 105.5),   # target 105 hits
                (105.5, 105.6, 105.0, 105.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=105.0, max_hold_bars=10)
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
            f"costed ({costed.total_pnl}) must be strictly less than "
            f"clean ({clean.total_pnl}) on a winning trade"
        )

    def test_zero_cost_model_is_identity_with_none(self):
        # ZERO CostModel must produce exactly the same P&L as cost_model=None.
        df = _padded(
            post_bars=[
                (101.0, 102.0, 100.5, 101.0),
                (101.0, 106.0, 100.8, 105.5),
                (105.5, 105.6, 105.0, 105.2),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=105.0, max_hold_bars=10)
        r_none = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"], cost_model=None,
        )
        r_zero = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"], cost_model=CostModel(),
        )
        assert r_none.total_pnl == r_zero.total_pnl


# ── 3. Same-bar trailing-stop lookahead removed ─────────────────────────────


class TestTrailingNoLookahead:
    def test_same_bar_trap_does_not_exit_on_trap_bar(self):
        """
        Scenario: the BAR right after entry is a "trap" that pokes a high
        well above 1R (entry 100, initial_risk 5 → trigger at 105) then
        dips to 96 within the same bar.

        Under the OLD code (`high_today = iloc[i]`), bar 1 would:
            - ratchet stop to 110 - 0.5*5 = 107.5 using the SAME bar's high
            - check intraday low (96) against the ratcheted 107.5 → exit
            - exit fill = 107.5, EXIT DATE = the trap bar
        Under the NEW code (`iloc[i-1]`), bar 1 sees the prior bar's high
        (100.5), no ratchet, original stop 95 holds; low 96 > 95 → no exit.

        The decisive assertion is the EXIT DATE: it must NOT be the trap
        bar's date. A later exit (e.g. trail catching up on bar 2) is the
        expected, semantically-correct trail behavior — what we forbid is
        the same-bar future read.
        """
        df = _padded(
            post_bars=[
                (100.0, 100.5, 99.5, 100.0),         # bar 0 = entry bar (post warmup)
                (100.0, 110.0, 96.0, 102.0),         # bar 1 = the trap
                (102.0, 102.5, 101.5, 102.0),        # bar 2 = flat
                (102.0, 102.5, 101.5, 102.0),        # bar 3
                (102.0, 102.5, 101.5, 102.0),        # bar 4
                (102.0, 102.5, 101.5, 102.0),        # bar 5 = time-exit budget end
                (102.0, 102.5, 101.5, 102.0),
            ],
            warmup_close=100.0, warmup_n=220,
        )
        strat = _OneShotBuy(entry_idx=220, stop=95.0, target=200.0, max_hold_bars=5)
        r = run_perplexity_backtest(
            strat, "TEST", period="2y", initial_capital=100_000.0,
            df_full=df, spy_close=df["Close"],
        )
        sell_legs = [t for t in r.trades if "SELL" in t["side"]]
        assert sell_legs, f"expected a sell leg; got {r.trades}"
        # The trap bar's date is the 222nd business day from 2024-01-02.
        # The bar after entry is post[1], which is index 221 (entry was post[0]
        # at index 220). Just check the exit isn't on the entry-day or
        # entry-day+1 (the trap).
        exit_date = sell_legs[0]["date"]
        entry_trade = [t for t in r.trades if t["side"] == "BUY"][0]
        entry_date = entry_trade["date"]
        all_dates = [str(d)[:10] for d in df.index]
        entry_idx = all_dates.index(entry_date)
        trap_date = all_dates[entry_idx + 1]
        assert exit_date != trap_date, (
            f"exit on the trap bar ({trap_date}) means same-bar ratchet "
            f"still firing; exit_date={exit_date} side={sell_legs[0]['side']}"
        )
