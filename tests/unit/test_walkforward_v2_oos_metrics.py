"""Regression test for the walkforward_v2 OOS-contamination bug.

Before the fix, _run_slice copied r.total_trades / r.win_rate_pct /
r.max_drawdown_pct / r.sharpe_ratio straight from the run_backtest result for
the OOS segment — but when warmup_df is supplied those metrics span the full
IS+OOS run, so OOS reporting was contaminated with IS activity. This test pins
the corrected behaviour:

  * IS segment (no warmup_df) — metrics match the underlying run_backtest, since
    the backtest covers only the IS slice anyway.
  * OOS segment (warmup_df = IS slice) — metrics are recomputed on the OOS-only
    portion of the trades / equity curve.
"""
import pandas as pd

from app.services.backtest.engine import run_backtest
from app.services.backtest.walkforward_v2 import _run_slice
from tests.golden.synth import make_ohlcv

PARAMS = {
    "rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 70,
    "sma_trend": 200, "exit_sma": 5, "atr_skip_threshold": 5.0,
}


def _split(df, train_pct=0.70):
    n = len(df); s = max(60, min(int(n * train_pct), n - 60))
    return df.iloc[:s], df.iloc[s:]


def test_is_segment_unchanged_no_warmup():
    """IS run (no warmup) must match run_backtest one-for-one."""
    df = make_ohlcv(n=400)
    train, _ = _split(df)
    seg = _run_slice("rsi2_mean_reversion", "TEST", PARAMS, train, None, 100_000.0)
    r = run_backtest(strategy_name="x", symbol="TEST", strategy_type="rsi2_mean_reversion",
                     params=PARAMS, period="slice", initial_capital=100_000.0,
                     quantity=0, df=train.copy())
    assert seg.trades == r.total_trades
    assert seg.win_rate_pct == r.win_rate_pct
    assert seg.max_drawdown_pct == r.max_drawdown_pct


def test_oos_segment_metrics_are_oos_only():
    """OOS run (with warmup_df) must report trades, win_rate, max_dd derived
    from the OOS portion only — NOT the IS+OOS continuous run."""
    df = make_ohlcv(n=400)
    train, test = _split(df)
    seg = _run_slice("rsi2_mean_reversion", "TEST", PARAMS, test, train, 100_000.0)

    # Run the full backtest the same way the slice does, then independently
    # recompute the OOS-only counters and compare to the slice's reported values.
    r_full = run_backtest(strategy_name="x", symbol="TEST", strategy_type="rsi2_mean_reversion",
                          params=PARAMS, period="slice", initial_capital=100_000.0,
                          quantity=0, df=pd.concat([train, test]).copy())
    oos_start = str(test.index[0])[:10]

    oos_events = [t for t in r_full.trades if t.date >= oos_start]
    while oos_events and "SELL" in oos_events[0].side:
        oos_events.pop(0)
    buys = [t for t in oos_events if t.side == "BUY"]
    sells = [t for t in oos_events if "SELL" in t.side]
    rt = min(len(buys), len(sells))
    expected_wins = sum(1 for b, s in zip(buys[:rt], sells[:rt]) if s.value > b.value)
    expected_wr = round(expected_wins / rt * 100, 2) if rt > 0 else 0.0

    assert seg.trades == len(oos_events), (
        f"OOS trades {seg.trades} should equal OOS-only events {len(oos_events)}, "
        f"NOT full-run total ({r_full.total_trades})"
    )
    assert seg.win_rate_pct == expected_wr, (
        f"OOS win rate {seg.win_rate_pct} should equal OOS-only "
        f"{expected_wr}, NOT full-run win rate ({r_full.win_rate_pct})"
    )
    # OOS max DD must be less than or equal to full-run max DD (peak resets at OOS start).
    assert seg.max_drawdown_pct <= r_full.max_drawdown_pct + 1e-9


def test_oos_max_dd_bounded_by_full_run():
    """OOS max DD must never exceed the full IS+OOS run's max DD (the OOS peak
    resets at the OOS start, so OOS DD is always ≤ full DD)."""
    df = make_ohlcv(n=400)
    train, test = _split(df)
    seg = _run_slice("rsi2_mean_reversion", "TEST", PARAMS, test, train, 100_000.0)
    r_full = run_backtest(strategy_name="x", symbol="TEST", strategy_type="rsi2_mean_reversion",
                          params=PARAMS, period="slice", initial_capital=100_000.0,
                          quantity=0, df=pd.concat([train, test]).copy())
    assert seg.max_drawdown_pct <= r_full.max_drawdown_pct + 1e-9
