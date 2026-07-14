"""Regression tests for the 2026-07-13 strategy revamp.

Pins the two behavior changes from the strategy audit:
  * engine time-stop — max_hold_bars is now ENFORCED by run_backtest (it was a
    dead parameter that the calibration grids were nonetheless searching over,
    and multi-month "winners" in backtests were just positions no exit ever
    closed until end-of-data);
  * pullback_ema50 trend-fail SELL — a close > trend_fail_below_ema_pct below
    the EMA50 must emit SELL (the rule previously had no downside exit at all:
    RSI sits low and the extension is negative in a decline, so failed entries
    rode the wide protective trail down — GNTX/HLT, July 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.strategy.models import StrategySignal
from app.services.strategy.rules import rule_pullback_ema50


def _ohlcv(closes) -> pd.DataFrame:
    closes = pd.Series([float(c) for c in closes])
    df = pd.DataFrame({
        "Open":   closes.shift(1).fillna(closes.iloc[0]),
        "High":   closes * 1.004,
        "Low":    closes * 0.996,
        "Close":  closes,
        "Volume": 1_000_000.0,
    })
    df.index = pd.date_range("2024-01-02", periods=len(df), freq="B")
    return df


# ── engine time-stop ──────────────────────────────────────────────────────────


def _run_with_stub_strategy(monkeypatch, params: dict):
    """Backtest a stub strategy that BUYs once and then HOLDs forever, so the
    only possible exits are the engine's own (stop-loss / time-stop / end)."""
    from app.services.backtest import engine as eng

    state = {"bought": False}

    def _stub(strategy_type, symbol, prices, p, ohlcv=None, position=None):
        direction = "HOLD"
        if not state["bought"]:
            direction = "BUY"
            state["bought"] = True
        return StrategySignal(symbol=symbol, direction=direction,
                              price_at_signal=float(prices.iloc[-1]),
                              indicators={}, strategy_name="stub")

    monkeypatch.setattr(eng, "evaluate_strategy", _stub)
    df = _ohlcv(100 + 0.05 * np.arange(120))  # gentle uptrend: no stop-loss
    return eng.run_backtest("stub", "TEST", "stub", params, df=df)


def test_time_stop_closes_stale_position(monkeypatch):
    r = _run_with_stub_strategy(monkeypatch, {"max_hold_bars": 5})
    sells = [t for t in r.trades if t.side == "SELL"]
    assert sells, "time-stop should have produced a SELL"
    assert "time_stop_5bars" in sells[0].signal_from


def test_no_time_stop_when_disabled(monkeypatch):
    r = _run_with_stub_strategy(monkeypatch, {"max_hold_bars": 0})
    assert not any("time_stop" in t.signal_from for t in r.trades)


# ── pullback_ema50 trend-fail SELL ───────────────────────────────────────────


def test_pullback_trend_fail_emits_sell():
    # 70 flat bars establish EMA50 ≈ 100, then price collapses ~6% below it.
    closes = [100.0] * 70 + [98.0, 96.0, 94.0]
    df = _ohlcv(closes)
    sig = rule_pullback_ema50("TEST", df["Close"], {}, ohlcv=df)
    assert sig.direction == "SELL"
    assert sig.indicators["ext_pct"] < -3.0


def test_pullback_trend_fail_disabled_holds():
    closes = [100.0] * 70 + [98.0, 96.0, 94.0]
    df = _ohlcv(closes)
    sig = rule_pullback_ema50(
        "TEST", df["Close"], {"trend_fail_below_ema_pct": 0}, ohlcv=df)
    assert sig.direction == "HOLD"


def test_pullback_mild_dip_still_holds():
    # 1.5% below EMA50 is a normal pullback, NOT a trend failure.
    closes = [100.0] * 70 + [99.5, 99.0, 98.6]
    df = _ohlcv(closes)
    sig = rule_pullback_ema50("TEST", df["Close"], {}, ohlcv=df)
    assert sig.direction != "SELL"
