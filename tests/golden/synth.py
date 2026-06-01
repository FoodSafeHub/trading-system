"""Deterministic synthetic fixtures for Phase 0 golden parity tests.

Everything here is seeded and offline so the golden baseline is reproducible:
the same call always yields the same OHLCV frame and therefore the same
BacktestResult. Both the one-time capture script and the pytest assertions
import from this module, so they exercise identical inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(seed: int = 42, n: int = 400, start: str = "2020-01-01") -> pd.DataFrame:
    """A deterministic OHLCV frame with enough bars (>=250) for every rule.

    Built from a seeded geometric random walk with a small upward drift, then
    OHLC is synthesised around the close so wick/ATR-based rules have real
    High/Low structure. Volume is seeded integers.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# Strategy/param combos captured by the golden engine snapshot. One trailing
# (trail_enabled) case is included so the legacy Chandelier path is locked.
ENGINE_CASES = [
    {
        "name": "rsi2_no_trail",
        "strategy_type": "rsi2_mean_reversion",
        "params": {
            "rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 70,
            "sma_trend": 200, "exit_sma": 5, "atr_skip_threshold": 5.0,
        },
    },
    {
        "name": "pullback_ema50_trail",
        "strategy_type": "pullback_ema50",
        "params": {
            "ema_trend": 50, "ema_slope_bars": 5, "price_ema_proximity_pct": 3.0,
            "rsi_period": 14, "rsi_min": 35, "rsi_max": 60, "wick_ratio_min": 0.3,
            "exit_rsi": 65, "exit_extension_pct": 3.0, "bear_skip_threshold_pct": 10.0,
            # Trailing overlay ON — locks the legacy Chandelier behaviour.
            "trail_enabled": True, "trail_trigger_pct": 3.0,
            "atr_trail_mult": 3.0, "atr_trail_period": 22,
            # Disable hard stop-loss so golden baseline matches pre-stop-loss engine.
            "stop_loss_pct": 0,
        },
    },
    {
        "name": "vix_spike_no_trail",
        "strategy_type": "vix_spike_reversal",
        "params": {
            "atr_period": 14, "atr_spike_threshold": 3.0, "atr_exit_threshold": 2.0,
            "rsi_period": 14, "rsi_entry_max": 30, "rsi_exit": 55,
            "bb_pos_max": 0.15, "wick_ratio_min": 0.5,
            "prior_decline_pct": 2.0, "prior_decline_bars": 3,
        },
    },
]


def _trail_params(enabled: bool = True) -> dict:
    return {
        "trail_enabled": enabled, "trail_trigger_pct": 3.0,
        "atr_trail_mult": 3.0, "atr_trail_period": 22,
        "stop_loss_pct": 0,   # disable hard stop so chandelier golden tests are stable
    }


# Crafted, fully deterministic scenarios that exercise every branch of the
# legacy Chandelier overlay. Primitives only (JSON-able) — both the capture
# script and the parity test rebuild the StrategySignal/frames from these and
# compare the overlay's OUTPUT.
CHANDELIER_CASES = [
    # Unrealized < trigger -> overlay must return the signal unchanged.
    {"name": "below_trigger", "n": 30, "c_now": 101.0, "entry": 100.0,
     "highest_close": 101.0, "direction": "HOLD", "has_position": True,
     "trail_enabled": True},
    # In profit, close above chandelier, rule says SELL -> ride winner (HOLD).
    {"name": "ride_winner", "n": 30, "c_now": 109.0, "entry": 100.0,
     "highest_close": 110.0, "direction": "SELL", "has_position": True,
     "trail_enabled": True},
    # In profit, close below chandelier -> forced SELL (trail_exit).
    {"name": "forced_sell", "n": 30, "c_now": 103.0, "entry": 99.0,
     "highest_close": 112.0, "direction": "HOLD", "has_position": True,
     "trail_enabled": True},
    # Trail disabled -> unchanged even when a SELL would otherwise be held.
    {"name": "trail_disabled", "n": 30, "c_now": 109.0, "entry": 100.0,
     "highest_close": 110.0, "direction": "SELL", "has_position": True,
     "trail_enabled": False},
    # No open position -> overlay is a no-op.
    {"name": "no_position", "n": 30, "c_now": 109.0, "entry": 100.0,
     "highest_close": 110.0, "direction": "SELL", "has_position": False,
     "trail_enabled": True},
]


def build_chandelier_inputs(case: dict):
    """Rebuild (prices, ohlcv, params, signal_kwargs, position_or_none) from a
    CHANDELIER_CASES primitive dict. Deterministic — no RNG."""
    n = case["n"]
    c_now = case["c_now"]
    # A gentle ramp into c_now gives a stable ATR; final bar pinned to c_now.
    closes = list(np.linspace(c_now - 5.0, c_now, n - 1)) + [c_now]
    closes = [round(float(c), 6) for c in closes]
    idx = pd.bdate_range("2020-01-01", periods=n)
    close_s = pd.Series(closes, index=idx)
    ohlcv = pd.DataFrame(
        {
            "Open": close_s.values,
            "High": close_s.values + 1.0,
            "Low": close_s.values - 1.0,
            "Close": close_s.values,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    params = _trail_params(case["trail_enabled"])
    signal_kwargs = {
        "symbol": "TEST", "direction": case["direction"], "strength": 0.5,
        "price_at_signal": c_now, "indicators": {"src": "test"},
        "strategy_name": "golden",
    }
    return close_s, ohlcv, params, signal_kwargs, case


def overlay_to_snapshot(sig) -> dict:
    """JSON-safe reduction of an overlay-returned StrategySignal."""
    return {
        "direction": sig.direction,
        "strategy_name": sig.strategy_name,
        "indicators": sig.indicators,
    }


def result_to_snapshot(r) -> dict:
    """Reduce a BacktestResult to a JSON-safe, comparable dict."""
    return {
        "final_capital": r.final_capital,
        "total_return_pct": r.total_return_pct,
        "total_pnl": r.total_pnl,
        "total_trades": r.total_trades,
        "winning_trades": r.winning_trades,
        "losing_trades": r.losing_trades,
        "win_rate_pct": r.win_rate_pct,
        "max_drawdown_pct": r.max_drawdown_pct,
        "sharpe_ratio": r.sharpe_ratio,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "trades": [
            {"date": t.date, "side": t.side, "price": round(t.price, 6),
             "quantity": round(t.quantity, 6), "value": round(t.value, 6)}
            for t in r.trades
        ],
        "equity_curve": r.equity_curve,
    }
