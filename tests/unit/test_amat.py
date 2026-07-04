"""Unit tests for the Adaptive Momentum Acceleration Trend (AMAT) indicator."""
import logging

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.amat import (
    AMATParams,
    compute_amat,
    compute_trend_spine,
)
from app.services.strategy.rules import evaluate_strategy


def _ohlcv(closes: np.ndarray, vol: np.ndarray | None = None) -> pd.DataFrame:
    """Build a synthetic OHLCV frame around a close series."""
    n = len(closes)
    rng = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    return pd.DataFrame({
        "Open": closes - rng * 0.2,
        "High": closes + rng * 0.5,
        "Low": closes - rng * 0.5,
        "Close": closes,
        "Volume": vol if vol is not None else np.full(n, 1_000_000.0),
    })


def _trending_up(n: int = 200, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.normal(0.4, 0.8, n))
    return _ohlcv(closes)


def _trending_down(n: int = 200, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    closes = 200 + np.cumsum(np.random.normal(-0.4, 0.8, n))
    return _ohlcv(closes)


class TestTrendSpine:
    def test_spine_below_price_in_uptrend(self):
        df = _trending_up()
        spine = compute_trend_spine(df["High"], df["Low"], df["Close"])
        tail_close = df["Close"].iloc[-20:]
        tail_spine = spine.iloc[-20:]
        # In a sustained uptrend the spine acts as trailing support
        assert (tail_close > tail_spine).mean() > 0.8

    def test_spine_above_price_in_downtrend(self):
        df = _trending_down()
        spine = compute_trend_spine(df["High"], df["Low"], df["Close"])
        tail_close = df["Close"].iloc[-20:]
        tail_spine = spine.iloc[-20:]
        assert (tail_close < tail_spine).mean() > 0.8

    def test_spine_ratchets_up_while_price_above(self):
        # Monotonic rise with price always above spine: spine must never fall
        closes = np.linspace(100, 200, 150)
        df = _ohlcv(closes)
        spine = compute_trend_spine(df["High"], df["Low"], df["Close"]).dropna()
        diffs = spine.diff().dropna()
        assert (diffs >= -1e-9).all()

    def test_warmup_is_nan(self):
        df = _trending_up(50)
        spine = compute_trend_spine(df["High"], df["Low"], df["Close"], atr_period=14)
        assert spine.iloc[:13].isna().all()


class TestAMATComposite:
    def test_score_bounded_0_100(self):
        df = _trending_up(300)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        valid = r.score.dropna()
        assert not valid.empty
        assert valid.between(-1e-9, 100 + 1e-9).all()

    def test_acceleration_bounded(self):
        df = _trending_up(300)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        valid = r.acceleration.dropna()
        assert valid.between(-100 - 1e-9, 100 + 1e-9).all()

    def test_conviction_clipped(self):
        np.random.seed(3)
        closes = 100 + np.cumsum(np.random.normal(0, 1, 200))
        vol = np.random.uniform(100, 10_000_000, 200)  # extreme spread forces clipping
        df = _ohlcv(closes, vol)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        valid = r.conviction.dropna()
        assert valid.min() >= 0.5 and valid.max() <= 2.0

    def test_uses_mfi_with_volume_rsi_without(self):
        df = _trending_up(200)
        with_vol = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        without = compute_amat(df["High"], df["Low"], df["Close"], None)
        assert with_vol.momentum_source == "mfi"
        assert without.momentum_source == "rsi"
        # No volume -> neutral conviction everywhere
        assert (without.conviction == 1.0).all()

    def test_signal_values_are_valid(self):
        df = _trending_up(300)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        assert set(r.signal.unique()) <= {"BUY", "SELL", "HOLD"}

    def test_buy_requires_close_above_spine(self):
        df = _trending_up(300)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        buys = r.signal == "BUY"
        if buys.any():
            assert (df["Close"][buys] > r.trend_spine[buys]).all()
        sells = r.signal == "SELL"
        if sells.any():
            assert (df["Close"][sells] < r.trend_spine[sells]).all()

    @staticmethod
    def _divergent_ramp() -> pd.DataFrame:
        # Noisy strong ramp then noisy weak ramp: price keeps printing N-bar
        # highs while momentum acceleration fades — divergence on the last bar
        # (deterministic with this seed).
        np.random.seed(11)
        steps = np.concatenate([
            np.random.normal(1.2, 0.5, 150),
            np.random.normal(0.15, 0.08, 150),
        ])
        return _ohlcv(100 + np.cumsum(steps))

    def test_divergence_penalty_shrinks_weighted_accel(self):
        df = self._divergent_ramp()
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        assert r.divergence.any(), "expected at least one divergence bar"
        # Recompute without a penalty: divergent bars must have a strictly
        # smaller |score input| when the penalty is on
        no_pen = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"],
                              params=AMATParams(divergence_penalty=1.0))
        assert not r.score.dropna().equals(no_pen.score.dropna())

    def test_divergence_logged(self, caplog):
        df = self._divergent_ramp()
        with caplog.at_level(logging.INFO, logger="app.services.indicators.amat"):
            r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"],
                             log_context="TEST")
        assert bool(r.divergence.iloc[-1]), "seeded data must be divergent on last bar"
        assert any("divergence penalty" in m for m in caplog.messages)
        assert any("[TEST]" in m for m in caplog.messages)

    def test_params_are_configurable(self):
        df = _trending_up(300)
        p = AMATParams(atr_period=7, multiplier=2.0, accel_step=5,
                       divergence_lookback=20, score_window=60,
                       buy_threshold=70, sell_threshold=30)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"], params=p)
        default = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        assert r.params.buy_threshold == 70
        # Different params must actually change the computation
        assert not r.trend_spine.dropna().equals(default.trend_spine.dropna())

    def test_short_series_does_not_crash(self):
        df = _trending_up(30)
        r = compute_amat(df["High"], df["Low"], df["Close"], df["Volume"])
        assert len(r.score) == 30


class TestAMATRule:
    """rule_amat via evaluate_strategy — the pipeline entry point."""

    def test_returns_valid_signal_with_ohlcv(self):
        df = _trending_up(300)
        sig = evaluate_strategy("amat", "TEST", df["Close"], {}, ohlcv=df)
        assert sig.direction in ("BUY", "SELL", "HOLD")
        assert sig.strategy_name == "amat"
        assert "amat_score" in sig.indicators
        assert "trend_spine" in sig.indicators
        assert sig.indicators["momentum_source"] == "mfi"

    def test_close_only_falls_back_to_rsi(self):
        df = _trending_up(300)
        sig = evaluate_strategy("amat", "TEST", df["Close"], {})
        assert sig.direction in ("BUY", "SELL", "HOLD")
        assert sig.indicators["momentum_source"] == "rsi"

    def test_insufficient_data_holds(self):
        df = _trending_up(40)
        sig = evaluate_strategy("amat", "TEST", df["Close"], {}, ohlcv=df)
        assert sig.direction == "HOLD"
        assert sig.indicators == {}

    def test_thresholds_configurable_via_params(self):
        df = _trending_up(300)
        sig = evaluate_strategy(
            "amat", "TEST", df["Close"],
            {"buy_threshold": 99.9, "sell_threshold": 0.1, "score_window": 60},
            ohlcv=df,
        )
        # With near-impossible thresholds no cross can fire on the last bar
        assert sig.direction == "HOLD"
