"""Unit tests for the Compression Expansion Efficiency Indicator (CEEI)."""
import logging

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.ceei import CEEIParams, compute_ceei
from app.services.strategy.rules import evaluate_strategy


def _ohlcv(closes: np.ndarray, ranges: np.ndarray | None = None,
           vol: np.ndarray | None = None, clv: float = 0.0) -> pd.DataFrame:
    """Synthetic OHLCV. `ranges` sets each bar's high-low span; `clv` places the
    close within the bar (-1 = at low, +1 = at high)."""
    n = len(closes)
    if ranges is None:
        ranges = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    # close = low + (clv+1)/2 * range  =>  low = close - (clv+1)/2 * range
    frac = (clv + 1) / 2
    low = closes - frac * ranges
    high = low + ranges
    return pd.DataFrame({
        "Open": (high + low) / 2,
        "High": high,
        "Low": low,
        "Close": closes,
        "Volume": vol if vol is not None else np.full(n, 1_000_000.0),
    })


def _coil_and_fire(n_pre: int = 120, n_coil: int = 25, n_fire: int = 10,
                   seed: int = 5) -> pd.DataFrame:
    """Normal regime -> tight sideways coil -> high-volume upside breakout.
    The canonical CEEI setup."""
    np.random.seed(seed)
    pre = 100 + np.cumsum(np.random.normal(0.1, 1.2, n_pre))
    base = pre[-1]
    coil = base + np.random.normal(0, 0.15, n_coil)               # tight chop
    fire = base + np.cumsum(np.random.normal(2.0, 0.3, n_fire))   # directional release
    closes = np.concatenate([pre, coil, fire])

    ranges = np.concatenate([
        np.random.uniform(1.5, 3.0, n_pre),
        np.random.uniform(0.2, 0.5, n_coil),      # compressed ranges
        np.random.uniform(3.0, 5.0, n_fire),      # expanding ranges
    ])
    vol = np.concatenate([
        np.random.uniform(0.9e6, 1.1e6, n_pre),
        np.random.uniform(0.5e6, 0.7e6, n_coil),  # dry-up
        np.random.uniform(2.5e6, 3.5e6, n_fire),  # participation surge
    ])
    df = _ohlcv(closes, ranges, vol, clv=0.0)
    # Fire bars close near their highs
    fire_slice = slice(n_pre + n_coil, None)
    df.loc[df.index[fire_slice], "Low"] = closes[fire_slice] - ranges[fire_slice] * 0.9
    df.loc[df.index[fire_slice], "High"] = closes[fire_slice] + ranges[fire_slice] * 0.1
    return df


def _random_walk(n: int = 250, seed: int = 9) -> pd.DataFrame:
    np.random.seed(seed)
    return _ohlcv(100 + np.cumsum(np.random.normal(0, 1.0, n)))


class TestComponents:
    def test_all_scores_bounded_0_100(self):
        df = _random_walk(300)
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        for s in (r.compression_score, r.expansion_score, r.efficiency_score, r.ceei_score):
            valid = s.dropna()
            assert not valid.empty
            assert valid.between(-1e-9, 100 + 1e-9).all()

    def test_compression_high_during_coil(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        coil = slice(120 + 15, 120 + 25)     # late coil, percentiles caught up
        pre = slice(60, 110)
        assert r.compression_score.iloc[coil].mean() > r.compression_score.iloc[pre].mean() + 20

    def test_expansion_high_on_breakout(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        fire = slice(145, 155)
        coil = slice(125, 144)
        assert r.expansion_score.iloc[fire].mean() > r.expansion_score.iloc[coil].mean() + 15
        assert (r.expansion_direction.iloc[fire] == 1).mean() > 0.7

    def test_efficiency_high_in_trend_low_in_chop(self):
        trend = _ohlcv(np.linspace(100, 150, 100))
        np.random.seed(1)
        chop = _ohlcv(100 + np.random.normal(0, 1.5, 100))
        rt = compute_ceei(trend["High"], trend["Low"], trend["Close"], trend["Volume"])
        rc = compute_ceei(chop["High"], chop["Low"], chop["Close"], chop["Volume"])
        assert rt.efficiency_score.iloc[-30:].mean() > 90
        assert rc.efficiency_score.iloc[-30:].mean() < 50

    def test_breakout_level_excludes_current_bar(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        # Level must equal the max high of the PRIOR lookback bars
        i = 150
        lb = r.params.breakout_lookback
        expected = df["High"].iloc[i - lb:i].max()
        assert r.breakout_level.iloc[i] == pytest.approx(expected)


class TestStatesAndSignals:
    def test_setup_state_during_coil(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        assert r.setup_state.iloc[130:145].any(), "coil should mark setup_state"

    def test_trigger_and_buy_on_ignition(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        fire_zone = r.signal.iloc[143:]
        assert (fire_zone == "BUY").any(), "ignition should produce a BUY"
        assert r.trigger_state.iloc[143:].any()
        # No BUYs during the quiet pre-period or the coil itself
        assert (r.signal.iloc[40:140] != "BUY").all()

    def test_sell_on_downside_ignition(self):
        df = _coil_and_fire()
        # Mirror the frame: downside coil-and-fire
        flipped = df.copy()
        pivot = float(df["Close"].max()) + 10
        flipped["Close"] = pivot - df["Close"]
        flipped["High"] = pivot - df["Low"]
        flipped["Low"] = pivot - df["High"]
        r = compute_ceei(flipped["High"], flipped["Low"], flipped["Close"], flipped["Volume"])
        assert (r.signal.iloc[143:] == "SELL").any(), "downside ignition should produce a SELL"

    def test_no_signals_in_pure_chop(self):
        df = _random_walk(300, seed=2)
        # Low-variance chop: shrink all moves
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        # Chop may rarely fire; assert it does not fire persistently
        assert (r.signal != "HOLD").mean() < 0.05

    def test_signal_values_valid(self):
        df = _random_walk(300)
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        assert set(r.signal.unique()) <= {"BUY", "SELL", "HOLD"}

    def test_works_without_volume(self):
        df = _coil_and_fire()
        r = compute_ceei(df["High"], df["Low"], df["Close"], None)
        assert r.ceei_score.dropna().between(0, 100).all()
        assert (r.signal.iloc[143:] == "BUY").any(), "should still ignite without volume"

    def test_weights_configurable_and_renormalized(self):
        df = _coil_and_fire()
        p = CEEIParams(w_compression=2.0, w_expansion=1.0, w_efficiency=1.0)
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"], params=p)
        assert r.ceei_score.dropna().between(0, 100 + 1e-9).all()

    def test_thresholds_configurable(self):
        df = _coil_and_fire()
        p = CEEIParams(buy_threshold=99.9, expansion_threshold=99.9)
        r = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"], params=p)
        assert (r.signal == "HOLD").all(), "impossible thresholds must suppress all signals"

    def test_signal_logging_includes_components(self, caplog):
        df = _coil_and_fire()
        with caplog.at_level(logging.INFO, logger="app.services.indicators.ceei"):
            # Recompute bar-by-bar over the fire zone until the last bar is a signal
            fired = False
            for end in range(144, len(df) + 1):
                sub = df.iloc[:end]
                r = compute_ceei(sub["High"], sub["Low"], sub["Close"], sub["Volume"],
                                 log_context="TEST")
                if r.signal.iloc[-1] == "BUY":
                    fired = True
                    break
        assert fired
        msgs = [m for m in caplog.messages if "CEEI BUY fired" in m]
        assert msgs, "BUY should be logged"
        assert "[TEST]" in msgs[0]
        assert "compression=" in msgs[0] and "expansion=" in msgs[0] and "efficiency=" in msgs[0]


class TestCEEIRule:
    """rule_ceei via evaluate_strategy — the pipeline entry point."""

    def test_returns_valid_signal_with_ohlcv(self):
        df = _random_walk(300)
        sig = evaluate_strategy("ceei", "TEST", df["Close"], {}, ohlcv=df)
        assert sig.direction in ("BUY", "SELL", "HOLD")
        assert sig.strategy_name == "ceei"
        for key in ("ceei_score", "compression", "expansion", "efficiency",
                    "setup_state", "trigger_state", "breakout_level"):
            assert key in sig.indicators

    def test_buy_fires_through_pipeline_on_ignition(self):
        df = _coil_and_fire()
        fired = False
        for end in range(144, len(df) + 1):
            sub = df.iloc[:end]
            sig = evaluate_strategy("ceei", "TEST", sub["Close"],
                                    {"paper_only": False}, ohlcv=sub)
            if sig.direction == "BUY":
                fired = True
                assert sig.indicators["trigger_state"] is True
                assert sig.confidence >= 0.6
                assert "gates" in sig.indicators
                assert all(isinstance(v, bool) for v in sig.indicators["gates"].values())
                break
        assert fired

    def _fire_bar_end(self, df) -> int | None:
        """First bar-count at which the live-enabled rule emits BUY."""
        for end in range(144, len(df) + 1):
            sub = df.iloc[:end]
            sig = evaluate_strategy("ceei", "TEST", sub["Close"],
                                    {"paper_only": False}, ohlcv=sub)
            if sig.direction == "BUY":
                return end
        return None

    def test_paper_only_suppresses_live_signal(self, caplog):
        df = _coil_and_fire()
        end = self._fire_bar_end(df)
        assert end is not None
        sub = df.iloc[:end]
        with caplog.at_level(logging.WARNING, logger="app.services.strategy.rules"):
            sig = evaluate_strategy("ceei", "TEST", sub["Close"], {}, ohlcv=sub)
        assert sig.direction == "HOLD", "default paper_only must suppress live BUY"
        assert sig.indicators["paper_only_suppressed"] == "BUY"
        assert any("PAPER-ONLY" in m for m in caplog.messages)

    def test_backtest_mode_exempt_from_paper_only(self):
        df = _coil_and_fire()
        end = self._fire_bar_end(df)
        assert end is not None
        sub = df.iloc[:end]
        sig = evaluate_strategy("ceei", "TEST", sub["Close"],
                                {"_backtest_mode": True}, ohlcv=sub)
        assert sig.direction == "BUY", "backtest engine must see the real signal"

    def test_close_only_fallback(self):
        df = _random_walk(300)
        sig = evaluate_strategy("ceei", "TEST", df["Close"], {})
        assert sig.direction in ("BUY", "SELL", "HOLD")

    def test_insufficient_data_holds(self):
        df = _random_walk(30)
        sig = evaluate_strategy("ceei", "TEST", df["Close"], {}, ohlcv=df)
        assert sig.direction == "HOLD"
        assert sig.indicators == {}

    def test_params_flow_through(self):
        df = _coil_and_fire()
        sig = evaluate_strategy(
            "ceei", "TEST", df["Close"],
            {"buy_threshold": 99.9, "expansion_threshold": 99.9},
            ohlcv=df,
        )
        assert sig.direction == "HOLD"
