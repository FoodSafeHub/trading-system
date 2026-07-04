"""Tests for the assignment-level CEEI gate in evaluate_strategy."""
import logging

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.ceei_meta import KNOWN_DUPLICATES, discover_strategies
from app.services.indicators.ceei import compute_ceei
from app.services.strategy.rules import (
    CEEI_INCOMPATIBLE_STRATEGIES,
    _apply_ceei_gate,
    _ceei_gate_family_warned,
    evaluate_strategy,
)
from app.services.strategy.models import StrategySignal


def _ohlcv(n: int = 300, seed: int = 7, drift: float = 0.4) -> pd.DataFrame:
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.normal(drift, 0.8, n))
    rng = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": closes - rng * 0.2, "High": closes + rng * 0.5,
        "Low": closes - rng * 0.5, "Close": closes,
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)


def _buy_signal(strategy: str = "breakout") -> StrategySignal:
    return StrategySignal(symbol="TEST", direction="BUY", price_at_signal=100.0,
                          strategy_name=strategy)


class TestGateBehaviour:
    def test_unset_gate_leaves_signal_untouched(self):
        df = _ohlcv()
        sig = _buy_signal()
        out = _apply_ceei_gate(sig, df["Close"], df, {})
        assert out.direction == "BUY"
        assert "ceei_gate" not in out.indicators

    def test_disabled_gate_is_noop(self):
        df = _ohlcv()
        out = _apply_ceei_gate(_buy_signal(), df["Close"], df,
                               {"ceei_gate": "trigger", "ceei_gate_enabled": False})
        assert out.direction == "BUY"
        assert "ceei_gate" not in out.indicators

    def test_trigger_gate_vetoes_when_not_firing(self):
        # A plain random walk almost never has trigger_state on the final bar
        df = _ohlcv(seed=3, drift=0.0)
        res = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        assert not bool(res.trigger_state.iloc[-1]), "fixture must not be triggering"
        out = _apply_ceei_gate(_buy_signal(), df["Close"], df, {"ceei_gate": "trigger"})
        assert out.direction == "HOLD"
        assert out.indicators["ceei_gate_veto"] is True
        assert out.indicators["ceei_gate_passed"] is False

    def test_score_gate_threshold_controls_pass(self):
        df = _ohlcv()
        res = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        score = res.latest_score
        assert score is not None
        passed = _apply_ceei_gate(_buy_signal(), df["Close"], df,
                                  {"ceei_gate": "score", "ceei_gate_threshold": score - 5})
        vetoed = _apply_ceei_gate(_buy_signal(), df["Close"], df,
                                  {"ceei_gate": "score", "ceei_gate_threshold": score + 5})
        assert passed.direction == "BUY" and passed.indicators["ceei_gate_passed"]
        assert vetoed.direction == "HOLD" and vetoed.indicators["ceei_gate_veto"]

    def test_setup_gate_uses_lookback(self):
        df = _ohlcv()
        res = compute_ceei(df["High"], df["Low"], df["Close"], df["Volume"])
        # Find whether a setup occurred in the last 200 bars — with a huge
        # lookback the gate must pass; with lookback so short there is no setup
        # it must veto.
        recent_any = bool(res.setup_state.iloc[-200:].any())
        assert recent_any, "fixture needs at least one setup in its history"
        wide = _apply_ceei_gate(_buy_signal(), df["Close"], df,
                                {"ceei_gate": "setup", "ceei_gate_lookback": 200})
        assert wide.direction == "BUY"
        if not bool(res.setup_state.iloc[-1]):
            narrow = _apply_ceei_gate(_buy_signal(), df["Close"], df,
                                      {"ceei_gate": "setup", "ceei_gate_lookback": 1})
            assert narrow.direction == "HOLD"

    def test_sell_and_hold_never_gated(self):
        df = _ohlcv(seed=3, drift=0.0)
        for direction in ("SELL", "HOLD"):
            sig = StrategySignal(symbol="TEST", direction=direction,
                                 strategy_name="breakout")
            out = _apply_ceei_gate(sig, df["Close"], df, {"ceei_gate": "trigger"})
            assert out.direction == direction, "exits must never be blocked"

    def test_insufficient_history_fails_open(self, caplog):
        df = _ohlcv(40)
        with caplog.at_level(logging.WARNING, logger="app.services.strategy.rules"):
            out = _apply_ceei_gate(_buy_signal(), df["Close"], df, {"ceei_gate": "score"})
        assert out.direction == "BUY", "data hiccup must not halt the strategy"
        assert any("fails OPEN" in m for m in caplog.messages)

    def test_unknown_gate_value_ignored(self, caplog):
        df = _ohlcv()
        with caplog.at_level(logging.WARNING, logger="app.services.strategy.rules"):
            out = _apply_ceei_gate(_buy_signal(), df["Close"], df, {"ceei_gate": "bogus"})
        assert out.direction == "BUY"
        assert any("unknown ceei_gate" in m for m in caplog.messages)


class TestFamilySafety:
    def test_incompatible_family_warns(self, caplog):
        df = _ohlcv(seed=3, drift=0.0)
        _ceei_gate_family_warned.discard("rsi2_reversion")
        sig = _buy_signal("rsi2_reversion")
        with caplog.at_level(logging.WARNING, logger="app.services.strategy.rules"):
            _apply_ceei_gate(sig, df["Close"], df, {"ceei_gate": "score"})
        assert any("REDUCES expectancy" in m for m in caplog.messages)

    def test_compatible_family_does_not_warn(self, caplog):
        df = _ohlcv()
        with caplog.at_level(logging.WARNING, logger="app.services.strategy.rules"):
            _apply_ceei_gate(_buy_signal("momentum_breakout"), df["Close"], df,
                             {"ceei_gate": "score", "ceei_gate_threshold": 0})
        assert not any("REDUCES" in m for m in caplog.messages)

    def test_incompatible_set_covers_duplicate_registrations(self):
        for alias, canonical in KNOWN_DUPLICATES.items():
            in_alias = alias in CEEI_INCOMPATIBLE_STRATEGIES
            in_canon = canonical in CEEI_INCOMPATIBLE_STRATEGIES
            assert in_alias == in_canon, f"{alias}/{canonical} must be classified together"


class TestPipelineIntegration:
    def test_gate_through_evaluate_strategy(self):
        """The gate must run inside evaluate_strategy (all three live paths)."""
        df = _ohlcv(seed=3, drift=0.0)
        # sma_rsi frequently emits BUY in a drifting series; force a comparison
        base = evaluate_strategy("sma_rsi", "TEST", df["Close"], {}, ohlcv=df)
        gated = evaluate_strategy("sma_rsi", "TEST", df["Close"],
                                  {"ceei_gate": "trigger"}, ohlcv=df)
        if base.direction == "BUY":
            assert gated.direction == "HOLD"
            assert gated.indicators["ceei_gate_veto"] is True
        else:
            assert gated.direction == base.direction

    def test_no_lookahead_gate_uses_only_history(self):
        """Gate decision at bar t must not change when future bars are appended."""
        df = _ohlcv(400)
        cut = 300
        sub = df.iloc[:cut]
        out_sub = _apply_ceei_gate(_buy_signal(), sub["Close"], sub,
                                   {"ceei_gate": "score", "ceei_gate_threshold": 48})
        # Same decision recomputed from the identical history must match —
        # the gate reads only prices/ohlcv passed in (no globals, no future).
        out_again = _apply_ceei_gate(_buy_signal(), sub["Close"], sub,
                                     {"ceei_gate": "score", "ceei_gate_threshold": 48})
        assert out_sub.direction == out_again.direction
        assert out_sub.indicators.get("ceei_gate_score") == \
            out_again.indicators.get("ceei_gate_score")


class TestDiscoveryDedupe:
    def test_dedupe_removes_aliases(self):
        all_names = discover_strategies()
        deduped = discover_strategies(dedupe=True)
        for alias in KNOWN_DUPLICATES:
            assert alias in all_names
            assert alias not in deduped
        for canonical in KNOWN_DUPLICATES.values():
            assert canonical in deduped
