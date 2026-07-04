"""Unit tests for the CEEI meta-layer coupling framework."""
import numpy as np
import pandas as pd

from app.services.backtest.ceei_meta import (
    MODES,
    CouplingConfig,
    compute_ceei_for,
    discover_strategies,
    entry_mask,
    metrics,
    ranking_selection,
    simulate,
)


def _df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.normal(0.1, 1.0, n))
    rng = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": closes - rng * 0.2, "High": closes + rng * 0.5,
        "Low": closes - rng * 0.5, "Close": closes,
        "Volume": np.random.uniform(1e6, 2e6, n),
    }, index=idx)


class TestDiscovery:
    def test_excludes_meta_indicators(self):
        strategies = discover_strategies()
        assert "ceei" not in strategies and "amat" not in strategies
        assert len(strategies) >= 15, "should find the swing strategy registry"

    def test_whitelist_and_exclude(self):
        assert discover_strategies(include=["breakout"]) == ["breakout"]
        assert "supertrend" not in discover_strategies(exclude=["supertrend"])


class TestMasks:
    def setup_method(self):
        self.df = _df()
        self.ceei = compute_ceei_for(self.df)
        self.cfg = CouplingConfig()

    def test_all_modes_produce_boolean_masks(self):
        for mode in MODES:
            if mode == "E_ranking":
                continue  # cross-symbol, handled separately
            m = entry_mask(mode, self.ceei, self.cfg)
            assert m.dtype == bool
            assert len(m) == len(self.df)

    def test_base_mask_allows_everything(self):
        assert entry_mask("A_base", self.ceei, self.cfg).all()

    def test_filters_are_subsets_of_base(self):
        for mode in ("B_setup_filter", "C_trigger_filter", "D_score_filter", "G_veto"):
            m = entry_mask(mode, self.ceei, self.cfg)
            assert m.sum() <= len(self.df)

    def test_trigger_stricter_than_setup(self):
        setup = entry_mask("B_setup_filter", self.ceei, self.cfg)
        trigger = entry_mask("C_trigger_filter", self.ceei, self.cfg)
        assert trigger.sum() <= setup.sum()

    def test_score_threshold_configurable(self):
        loose = entry_mask("D_score_filter", self.ceei, CouplingConfig(score_threshold=10))
        strict = entry_mask("D_score_filter", self.ceei, CouplingConfig(score_threshold=90))
        assert strict.sum() < loose.sum()


class TestRanking:
    def test_keeps_top_scorer_per_date(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="D")
        dates = {"X": idx, "Y": idx}
        scores = {"X": pd.Series([10, 80, 10, 10, 10], index=idx),
                  "Y": pd.Series([90, 20, 10, 10, 10], index=idx)}
        candidates = {"X": [0, 1], "Y": [0, 1]}
        allowed = ranking_selection(candidates, dates, scores, top=1)
        assert allowed["Y"] == {0}   # Y wins day 0 (90 > 10)
        assert allowed["X"] == {1}   # X wins day 1 (80 > 20)


class TestSimulator:
    def test_base_sell_exit_and_metrics(self):
        df = _df(120)
        ceei = compute_ceei_for(df)
        sig = pd.Series("HOLD", index=df.index)
        sig.iloc[50] = "BUY"
        sig.iloc[55] = "SELL"
        cfg = CouplingConfig()
        trades = simulate("T", df, sig, pd.Series(True, index=df.index), ceei, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "base_sell"
        assert t["hold_bars"] == 5  # entry bar 51, exit next open after SELL = bar 56
        m = metrics(pd.DataFrame(trades), 1.0)
        assert m["trades"] == 1

    def test_time_stop_fallback(self):
        df = _df(150)
        ceei = compute_ceei_for(df)
        sig = pd.Series("HOLD", index=df.index)
        sig.iloc[50] = "BUY"
        cfg = CouplingConfig(time_stop_bars=10)
        trades = simulate("T", df, sig, pd.Series(True, index=df.index), ceei, cfg)
        assert trades[0]["exit_reason"] == "time_stop"
        assert trades[0]["hold_bars"] == 10

    def test_mask_blocks_entry(self):
        df = _df(120)
        ceei = compute_ceei_for(df)
        sig = pd.Series("HOLD", index=df.index)
        sig.iloc[50] = "BUY"
        trades = simulate("T", df, sig, pd.Series(False, index=df.index), ceei,
                          CouplingConfig())
        assert trades == []

    def test_exit_assist_can_only_shorten_holds(self):
        df = _df(200, seed=8)
        ceei = compute_ceei_for(df)
        sig = pd.Series("HOLD", index=df.index)
        for i in (60, 120, 160):
            sig.iloc[i] = "BUY"
        cfg = CouplingConfig(assist_score=101.0)  # always armed → tight trail active
        mask = pd.Series(True, index=df.index)
        base = simulate("T", df, sig, mask, ceei, cfg, exit_assist=False)
        assisted = simulate("T", df, sig, mask, ceei, cfg, exit_assist=True)
        base_first = base[0]
        assist_first = assisted[0]
        assert assist_first["hold_bars"] <= base_first["hold_bars"]
        assert any(t["exit_reason"].startswith("assist_trail") for t in assisted)
