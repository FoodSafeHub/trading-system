"""Scan-by-signal mode.

The scanner already records, per symbol, which strategies fired in which
direction. Signal mode is a strategy-first lens over that same vote map: pick one
or more strategies + a direction, and return every symbol where ANY selected
strategy fired that side. Consensus mode must stay unchanged.

These tests pin:
  - _strategy_matches maps stored vote names (generic "{sym}_{Label}", perplexity
    "perplexity:Name") to the picker identifiers.
  - run_scan in signal mode returns only symbols whose selected strategy fired the
    requested direction; multi-select is an ANY union.
  - run_scan in consensus mode is unaffected by the new fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.services.scanner.scanner_service as svc
from app.schemas.scanner import ScanConfig
from app.services.scanner.scanner_service import _strategy_matches, run_scan


def test_strategy_matches_generic_and_perplexity():
    assert _strategy_matches("AAPL_RSI2_Mean_Reversion", {"rsi2_mean_reversion"})
    assert not _strategy_matches("AAPL_Pullback_EMA50", {"rsi2_mean_reversion"})
    assert _strategy_matches("perplexity:RSI_Swing_Reversal",
                             {"perplexity:RSI_Swing_Reversal"})
    assert not _strategy_matches("perplexity:BB_Breakout",
                                 {"perplexity:RSI_Swing_Reversal"})


def _fake_df():
    # 260 rows of gently rising closes so filters (price/vol) pass and there's
    # enough history for the strategy/score paths.
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    close = np.linspace(50, 80, 260)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1,
         "Close": close, "Volume": 5_000_000},
        index=idx,
    )


@pytest.fixture()
def patched_scan(monkeypatch):
    """Stub data + strategy layers so run_scan exercises only the scan/score/
    filter logic with deterministic votes. Each symbol 'fires' a configured
    (strategy_name, direction) via a fake bollinger config + engine.
    """
    df = _fake_df()
    monkeypatch.setattr(svc, "get_ohlcv", lambda *a, **k: df)
    monkeypatch.setattr(svc, "get_universe",
                        lambda universe, custom: ["AAA", "BBB"])
    # No perplexity noise.
    monkeypatch.setattr(svc, "run_perplexity_signal", lambda *a, **k: [])

    # Per-symbol scripted signals: AAA fires RSI2 BUY, BBB fires Pullback BUY.
    from types import SimpleNamespace
    fires = {
        "AAA": ("AAA_RSI2_Mean_Reversion", "BUY"),
        "BBB": ("BBB_Pullback_EMA50", "BUY"),
    }

    def fake_generic(symbol):
        name, _dir = fires[symbol]
        return [SimpleNamespace(name=name, type="x", enabled=True, symbol=symbol)]

    monkeypatch.setattr(svc, "_make_generic_configs", fake_generic)
    # Keep the real type->label map stable for matching (uses real factory),
    # so override the helper to the known generic mapping.
    monkeypatch.setattr(svc, "_generic_type_to_label", lambda: {
        "rsi2_mean_reversion": "RSI2_Mean_Reversion",
        "pullback_ema50": "Pullback_EMA50",
    })

    class _Eng:
        def run(self, cfg, prices):
            name, direction = fires[cfg.symbol]
            return [SimpleNamespace(direction=direction, name=cfg.name)]

    monkeypatch.setattr(svc, "_engine", _Eng())
    # strategies.json lookup returns nothing so fake_generic is used.
    monkeypatch.setattr(svc, "load_strategies_from_config", lambda: [])
    monkeypatch.setattr(svc, "apply_filters", lambda symbol, df, **k: SimpleNamespace(
        passed=True, reason="", price=80.0, avg_volume=5_000_000))
    return df


def test_signal_mode_returns_only_matching_strategy(patched_scan):
    cfg = ScanConfig(universe="custom", custom_symbols=["AAA", "BBB"], top_n=10,
                     scan_mode="signal", signal_strategies=["rsi2_mean_reversion"],
                     scan_direction="BUY", min_price=1.0)
    summary = run_scan(cfg)
    syms = {c.symbol for c in summary.top_candidates}
    assert syms == {"AAA"}  # only RSI2 firer; BBB's Pullback is excluded


def test_signal_mode_any_union(patched_scan):
    cfg = ScanConfig(universe="custom", custom_symbols=["AAA", "BBB"], top_n=10,
                     scan_mode="signal",
                     signal_strategies=["rsi2_mean_reversion", "pullback_ema50"],
                     scan_direction="BUY", min_price=1.0)
    summary = run_scan(cfg)
    syms = {c.symbol for c in summary.top_candidates}
    assert syms == {"AAA", "BBB"}  # ANY: both selected strategies' firers


def test_consensus_mode_unaffected(patched_scan):
    cfg = ScanConfig(universe="custom", custom_symbols=["AAA", "BBB"], top_n=10,
                     scan_mode="consensus", scan_direction="BUY", min_price=1.0)
    summary = run_scan(cfg)
    syms = {c.symbol for c in summary.top_candidates}
    assert syms == {"AAA", "BBB"}  # consensus keeps both (each has 1 agreeing)
