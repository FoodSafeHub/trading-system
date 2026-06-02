"""
Registry pins for the perplexity decision pass (2026-06-02).

These tests are the canonical record of "which strategies are LIVE / RESEARCH-
ONLY / RETIRED" — flipping a flag must trip a test so the decision pass can't
be reverted silently.

Source artifact: reports/perplexity_strategy_decisions.md
"""
from __future__ import annotations

import importlib

import pytest

from app.services.strategy.perplexity import PERPLEXITY_STRATEGIES
from app.services.strategy.perplexity.base import PerplexityStrategy


# Decision pass output. Keep these sets in sync with
# reports/perplexity_strategy_decisions.md.
KEEP = {
    "EMA_Mean_Reversion",
    "BB_Mean_Reversion",
}
RESEARCH_ONLY = {
    # RESEARCH-ONLY (positive but fragile / thin)
    "Breakout_Consolidation",
    "RSI_Swing_Reversal",
    # NEEDS-FOLLOW-UP (engine drops short-side SELLs from these patterns)
    "Daily_NR_Breakout",
    "Daily_Hammer_Star",
    "Daily_Engulfing_Volume",
    "Daily_Three_Bar_Push",
}
RETIRE = {
    "Supertrend_Swing",
    "BB_Breakout",
    "MA_Crossover_RSI",
    "Fib_Pullback_Support",
}

ALL_KNOWN = KEEP | RESEARCH_ONLY | RETIRE


def _by_name() -> dict[str, PerplexityStrategy]:
    return {s.name: s for s in PERPLEXITY_STRATEGIES}


# ── Registry shape ───────────────────────────────────────────────────────────


def test_all_decision_pass_strategies_are_registered():
    """Every decision-pass-named strategy must still be in PERPLEXITY_STRATEGIES.
    Removing one from the list (vs flipping its flag) is what we're guarding
    against — the brief said keep the code, just don't fire live."""
    names = set(_by_name().keys())
    missing = ALL_KNOWN - names
    assert not missing, f"strategies disappeared from PERPLEXITY_STRATEGIES: {missing}"


def test_registry_contains_no_unexpected_strategies():
    """Conversely, if a new strategy is added without a decision-pass call,
    this test will alert the next reviewer to update the decision artifact."""
    names = set(_by_name().keys())
    unexpected = names - ALL_KNOWN
    assert not unexpected, (
        f"strategies in registry without a decision-pass classification: "
        f"{unexpected}. Update reports/perplexity_strategy_decisions.md "
        f"and this test."
    )


# ── KEEP set is live ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(KEEP))
def test_keep_strategies_are_enabled_and_live(name):
    s = _by_name()[name]
    assert s.enabled is True, f"KEEP strategy {name} must remain enabled"
    assert getattr(s, "research_only", False) is False, (
        f"KEEP strategy {name} must NOT be research_only"
    )


# ── RESEARCH-ONLY set ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(RESEARCH_ONLY))
def test_research_only_strategies_are_research_only(name):
    """RESEARCH-ONLY strategies stay importable and runnable in backtests
    (enabled=True) but are filtered out of live signals (research_only=True)."""
    s = _by_name()[name]
    assert s.enabled is True, (
        f"{name} should stay enabled so backtests can still evaluate it"
    )
    assert getattr(s, "research_only", False) is True, (
        f"{name} must be marked research_only to skip live signals"
    )


# ── RETIRE set ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(RETIRE))
def test_retired_strategies_are_disabled(name):
    """RETIRE strategies are off entirely (enabled=False). Code remains in
    the module so backtests/tests can still import the class for comparison
    or future re-enablement, but the runner skips them."""
    s = _by_name()[name]
    assert s.enabled is False, (
        f"RETIRED strategy {name} must have enabled=False"
    )


# ── Live runner filter behaviour ────────────────────────────────────────────


def test_live_runner_skips_disabled_and_research_only():
    """run_perplexity_signal must skip both enabled=False and research_only
    strategies. We verify the FILTER without running real backtests: any
    strategy that would be returned to live must be in KEEP."""
    live_strategies = [
        s for s in PERPLEXITY_STRATEGIES
        if s.enabled and not getattr(s, "research_only", False)
    ]
    live_names = {s.name for s in live_strategies}
    assert live_names == KEEP, (
        f"live filter mismatch: got {live_names}, expected {KEEP}"
    )


def test_retired_strategies_still_importable():
    """RETIRE classes must remain importable so future re-enablement is a
    flag flip, not a code resurrection."""
    # If any import below fails, the test fails with the original ImportError.
    mod = importlib.import_module("app.services.strategy.perplexity.strategies")
    for name in ("SupertrendSwing", "BollingerBandBreakout",
                 "MaCrossoverRsi", "FibPullbackSupport"):
        assert hasattr(mod, name), (
            f"RETIRE strategy class {name} must remain in strategies.py; "
            f"do not delete the implementation"
        )


def test_research_only_strategies_still_importable():
    """RESEARCH-ONLY / NEEDS-FOLLOW-UP classes must remain importable for
    the same reason — flag flip, not code resurrection."""
    daily = importlib.import_module("app.services.strategy.perplexity.momentum_strategies")
    swing = importlib.import_module("app.services.strategy.perplexity.strategies")
    for name in ("PerpEngulfingVolumeSurge", "PerpNarrowRangeBreakout",
                 "PerpThreeBarPush", "PerpHammerShootingStar"):
        assert hasattr(daily, name), f"{name} must remain importable"
    for name in ("BreakoutConsolidation", "RsiSwingReversal"):
        assert hasattr(swing, name), f"{name} must remain importable"
