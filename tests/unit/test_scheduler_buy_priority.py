"""
Contract tests for _buy_priority_key — the sort that decides which signals
claim scarce cash first in the scheduler's assigned-signal execute loop.

Pin the contract:
    * SELLs always sort before BUYs (exits are time-critical).
    * Among BUYs, higher confidence sorts first.
    * A BUY without a recorded confidence falls back to a neutral 0.5, so it
      interleaves with scored BUYs rather than always winning or losing.
    * The sort is stable (equal-conviction BUYs keep their original order).
"""
from __future__ import annotations

from app.services.strategy.scheduler import _buy_priority_key


def _sig(symbol, direction, system="perplexity", strat="s"):
    # (symbol, direction, label, entry, stop, system, strategy_name)
    return (symbol, direction, f"{system}:{strat}", 100.0, None, system, strat)


def _order(signals, conf):
    return [s[0] for s in sorted(signals, key=lambda i: _buy_priority_key(i, conf))]


def test_sells_before_buys():
    sigs = [_sig("AAA", "BUY"), _sig("BBB", "SELL")]
    assert _order(sigs, {}) == ["BBB", "AAA"]


def test_buys_ordered_by_confidence_desc():
    sigs = [_sig("LOW", "BUY"), _sig("HIGH", "BUY"), _sig("MID", "BUY")]
    conf = {
        ("LOW", "perplexity", "s"): 0.55,
        ("HIGH", "perplexity", "s"): 0.92,
        ("MID", "perplexity", "s"): 0.70,
    }
    assert _order(sigs, conf) == ["HIGH", "MID", "LOW"]


def test_missing_confidence_falls_back_to_neutral():
    # UNSCORED has no conf entry → 0.5; it should sit between 0.6 and 0.4.
    sigs = [_sig("HI", "BUY"), _sig("UNSCORED", "BUY"), _sig("LO", "BUY")]
    conf = {
        ("HI", "perplexity", "s"): 0.60,
        ("LO", "perplexity", "s"): 0.40,
    }
    assert _order(sigs, conf) == ["HI", "UNSCORED", "LO"]


def test_all_sells_and_buys_grouped():
    sigs = [
        _sig("B1", "BUY"),
        _sig("S1", "SELL"),
        _sig("B2", "BUY"),
        _sig("S2", "SELL"),
    ]
    conf = {("B1", "perplexity", "s"): 0.9, ("B2", "perplexity", "s"): 0.8}
    out = _order(sigs, conf)
    # Both SELLs first (original relative order preserved), then BUYs by conf.
    assert out[:2] == ["S1", "S2"]
    assert out[2:] == ["B1", "B2"]


def test_stable_among_equal_confidence():
    # Two BUYs, same fallback 0.5 → original order kept (stable sort).
    sigs = [_sig("FIRST", "BUY"), _sig("SECOND", "BUY")]
    assert _order(sigs, {}) == ["FIRST", "SECOND"]
