"""Regression tests for the trader-style strategy ranking service.

These lock in the two behaviors that motivated the rewrite:

  1. A tiny "perfect" backtest (e.g. 6 trades, 100% win rate, undefined profit
     factor) must NOT win over a well-sampled, solidly-profitable strategy.
     The old ranker treated an undefined profit factor as +infinity, so it did.

  2. Rule 5 still holds: a small sample CAN win if it is *dramatically* superior
     on raw, risk-adjusted magnitudes — the steep sample-size penalty only
     blocks marginal tiny samples, not genuinely outstanding ones.
"""

from app.services.recommendations.winner import (
    confidence_label,
    pick_winner,
    score_strategies,
    trade_confidence,
)


def _row(name, n, wr, pf, ret, sharpe, exp, dd):
    return {
        "strategy_name": name,
        "total_trades": n,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "total_return_pct": ret,
        "sharpe_ratio": sharpe,
        "expectancy_pct": exp,
        "max_drawdown_pct": dd,
    }


def test_tiny_perfect_sample_does_not_beat_proven_strategy():
    rows = [
        _row("Proven_BB_MeanRev", 47, 66.0, 2.1, 140.0, 1.3, 1.8, 18.0),
        _row("Mid_Momentum",      32, 59.0, 1.7, 210.0, 1.1, 2.4, 33.0),
        # 6 trades, 100% win rate, undefined PF (no losses) — the trap.
        _row("Tiny_RSI2",          6, 100.0, None, 95.0, 2.0, 3.0, 5.0),
    ]
    winner = pick_winner(rows)
    assert winner is not None
    assert winner["strategy_name"] == "Proven_BB_MeanRev"

    ranked = score_strategies(rows)
    tiny = next(r for r in ranked if r["strategy_name"] == "Tiny_RSI2")
    # The tiny sample is demoted and carries the overfit warnings.
    assert ranked[-1]["strategy_name"] == "Tiny_RSI2"
    assert tiny["_confidence_label"] == "very low confidence"
    assert any("small sample" in w.lower() for w in tiny["_warnings"])


def test_dramatically_superior_rare_setup_can_still_win():
    rows = [
        # 8 trades but crushing on every risk-adjusted metric.
        _row("RareGem",  8, 88.0, 4.5, 300.0, 3.0, 6.0, 4.0),
        _row("Mediocre", 60, 52.0, 1.2, 40.0, 0.5, 0.4, 35.0),
    ]
    winner = pick_winner(rows)
    assert winner is not None
    assert winner["strategy_name"] == "RareGem"


def test_no_hard_trade_cutoff_low_sample_still_ranked():
    # A single profitable low-sample strategy must still produce a winner —
    # there is intentionally no minimum-trade gate that drops everything.
    rows = [_row("Solo", 4, 75.0, 1.6, 30.0, 0.9, 1.2, 10.0)]
    winner = pick_winner(rows)
    assert winner is not None
    assert winner["strategy_name"] == "Solo"


def test_undefined_profit_factor_is_not_infinity():
    # Two identical strategies except one has undefined PF on a small sample;
    # the well-sampled one with a real PF must score at least as high.
    rows = [
        _row("Defined_PF", 50, 60.0, 2.0, 100.0, 1.2, 1.5, 15.0),
        _row("Undefined_PF", 6, 100.0, None, 100.0, 1.2, 1.5, 15.0),
    ]
    ranked = score_strategies(rows)
    assert ranked[0]["strategy_name"] == "Defined_PF"


def test_trade_confidence_monotonic_and_bounded():
    assert trade_confidence(0) == 0.0
    assert 0.0 < trade_confidence(6) < trade_confidence(20) < trade_confidence(100) <= 1.0


def test_confidence_labels():
    assert confidence_label(5) == "very low confidence"
    assert "promising" in confidence_label(15)
    assert confidence_label(30) == "usable sample"
    assert confidence_label(75) == "strong sample"
    assert confidence_label(150) == "robust sample"


def test_error_rows_dropped_empty_pool_returns_none():
    assert pick_winner([{"strategy_name": "x", "error": "boom"}]) is None
    assert score_strategies([]) == []
