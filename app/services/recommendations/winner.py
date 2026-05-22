from __future__ import annotations

"""Pick the historically best strategy from a Compare All result set.

This is the SAME ranking the dashboard's Backtest page uses for its green
"Recommended strategy" banner — pulled into a service so the scanner can call
it without spawning a UI. Keeping one implementation prevents the badge on
the scanner from disagreeing with the banner on the backtest page.

Ranking weights (sum to 1.0): total_return 40%, profit_factor 25%,
win_rate 20%, sharpe 10%, total_trades 5%. Profitable strategies are ranked
against each other first; only when no strategy is profitable does the
ranker fall back to the unprofitable set so we still pick a winner.
"""

from typing import Sequence


_METRIC_WEIGHTS = {
    "total_return_pct": 0.40,
    "profit_factor":    0.25,
    "win_rate_pct":     0.20,
    "sharpe_ratio":     0.10,
    "total_trades":     0.05,
}

# Same statistical guardrail the dashboard uses — under 3 round trips the
# numbers are noise. Filtered out before ranking.
MIN_TRADES_FOR_RANK = 3


def pick_winner(rows: Sequence[dict]) -> dict | None:
    """Return the best-ranked row from a Compare All result list, or None.

    Rows are the dicts produced by /backtest/custom-compare-all. We drop rows
    that errored or that have too few trades to be statistically meaningful.
    """
    valid = [
        r for r in rows
        if not r.get("error") and (r.get("total_trades") or 0) >= MIN_TRADES_FOR_RANK
    ]
    if not valid:
        return None

    profitable = [r for r in valid if (r.get("total_return_pct") or 0) > 0]
    pool = profitable if profitable else valid

    scores: dict[str, float] = {r["strategy_name"]: 0.0 for r in pool}
    n = len(pool)
    for metric, weight in _METRIC_WEIGHTS.items():
        def _val(r, m=metric) -> float:
            v = r.get(m)
            return float("-inf") if v is None else float(v)
        ranked = sorted(pool, key=_val, reverse=True)
        for rank, r in enumerate(ranked, start=1):
            scores[r["strategy_name"]] += weight * (n - rank + 1) / n

    return max(pool, key=lambda r: scores[r["strategy_name"]])
