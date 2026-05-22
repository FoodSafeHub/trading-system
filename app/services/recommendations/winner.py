from __future__ import annotations

"""Pick the historically best strategy from a Compare All result set.

This is the SAME ranking the dashboard's Backtest page uses for its green
"Recommended strategy" banner — pulled into a service so the scanner can call
it without spawning a UI. Keeping one implementation prevents the badge on
the scanner from disagreeing with the banner on the backtest page.

Trader-style scoring (replaces the old rank-position scheme)
------------------------------------------------------------
The old ranker awarded points by finish position per metric and treated a
missing profit factor (a strategy with zero losing trades) as +infinity — so
a 6-trade flawless backtest could win on a technicality. This version scores
each strategy on its *normalized magnitudes* and multiplies by a trade-count
confidence factor, so a tiny perfect sample is heavily penalized but still
allowed to win when it is dramatically better on risk-adjusted terms.

There is intentionally NO hard minimum-trade cutoff: rare-but-real setups
(which may never reach 20 trades in a 5y backtest) can still rank and win.
Sample size is expressed as confidence, not a pass/fail gate.

Score = base_score * confidence_multiplier, where the multiplier uses the
sample-size confidence raised to a power (so the penalty bites hard at low
trade counts), and base_score is a weighted sum (weights sum to 1.0):

    expectancy           28%   per-trade edge after losses
    profit_factor        22%   capped at PF_CAP so "no losses" can't dominate
    max_drawdown_penalty 22%   lower drawdown scores higher
    sharpe / ret-to-DD   18%   risk-adjusted quality
    trade_confidence     10%   sample-size term (also applied as a multiplier)

NOTE: A walk-forward / out-of-sample term (originally specced at 30%) is not
included because the Compare All endpoint does not yet produce a walk-forward
metric for these generic strategies. When it does, add it here and renormalize.
"""

import math
from typing import Sequence

# ── Scoring weights (sum to 1.0) ──────────────────────────────────────────────
_W_EXPECTANCY = 0.28
_W_PROFIT_FACTOR = 0.22
_W_DRAWDOWN = 0.22
_W_SHARPE = 0.18
_W_TRADE_CONF = 0.10

# Profit factor above this is clamped — beyond ~5 the marginal information is
# noise, and we never want an undefined PF (zero losses) to act as infinity.
PF_CAP = 5.0

# Confidence saturates toward 1.0 around ~30 trades. 1 - e^(-n/15) gives:
#   n=6 -> 0.33, n=10 -> 0.49, n=20 -> 0.74, n=30 -> 0.86, n=50 -> 0.96.
_CONF_TAU = 15.0

# The sample-size penalty is applied as a STEEP multiplier on the base score:
#   final = base * (floor + (1 - floor) * confidence**_CONF_EXPONENT)
# The exponent makes the penalty bite hard at low trade counts (rule 5 stays
# alive — a dramatically superior rare setup can still win — but a *marginal*
# tiny sample cannot beat a well-sampled strong strategy on a technicality).
# With exponent 2 and floor 0.15:
#   n=6  conf=0.33 -> mult 0.24   (6-trade run keeps ~1/4 of its base score)
#   n=20 conf=0.74 -> mult 0.61
#   n=47 conf=0.96 -> mult 0.93
_CONF_MULT_FLOOR = 0.15
_CONF_EXPONENT = 2.0


def trade_confidence(n_trades: int | None) -> float:
    """Smooth 0..1 sample-size confidence. No hard cutoff."""
    n = int(n_trades or 0)
    if n <= 0:
        return 0.0
    return round(min(1.0, 1.0 - math.exp(-n / _CONF_TAU)), 4)


def confidence_label(n_trades: int | None) -> str:
    """Human label for sample size (rules 11/12)."""
    n = int(n_trades or 0)
    if n < 10:
        return "very low confidence"
    if n < 20:
        return "low confidence (promising but small sample)"
    if n < 50:
        return "usable sample"
    if n < 100:
        return "strong sample"
    return "robust sample"


def _pf_score(row: dict) -> float:
    """Normalized 0..1 profit-factor score, capped.

    Undefined PF (no losing trades) is treated as the cap *scaled by sample
    confidence* — so a flawless 6-trade run gets a good-but-bounded score,
    never the full mark, and never infinity.
    """
    pf = row.get("profit_factor")
    n = row.get("total_trades")
    if pf is None or pf == float("inf"):
        return 1.0 * trade_confidence(n)
    try:
        pf = float(pf)
    except (TypeError, ValueError):
        return 0.0
    if pf <= 0:
        return 0.0
    return min(pf, PF_CAP) / PF_CAP


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to 0..1. Constant vectors map to 0.5 (neutral)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _safe(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isinf(f) or math.isnan(f):
        return default
    return f


def score_strategies(rows: Sequence[dict]) -> list[dict]:
    """Score every rankable strategy and return them sorted best-first.

    Each returned dict is the ORIGINAL row plus:
        _score        final score (base * confidence multiplier), 0..1
        _base_score   pre-multiplier weighted score
        _confidence   trade_confidence value 0..1
        _confidence_label  human sample-size label
        _subscores    {metric: normalized 0..1} for transparency
        _reason       one-line "why it ranked here"
        _warnings     list[str] (low sample / overfit flags)

    Drops only error rows. If any strategy is profitable, unprofitable ones are
    excluded from the pool (we don't recommend a loser when a winner exists);
    if none are profitable, the full set is ranked so a winner still emerges.
    """
    valid = [r for r in rows if not r.get("error")]
    if not valid:
        return []

    profitable = [r for r in valid if _safe(r.get("total_return_pct")) > 0]
    pool = profitable if profitable else valid

    # Per-metric normalized sub-scores across the pool.
    exp_norm = _normalize([_safe(r.get("expectancy_pct")) for r in pool])
    # Lower drawdown is better → normalize the negative.
    dd_norm = _normalize([-_safe(r.get("max_drawdown_pct")) for r in pool])
    shp_norm = _normalize([_safe(r.get("sharpe_ratio")) for r in pool])

    scored: list[dict] = []
    for i, r in enumerate(pool):
        n_trades = r.get("total_trades")
        conf = trade_confidence(n_trades)
        pf_sub = _pf_score(r)

        subs = {
            "expectancy": round(exp_norm[i], 3),
            "profit_factor": round(pf_sub, 3),
            "drawdown": round(dd_norm[i], 3),
            "sharpe": round(shp_norm[i], 3),
            "trade_confidence": round(conf, 3),
        }
        base = (
            _W_EXPECTANCY * subs["expectancy"]
            + _W_PROFIT_FACTOR * subs["profit_factor"]
            + _W_DRAWDOWN * subs["drawdown"]
            + _W_SHARPE * subs["sharpe"]
            + _W_TRADE_CONF * subs["trade_confidence"]
        )
        mult = _CONF_MULT_FLOOR + (1.0 - _CONF_MULT_FLOOR) * (conf ** _CONF_EXPONENT)
        final = base * mult

        out = dict(r)
        out["_score"] = round(final, 4)
        out["_base_score"] = round(base, 4)
        out["_confidence"] = round(conf, 4)
        out["_confidence_label"] = confidence_label(n_trades)
        out["_subscores"] = subs
        out["_warnings"] = _build_warnings(r)
        out["_reason"] = ""  # filled after sorting (needs the leader board)
        scored.append(out)

    scored.sort(key=lambda r: r["_score"], reverse=True)

    # Reasons reference the dominant contributor for each strategy.
    for rank, r in enumerate(scored, start=1):
        r["_reason"] = _build_reason(r, rank, len(scored))
    return scored


def _build_warnings(row: dict) -> list[str]:
    w: list[str] = []
    n = int(row.get("total_trades") or 0)
    if n < 10:
        w.append(f"Very small sample ({n} trades) — result may not repeat.")
    elif n < 20:
        w.append(f"Small sample ({n} trades) — treat as promising, not proven.")
    wr = _safe(row.get("win_rate_pct"))
    pf = row.get("profit_factor")
    if wr >= 100.0 and n < 20:
        w.append("100% win rate on a small sample is a classic overfit signature.")
    if pf in (None, float("inf")) and n < 20:
        w.append("Profit factor is undefined (no losing trades) on a small sample.")
    dd = _safe(row.get("max_drawdown_pct"))
    if dd >= 40.0:
        w.append(f"Deep max drawdown ({dd:.0f}%) — capital-at-risk is high.")
    return w


def _build_reason(row: dict, rank: int, n: int) -> str:
    """One-line explanation of why this strategy landed where it did."""
    subs = row.get("_subscores", {})
    # Identify the strongest and weakest contributing metric.
    if subs:
        best_metric = max(subs, key=lambda k: subs[k])
        worst_metric = min(subs, key=lambda k: subs[k])
    else:
        best_metric = worst_metric = "—"
    label = row.get("_confidence_label", "")
    pieces = [
        f"Rank {rank}/{n} (score {row.get('_score', 0):.3f}).",
        f"Strongest on {best_metric.replace('_', ' ')}, weakest on {worst_metric.replace('_', ' ')}.",
        f"Sample: {label}.",
    ]
    if row.get("_confidence", 1.0) < 0.5:
        pieces.append("Score discounted for small sample.")
    return " ".join(pieces)


def pick_top_n(rows: Sequence[dict], n: int = 2) -> list[dict]:
    """Return the top-N scored strategies (best first), each annotated.

    Used by 'Promote Top N'. Returns fewer than N if the pool is smaller.
    """
    return score_strategies(rows)[: max(0, int(n))]


def pick_winner(rows: Sequence[dict]) -> dict | None:
    """Return the single best-scored row (annotated), or None.

    Backwards compatible: callers that only read total_return_pct /
    profit_factor / etc. still work because the annotated dict is a superset
    of the original row.
    """
    scored = score_strategies(rows)
    return scored[0] if scored else None
