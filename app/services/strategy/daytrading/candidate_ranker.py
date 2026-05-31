"""
candidate_ranker.py — Mover-aware candidate ranking for the day-trading engine.

Role
----
Sits between the mover snapshot and the strategy engine:

    MarketMoverSnapshot
        ↓  merge_duplicate_symbols()
    {symbol: SymbolMoverContext}
        ↓  CandidateRanker.rank()
    list[RankedCandidate]  (sorted, hard-filtered)
        ↓  passed to NativeStrategyEntry / DayTradingScanner
    DayTradeSignal (only if a strategy also fires)

Critical invariant
------------------
A mover appearance NEVER triggers a trade. The ranker only decides:
  1. Which symbols are worth running strategy.generate_signals() on.
  2. In what priority order they should be evaluated.
  3. Whether a symbol has failed hard filters and should be skipped entirely.

Integration with existing scanner
----------------------------------
The ranker is designed to plug into DayTradingScanner.get_intraday_watchlist():

    ranker = CandidateRanker(config=CandidateRankerConfig())
    contexts = merge_duplicate_symbols(snapshot)
    ranked = ranker.rank(symbols, contexts, market_state="TREND_UP")
    # Pass [r.symbol for r in ranked if r.eligible] to the scanner's watchlist

All config knobs live in CandidateRankerConfig.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.strategy.daytrading.market_movers import (
    LIST_TYPE_DIRECTION,
    SymbolMoverContext,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class CandidateRankerConfig:
    """All ranking thresholds and weights — override at construction time."""

    # ── Hard filters (reject outright if any fail) ────────────────────────────
    min_rvol: float = 0.50              # relative volume vs 30d avg; 0 = skip check
    min_dollar_volume_5m: float = 500_000.0   # min $ volume in last 5m bar; 0 = skip
    max_spread_pct: float = 0.50        # max bid-ask spread % (0 = skip check)
    min_price: float = 2.0              # reject sub-penny / micro-cap junk
    max_atr_pct: float = 15.0           # reject blow-ups (daily ATR > 15% of price)
    min_atr_pct: float = 0.50           # reject dead names

    # Mover-list minimum rank to admit non-mover symbols via fallback
    fallback_non_mover_threshold: float = 0.40  # min total_score for non-movers to qualify

    # ── Mover list weights (contribution to attention_score) ──────────────────
    # Each weight is the proportion of attention_score (0–1) credited when a symbol
    # appears at rank 1 on that list. Actual contribution scales with rank.
    mover_list_weights: dict[str, float] = field(default_factory=lambda: {
        "top_gainers":    0.90,
        "top_losers":     0.90,
        "gap_up":         0.85,
        "gap_down":       0.85,
        "unusual_volume": 0.80,
        "most_active":    0.70,
    })

    # ── Score component weights (must sum to 1.0) ─────────────────────────────
    weight_attention:  float = 0.30    # mover-list prominence
    weight_liquidity:  float = 0.25    # dollar volume + RVOL
    weight_context:    float = 0.20    # regime fit + strategy bucket alignment
    weight_catalyst:   float = 0.15    # news / earnings presence
    weight_penalty:    float = 0.10    # deductions for spread, correlation, chop

    # ── Catalyst bonus ────────────────────────────────────────────────────────
    catalyst_bonus: float = 0.08       # added to catalyst_score when has_news=True

    # ── Regime penalties (subtracted from context_score) ─────────────────────
    # Applied when the symbol's strategy bucket doesn't fit the current regime.
    regime_mismatch_penalty: float = 0.20

    # ── CHOPPY regime controls ────────────────────────────────────────────────
    choppy_allowed_buckets: list[str] = field(
        default_factory=lambda: ["VWAP", "gap"]
    )
    choppy_size_multiplier: float = 0.50  # signal to caller to cut size

    # ── Correlation / concentration penalty ──────────────────────────────────
    max_correlated_positions: int = 2     # if open_positions has >= this many
                                          # symbols from the same sector, add penalty
    correlation_penalty: float = 0.15

    # ── Strategy bucket → compatible list types ───────────────────────────────
    # Used to boost context_score when the mover list aligns with the bucket.
    bucket_list_alignment: dict[str, list[str]] = field(default_factory=lambda: {
        "ORB":      ["top_gainers", "gap_up", "unusual_volume"],
        "gap":      ["gap_up", "gap_down", "top_gainers", "top_losers"],
        "VWAP":     ["most_active", "top_losers", "gap_down"],
        "momentum": ["top_gainers", "unusual_volume", "gap_up"],
    })

    # ── Logging verbosity ─────────────────────────────────────────────────────
    log_all_scores: bool = False          # True = log every evaluated symbol
    log_rejected: bool = True             # True = log rejected symbols with reason


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class RankedCandidate:
    """
    Fully scored and annotated candidate ready for strategy evaluation.

    The ranker produces one of these per symbol it evaluates. Only candidates
    with eligible=True should be passed to strategy.generate_signals().
    """
    symbol: str

    # ── Sub-scores (each 0.0–1.0 before weighting) ───────────────────────────
    attention_score:  float = 0.0    # mover-list prominence
    liquidity_score:  float = 0.0    # dollar volume + RVOL quality
    context_score:    float = 0.0    # regime fit + list-type/bucket alignment
    catalyst_score:   float = 0.0    # news / catalyst bonus
    penalty_score:    float = 0.0    # deductions (spread, correlation, chop)

    # ── Composite ─────────────────────────────────────────────────────────────
    total_score: float = 0.0

    # ── Strategy routing hints ────────────────────────────────────────────────
    directional_bias: str = "neutral"   # "long" | "short" | "neutral"
    strategy_hints: list[str] = field(default_factory=list)
    recommended_bucket: str = ""        # "ORB" | "gap" | "VWAP" | "momentum" | ""

    # ── Gate results ──────────────────────────────────────────────────────────
    eligible: bool = False              # passed all hard filters + score threshold
    reject_reasons: list[str] = field(default_factory=list)
    is_mover: bool = False              # appeared on at least one mover list
    list_memberships: list[str] = field(default_factory=list)
    mover_prominence_score: float = 0.0

    # ── Size hint ─────────────────────────────────────────────────────────────
    size_multiplier: float = 1.0        # 1.0 = full size; <1.0 = reduce in CHOPPY etc.

    # ── Diagnostics (for logging / UI) ────────────────────────────────────────
    market_state: str = ""
    open_position_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":               self.symbol,
            "total_score":          round(self.total_score, 4),
            "attention_score":      round(self.attention_score, 4),
            "liquidity_score":      round(self.liquidity_score, 4),
            "context_score":        round(self.context_score, 4),
            "catalyst_score":       round(self.catalyst_score, 4),
            "penalty_score":        round(self.penalty_score, 4),
            "eligible":             self.eligible,
            "reject_reasons":       list(self.reject_reasons),
            "is_mover":             self.is_mover,
            "list_memberships":     list(self.list_memberships),
            "mover_prominence_score": round(self.mover_prominence_score, 4),
            "directional_bias":     self.directional_bias,
            "strategy_hints":       list(self.strategy_hints),
            "recommended_bucket":   self.recommended_bucket,
            "size_multiplier":      round(self.size_multiplier, 2),
            "market_state":         self.market_state,
        }


# ── Ranker ────────────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    Ranks a list of symbols using mover context + live market metrics.

    Usage
    -----
        config = CandidateRankerConfig()
        ranker = CandidateRanker(config)

        contexts = merge_duplicate_symbols(snapshot)   # from market_movers
        ranked = ranker.rank(
            symbols=["AAPL", "NVDA", "SPY"],
            mover_contexts=contexts,
            market_state="TREND_UP",
            open_positions=[{"symbol": "TSLA", "sector": "Technology"}],
            live_metrics={"AAPL": {"rvol": 2.1, "dollar_vol_5m": 1_200_000}},
        )
        eligible = [r for r in ranked if r.eligible]
    """

    def __init__(self, config: CandidateRankerConfig | None = None) -> None:
        self.config = config or CandidateRankerConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def rank(
        self,
        symbols: list[str],
        mover_contexts: dict[str, SymbolMoverContext],
        *,
        market_state: str = "UNKNOWN",
        open_positions: list[dict] | None = None,
        live_metrics: dict[str, dict] | None = None,
        scanner_results: dict[str, dict] | None = None,
    ) -> list[RankedCandidate]:
        """
        Score and rank a list of symbols.

        Parameters
        ----------
        symbols
            Candidate symbols to evaluate (strings, will be upper-cased).
        mover_contexts
            Output of merge_duplicate_symbols(snapshot). Symbols not present
            here are treated as non-movers.
        market_state
            Current regime from the brain: TREND_UP / TREND_DOWN / CHOPPY /
            HIGH_VOL / NEWS_RISK / UNKNOWN.
        open_positions
            List of dicts with at least {"symbol": str, "sector": str} keys.
            Used for the correlation penalty.
        live_metrics
            Optional per-symbol dict with live intraday data:
              {"rvol": float, "dollar_vol_5m": float, "spread_pct": float,
               "atr_pct": float, "vwap_dist_pct": float, "orb_broken": bool}
            Values from this dict take priority over mover_context values.
        scanner_results
            Optional per-symbol dict from DayTradingScanner.score_symbol().
            Keys: {"score", "adjusted_score", "tags", "recommended_strategy_bucket",
                   "metrics": {"avg_daily_volume_30d", "atr_pct", ...}}

        Returns
        -------
        list[RankedCandidate]
            Sorted descending by total_score. Non-eligible candidates are included
            at the bottom (eligible=False) for diagnostics.
        """
        cfg = self.config
        positions = open_positions or []
        metrics_map = live_metrics or {}
        scanner_map = scanner_results or {}

        candidates: list[RankedCandidate] = []

        for raw_sym in symbols:
            sym = raw_sym.upper().strip()
            ctx = mover_contexts.get(sym)
            live = metrics_map.get(sym, {})
            scan = scanner_map.get(sym, {})

            cand = self._score_symbol(
                sym, ctx, live, scan, market_state, positions
            )
            candidates.append(cand)

            if cfg.log_all_scores:
                logger.info(
                    "[ranker] %s score=%.3f att=%.2f liq=%.2f ctx=%.2f cat=%.2f "
                    "pen=%.2f eligible=%s bias=%s lists=%s",
                    sym, cand.total_score, cand.attention_score, cand.liquidity_score,
                    cand.context_score, cand.catalyst_score, cand.penalty_score,
                    cand.eligible, cand.directional_bias, cand.list_memberships,
                )
            elif not cand.eligible and cfg.log_rejected:
                logger.debug(
                    "[ranker] REJECTED %s (score=%.3f): %s",
                    sym, cand.total_score, " | ".join(cand.reject_reasons),
                )

        # Eligible first (by score desc), then ineligible (by score desc)
        candidates.sort(key=lambda c: (int(c.eligible), c.total_score), reverse=True)
        return candidates

    def score_one(
        self,
        symbol: str,
        mover_contexts: dict[str, SymbolMoverContext],
        *,
        market_state: str = "UNKNOWN",
        open_positions: list[dict] | None = None,
        live_metrics: dict | None = None,
        scanner_result: dict | None = None,
    ) -> RankedCandidate:
        """Score a single symbol without sorting. Useful for debugging."""
        return self._score_symbol(
            symbol.upper().strip(),
            mover_contexts.get(symbol.upper().strip()),
            live_metrics or {},
            scanner_result or {},
            market_state,
            open_positions or [],
        )

    # ── Internal scoring ───────────────────────────────────────────────────────

    def _score_symbol(
        self,
        sym: str,
        ctx: SymbolMoverContext | None,
        live: dict,
        scan: dict,
        market_state: str,
        open_positions: list[dict],
    ) -> RankedCandidate:
        cfg = self.config
        rejects: list[str] = []

        # ── Pull metrics (live > mover context > scanner) ──────────────────────
        rvol         = _coalesce(live.get("rvol"), ctx.rvol if ctx else 0.0, 0.0)
        dollar_vol_5m = _coalesce(live.get("dollar_vol_5m"), ctx.dollar_volume if ctx else 0.0, 0.0)
        spread_pct   = _coalesce(live.get("spread_pct"), ctx.spread_pct if ctx else 0.0, 0.0)
        price        = _coalesce(live.get("price"), ctx.price if ctx else 0.0, 0.0)
        atr_pct      = _coalesce(
            live.get("atr_pct"),
            _nested(scan, "metrics", "atr_pct"),
            0.0,
        )

        # ── Hard filters ───────────────────────────────────────────────────────
        if price > 0 and price < cfg.min_price:
            rejects.append(f"price {price:.2f} < min {cfg.min_price}")

        if cfg.min_rvol > 0 and rvol > 0 and rvol < cfg.min_rvol:
            rejects.append(f"rvol {rvol:.2f} < min {cfg.min_rvol}")

        if cfg.min_dollar_volume_5m > 0 and dollar_vol_5m > 0 and dollar_vol_5m < cfg.min_dollar_volume_5m:
            rejects.append(
                f"dollar_vol_5m ${dollar_vol_5m/1e6:.2f}M < min ${cfg.min_dollar_volume_5m/1e6:.2f}M"
            )

        if cfg.max_spread_pct > 0 and spread_pct > cfg.max_spread_pct:
            rejects.append(f"spread {spread_pct:.3f}% > max {cfg.max_spread_pct}%")

        if atr_pct > 0:
            if atr_pct > cfg.max_atr_pct:
                rejects.append(f"atr_pct {atr_pct:.1f}% > max {cfg.max_atr_pct}%")
            if atr_pct < cfg.min_atr_pct:
                rejects.append(f"atr_pct {atr_pct:.1f}% < min {cfg.min_atr_pct}%")

        # NEWS_RISK regime — do not route any candidates
        if market_state == "NEWS_RISK":
            rejects.append("regime NEWS_RISK — no new candidates")

        # ── Attention score: mover-list prominence ─────────────────────────────
        attention_score = 0.0
        if ctx is not None and ctx.is_mover:
            # Weighted average of each list's contribution (rank-discounted weight)
            total_weight = 0.0
            weighted_sum = 0.0
            for lt, rank in ctx.best_rank_per_list.items():
                list_weight = cfg.mover_list_weights.get(lt, 0.70)
                rank_factor = max(0.0, 1.0 - (rank - 1) / 50.0)
                contribution = list_weight * rank_factor
                weighted_sum += contribution
                total_weight += list_weight
            attention_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        # Non-movers get attention_score = 0.0 (they can still qualify via total_score)

        # ── Liquidity score ───────────────────────────────────────────────────
        # RVOL component: log-scaled, floor at 0.5×, ceiling at 5×
        rvol_score = 0.0
        if rvol > 0:
            rvol_score = min(1.0, max(0.0, math.log(rvol + 0.1) / math.log(5.1)))

        # Dollar volume 5m component: log-scaled, $250K floor, $5M ceiling
        dvol_score = 0.0
        if dollar_vol_5m > 0:
            dvol_score = min(1.0, max(0.0,
                math.log10(max(dollar_vol_5m, 1) / 250_000) / math.log10(20.0)
            ))

        # Scanner liquidity score if available
        scan_liq = float(_nested(scan, "score") or 0.0)

        liquidity_score = max(
            0.5 * rvol_score + 0.5 * dvol_score,
            0.7 * scan_liq if scan_liq > 0 else 0.0,
        )
        liquidity_score = min(1.0, max(0.0, liquidity_score))

        # ── Context score: regime fit + list-bucket alignment ─────────────────
        context_score = 0.5   # baseline: neutral

        # Bucket from scanner or mover hints
        recommended_bucket = (
            str(scan.get("recommended_strategy_bucket", ""))
            or _infer_bucket_from_hints(ctx.strategy_hints if ctx else [])
        )

        # Regime fit
        regime_fit = _regime_bucket_fit(market_state, recommended_bucket)
        if regime_fit < 0:
            context_score -= cfg.regime_mismatch_penalty
            rejects_tmp = f"bucket '{recommended_bucket}' mismatches regime {market_state}"
            # Not a hard reject — add to context penalty but still score
        else:
            context_score += regime_fit * 0.40

        # CHOPPY: only allow certain buckets
        if market_state == "CHOPPY" and recommended_bucket:
            if recommended_bucket not in cfg.choppy_allowed_buckets:
                rejects.append(
                    f"CHOPPY regime: bucket '{recommended_bucket}' not in "
                    f"{cfg.choppy_allowed_buckets}"
                )

        # List-type / bucket alignment bonus
        if ctx and ctx.list_memberships and recommended_bucket:
            aligned_lists = cfg.bucket_list_alignment.get(recommended_bucket, [])
            matches = sum(1 for lt in ctx.list_memberships if lt in aligned_lists)
            if matches:
                context_score += 0.15 * min(matches, 2)   # up to +0.30

        context_score = min(1.0, max(0.0, context_score))

        # ── Catalyst score ────────────────────────────────────────────────────
        catalyst_score = 0.0
        if ctx and ctx.has_news:
            catalyst_score = cfg.catalyst_bonus
        # Earnings = both a catalyst AND a hard risk — add bonus but also note risk
        if ctx and "earnings_nearby" in ctx.news_tags:
            catalyst_score = min(1.0, catalyst_score + 0.05)
            # Also warn — earnings can cause gaps that destroy the strategy edge
            logger.debug("[ranker] %s: earnings_nearby — verify gap-fade rules apply", sym)

        # ── Penalty score: deductions ─────────────────────────────────────────
        penalty_score = 0.0

        # Wide spread penalty (only when max_spread_pct is configured)
        if cfg.max_spread_pct > 0 and spread_pct > cfg.max_spread_pct * 0.5:
            ratio = min(spread_pct / cfg.max_spread_pct, 1.0)
            penalty_score += 0.10 * ratio

        # Correlation / concentration penalty
        if open_positions:
            sym_sector = _find_sector(sym, open_positions)
            same_sector_count = sum(
                1 for p in open_positions if p.get("sector") == sym_sector and sym_sector
            )
            if same_sector_count >= cfg.max_correlated_positions:
                penalty_score += cfg.correlation_penalty
                rejects.append(
                    f"sector concentration ({same_sector_count} open in sector '{sym_sector}')"
                )

        # CHOPPY size discount (not a reject, but signals caller to reduce size)
        size_mult = 1.0
        if market_state == "CHOPPY":
            size_mult = cfg.choppy_size_multiplier

        penalty_score = min(1.0, max(0.0, penalty_score))

        # ── Total score ───────────────────────────────────────────────────────
        total = (
            cfg.weight_attention * attention_score
            + cfg.weight_liquidity  * liquidity_score
            + cfg.weight_context    * context_score
            + cfg.weight_catalyst   * catalyst_score
            - cfg.weight_penalty    * penalty_score
        )
        total = round(min(1.0, max(0.0, total)), 4)

        # ── Non-mover fallback gate ───────────────────────────────────────────
        if (ctx is None or not ctx.is_mover) and total < cfg.fallback_non_mover_threshold:
            rejects.append(
                f"non-mover total_score {total:.3f} < fallback threshold "
                f"{cfg.fallback_non_mover_threshold}"
            )

        eligible = len(rejects) == 0

        return RankedCandidate(
            symbol=sym,
            attention_score=round(attention_score, 4),
            liquidity_score=round(liquidity_score, 4),
            context_score=round(context_score, 4),
            catalyst_score=round(catalyst_score, 4),
            penalty_score=round(penalty_score, 4),
            total_score=total,
            directional_bias=ctx.directional_bias if ctx else "neutral",
            strategy_hints=list(ctx.strategy_hints) if ctx else [],
            recommended_bucket=recommended_bucket,
            eligible=eligible,
            reject_reasons=rejects,
            is_mover=bool(ctx and ctx.is_mover),
            list_memberships=list(ctx.list_memberships) if ctx else [],
            mover_prominence_score=ctx.mover_prominence_score if ctx else 0.0,
            size_multiplier=round(size_mult, 2),
            market_state=market_state,
            open_position_count=len(open_positions),
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _coalesce(*values: Any) -> Any:
    """Return the first non-None, non-zero value, or the last argument."""
    for v in values[:-1]:
        if v is not None and v != 0.0 and v != 0:
            return v
    return values[-1]


def _nested(d: dict, *keys: str) -> Any:
    """Safe nested dict lookup: _nested(d, "a", "b") == d.get("a", {}).get("b")."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _infer_bucket_from_hints(hints: list[str]) -> str:
    """Map strategy name hints to a scanner bucket string."""
    _HINT_TO_BUCKET: dict[str, str] = {
        "ORBBreakout":      "ORB",
        "NRSqueezeBreakout": "ORB",
        "OpeningGapFade":   "gap",
        "VWAPMeanReversion": "VWAP",
        "EMAMomentum":      "momentum",
        "SupertrendTrend":  "momentum",
    }
    for h in hints:
        b = _HINT_TO_BUCKET.get(h)
        if b:
            return b
    return ""


def _regime_bucket_fit(state: str, bucket: str) -> float:
    """
    Regime-bucket alignment score in [-0.5, +0.5].

    Positive = good fit; negative = bad fit; 0 = neutral / unknown.
    """
    _FIT: dict[str, dict[str, float]] = {
        "TREND_UP":   {"ORB": 0.4, "gap": 0.3, "momentum": 0.3, "VWAP": -0.1},
        "TREND_DOWN": {"ORB": -0.2, "gap": 0.3, "momentum": 0.1, "VWAP": 0.3},
        "CHOPPY":     {"ORB": -0.4, "gap": 0.1, "momentum": -0.2, "VWAP": 0.4},
        "HIGH_VOL":   {"ORB": -0.1, "gap": -0.1, "momentum": 0.1, "VWAP": 0.3},
        "NEWS_RISK":  {"ORB": -0.5, "gap": -0.3, "momentum": -0.3, "VWAP": -0.1},
    }
    return _FIT.get(state, {}).get(bucket, 0.0)


def _find_sector(symbol: str, open_positions: list[dict]) -> str:
    """Return sector for symbol from open_positions, or empty string."""
    for p in open_positions:
        if p.get("symbol", "").upper() == symbol.upper():
            return str(p.get("sector", ""))
    return ""


# ── Convenience factory ───────────────────────────────────────────────────────

def build_candidate_ranker(
    min_rvol: float = 0.50,
    min_dollar_volume_5m: float = 500_000.0,
    max_spread_pct: float = 0.50,
    fallback_non_mover_threshold: float = 0.40,
    log_all_scores: bool = False,
    **kwargs: Any,
) -> CandidateRanker:
    """
    Convenience constructor — pass only the knobs you want to override.

    All other settings come from CandidateRankerConfig defaults.
    """
    cfg = CandidateRankerConfig(
        min_rvol=min_rvol,
        min_dollar_volume_5m=min_dollar_volume_5m,
        max_spread_pct=max_spread_pct,
        fallback_non_mover_threshold=fallback_non_mover_threshold,
        log_all_scores=log_all_scores,
        **kwargs,
    )
    return CandidateRanker(cfg)
