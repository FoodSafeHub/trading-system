"""
Strategy Selector — routes a symbol to the strategies that actually work for it.

Combines symbol profile, market regime, and strategy routing logic to produce
an ordered recommendation with explicit human-readable reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.strategy.daytrading.brain.symbol_profiles import SymbolProfile

ALL_STRATEGY_NAMES = [
    "ORBBreakout",
    "VWAPMeanReversion",
    "EMAMomentum",
    "OpeningGapFade",
    "VolumeSpikeReversal",
]

# Market regime → strategies that are structurally enabled
_REGIME_ALLOWED: dict[str, list[str]] = {
    "BULL_OPEN":  ["ORBBreakout", "EMAMomentum", "VWAPMeanReversion", "OpeningGapFade", "VolumeSpikeReversal"],
    "BEAR_OPEN":  ["EMAMomentum", "VolumeSpikeReversal", "OpeningGapFade"],
    "CHOPPY":     ["VWAPMeanReversion", "EMAMomentum"],
}


@dataclass
class SelectionResult:
    primary: str
    secondary: str | None
    enabled: list[str]               # ordered best → reasonable for this context
    disabled: list[str]              # strategies that should NOT be used
    disabled_reasons: dict[str, str] # strategy → why
    recommendation_text: str         # one-paragraph human summary
    symbol_fit_scores: dict[str, float]  # 0–1 fit score per strategy


class StrategySelector:
    """
    Given a symbol profile and market regime, rank all strategies and
    explain which ones fit this context.
    """

    @staticmethod
    def select(
        symbol: str,
        regime: str,
        profile: SymbolProfile,
    ) -> SelectionResult:
        regime_key = regime if regime in _REGIME_ALLOWED else "BULL_OPEN"
        regime_allowed = set(_REGIME_ALLOWED[regime_key])

        scores: dict[str, float] = {}
        disabled_reasons: dict[str, str] = {}

        for strat in ALL_STRATEGY_NAMES:
            score = _base_score(strat, profile)

            # Penalise if regime blocks it
            if strat not in regime_allowed:
                disabled_reasons[strat] = f"Not suitable in {regime} regime."
                scores[strat] = 0.0
                continue

            # Hard disable: strategy explicitly in symbol's avoid list
            if strat in profile.avoid_strategies:
                disabled_reasons[strat] = _avoid_reason(strat, profile)
                scores[strat] = max(0.0, score * 0.2)   # very low but not zero (user can override)
                continue

            # Boost if strategy is in symbol's best list
            rank = profile.best_strategies.index(strat) if strat in profile.best_strategies else len(ALL_STRATEGY_NAMES)
            boost = [0.3, 0.15, 0.05][min(rank, 2)] if rank < 3 else 0.0
            scores[strat] = min(1.0, score + boost)

        # Sort by score descending
        ranked = sorted(ALL_STRATEGY_NAMES, key=lambda s: scores[s], reverse=True)
        enabled = [s for s in ranked if scores[s] > 0.25]
        disabled = [s for s in ranked if scores[s] <= 0.25]

        # Ensure avoid_strategies that weren't already blocked get a reason
        for strat in profile.avoid_strategies:
            if strat not in disabled_reasons:
                disabled_reasons[strat] = _avoid_reason(strat, profile)

        primary = enabled[0] if enabled else "EMAMomentum"
        secondary = enabled[1] if len(enabled) > 1 else None

        recommendation = _build_recommendation(symbol, primary, secondary, profile, regime, scores)

        return SelectionResult(
            primary=primary,
            secondary=secondary,
            enabled=enabled,
            disabled=disabled,
            disabled_reasons=disabled_reasons,
            recommendation_text=recommendation,
            symbol_fit_scores={s: round(scores[s], 2) for s in ALL_STRATEGY_NAMES},
        )


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _base_score(strategy: str, profile: SymbolProfile) -> float:
    """Base fit score 0–1 for a strategy given symbol characteristics."""
    vol = profile.volatility_pct
    trend = profile.trend_strength

    if strategy == "ORBBreakout":
        # Needs volatility for a wide, tradeable range + trending tendency
        vol_score = min(1.0, (vol - 0.5) / 1.5) if vol > 0.5 else 0.0
        return 0.5 * vol_score + 0.5 * trend

    if strategy == "VWAPMeanReversion":
        # Thrives on low-vol, stable names that revert to mean
        vol_score = max(0.0, 1.0 - (vol - 0.5) / 2.0)
        return 0.6 * vol_score + 0.4 * (1.0 - trend)

    if strategy == "EMAMomentum":
        # Works across the board — slight preference for trending names
        return 0.5 + 0.3 * trend

    if strategy == "OpeningGapFade":
        # Needs reasonable gap frequency
        gap_score = min(1.0, profile.gap_frequency_pct / 10.0)
        return 0.5 * gap_score + 0.3 * profile.liquidity_score

    if strategy == "VolumeSpikeReversal":
        # Better on high-vol names with real capitulation spikes
        vol_score = min(1.0, vol / 1.8)
        return 0.5 * vol_score + 0.3 * profile.liquidity_score

    return 0.5


def _avoid_reason(strategy: str, profile: SymbolProfile) -> str:
    sym = profile.symbol
    vol = profile.volatility_pct

    if strategy == "ORBBreakout":
        return (
            f"ORB Breakout is not recommended for {sym}. "
            f"With {vol:.1f}% ATR, the 15-min opening range is too narrow to produce "
            f"valid, non-fakeout breakouts. Expected setups per 60 days: 0–2."
        )
    if strategy == "VWAPMeanReversion":
        return (
            f"VWAP Mean Reversion is not recommended for {sym}. "
            f"High volatility ({vol:.1f}% ATR) means price overshoots VWAP frequently "
            f"and doesn't revert cleanly — stop-outs are common."
        )
    if strategy == "VolumeSpikeReversal":
        return (
            f"Volume Spike Reversal is unreliable for {sym}. "
            f"Stable names don't produce the extreme volume capitulation spikes this strategy needs."
        )
    return f"{strategy} is not a good fit for {sym}'s current volatility profile."


def _build_recommendation(
    symbol: str,
    primary: str,
    secondary: str | None,
    profile: SymbolProfile,
    regime: str,
    scores: dict[str, float],
) -> str:
    top_score = scores.get(primary, 0)
    lines = [
        f"**{symbol}** ({profile.market_type}) in **{regime}** regime.",
        f"",
        f"**Best fit:** {primary} (fit score {top_score:.0%})",
    ]
    if secondary:
        lines.append(f"**Also viable:** {secondary} (fit score {scores.get(secondary, 0):.0%})")
    lines += ["", profile.notes]
    return "\n".join(lines)
