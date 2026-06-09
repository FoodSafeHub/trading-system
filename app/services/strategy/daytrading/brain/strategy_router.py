"""
Strategy Router — decides which strategies are allowed given the current market state.

Rules (from the spec, implemented as explicit logic):
  TREND_UP   → ORBBreakout, EMAMomentum, VWAPMeanReversion
  TREND_DOWN → EMAMomentum (short side), VolumeSpikeReversal, OpeningGapFade
  CHOPPY     → VWAPMeanReversion only (mean reversion excels in range-bound sessions)
  HIGH_VOL   → VolumeSpikeReversal only, at reduced size
  NEWS_RISK  → all strategies disabled (no new entries during unknown catalyst)

Each routing decision includes an explicit reason so the trader knows why
a strategy was enabled or blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.strategy.daytrading.brain.market_state import (
    CHOPPY, HIGH_VOL, NEWS_RISK, TREND_DOWN, TREND_UP, MarketStateResult,
)

# All known strategy names — must match strategy class .name attributes
ALL_STRATEGY_NAMES = [
    "ORBBreakout",
    "VWAPMeanReversion",
    "EMAMomentum",
    "OpeningGapFade",
    "VolumeSpikeReversal",
    "BollingerMomentum",
    "SupertrendTrend",
]

# Routing table: state → allowed strategy names
_ROUTING_TABLE: dict[str, list[str]] = {
    TREND_UP:   ["ORBBreakout", "EMAMomentum", "VWAPMeanReversion", "BollingerMomentum", "SupertrendTrend"],
    TREND_DOWN: ["EMAMomentum", "VolumeSpikeReversal", "OpeningGapFade", "BollingerMomentum", "SupertrendTrend"],
    CHOPPY:     ["VWAPMeanReversion", "BollingerMomentum", "ORBBreakout", "OpeningGapFade"],
    HIGH_VOL:   ["VolumeSpikeReversal", "ORBBreakout"],
    NEWS_RISK:  [],   # all blocked
    # UNKNOWN = regime classifier couldn't determine state (e.g. pre-market, insufficient bars).
    # Allow the same conservative set as CHOPPY so the bot can still trade instead of
    # blocking every signal until the market opens fully.
    "UNKNOWN":  ["VWAPMeanReversion", "BollingerMomentum", "ORBBreakout", "OpeningGapFade", "EMAMomentum"],
}

# Rationale messages for each blocking decision
_BLOCK_REASONS: dict[str, dict[str, str]] = {
    TREND_UP: {
        "OpeningGapFade":       "Gap fades work against trend — skip in TREND_UP.",
        "VolumeSpikeReversal":  "Volume spike reversals fade moves — skip in TREND_UP.",
    },
    TREND_DOWN: {
        "ORBBreakout":          "ORB breakout is a long-bias strategy — disabled in TREND_DOWN.",
        "VWAPMeanReversion":    "VWAP mean reversion (long-only) disabled in TREND_DOWN.",
    },
    CHOPPY: {
        "EMAMomentum":          "EMA momentum requires directional trend — disabled in CHOPPY.",
        "VolumeSpikeReversal":  "Volume spikes without trend context produce noisy signals.",
        "SupertrendTrend":      "Supertrend requires clear macro direction — disabled in CHOPPY.",
    },
    HIGH_VOL: {
        "VWAPMeanReversion":    "VWAP mean reversion fails when price trends hard away from VWAP.",
        "EMAMomentum":          "EMA crossovers are unreliable during spike volatility.",
        "OpeningGapFade":       "Gaps in HIGH_VOL sessions are often news-driven and don't fill.",
        "BollingerMomentum":    "BB squeeze signals are invalidated by HIGH_VOL expansion — too many false breakouts.",
        "SupertrendTrend":      "Supertrend flips rapidly in HIGH_VOL — direction unreliable.",
    },
    NEWS_RISK: {s: "NEWS_RISK detected — no new entries until catalyst is understood." for s in ALL_STRATEGY_NAMES},
}

_ALLOW_REASONS: dict[str, dict[str, str]] = {
    TREND_UP: {
        "ORBBreakout":         "ORB breakouts have highest follow-through in TREND_UP.",
        "EMAMomentum":         "EMA momentum thrives in directional uptrend sessions.",
        "VWAPMeanReversion":   "VWAP pullbacks offer lower-risk long entries in uptrends.",
        "BollingerMomentum":   "BB squeeze breakouts in a trend confirm directional momentum.",
        "SupertrendTrend":     "Supertrend pullbacks in TREND_UP are high-probability continuation trades.",
    },
    TREND_DOWN: {
        "EMAMomentum":         "EMA bearish crossover is valid in TREND_DOWN (short side).",
        "VolumeSpikeReversal": "Capitulation spikes in downtrends offer high-quality bounce trades.",
        "OpeningGapFade":      "Gap-down fades work well when downtrend is already in place.",
        "BollingerMomentum":   "BB breakdown shorts are valid in TREND_DOWN.",
        "SupertrendTrend":     "Supertrend bearish pullbacks align with the session downtrend.",
    },
    CHOPPY: {
        "VWAPMeanReversion":   "VWAP mean reversion is the primary strategy in range-bound sessions.",
        "BollingerMomentum":   "BB squeezes can signal directional resolution even in choppy conditions.",
        "ORBBreakout":         "ORB breakouts allowed in CHOPPY at reduced size — breakouts can resolve choppy range.",
        "OpeningGapFade":      "Gap fades are valid in CHOPPY — gaps often fill when there is no trend to sustain them.",
    },
    HIGH_VOL: {
        "VolumeSpikeReversal": "Volume spike reversals are specifically designed for elevated volatility.",
        "ORBBreakout":         "ORB breakouts in HIGH_VOL can produce large moves — allowed at reduced size.",
    },
    NEWS_RISK: {},
}


@dataclass
class RoutingDecision:
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    # Per-strategy explanation
    enabled_reasons: dict[str, str] = field(default_factory=dict)
    disabled_reasons: dict[str, str] = field(default_factory=dict)
    market_state: str = ""
    # Size multiplier applied to all strategies in this state
    size_multiplier: float = 1.0
    routing_summary: str = ""


def route_strategies(
    market_state_result: MarketStateResult,
    user_disabled: list[str] | None = None,
) -> RoutingDecision:
    """
    Given a MarketStateResult, return which strategies to allow/block
    and why, with per-strategy reasoning.

    user_disabled: strategies the user has manually turned off in config.
    """
    state = market_state_result.state
    allowed_names = _ROUTING_TABLE.get(state, [])
    user_off = set(user_disabled or [])

    enabled: list[str] = []
    disabled: list[str] = []
    enabled_reasons: dict[str, str] = {}
    disabled_reasons: dict[str, str] = {}

    for name in ALL_STRATEGY_NAMES:
        if name in user_off:
            disabled.append(name)
            disabled_reasons[name] = "Disabled by user in Config tab."
            continue

        if name in allowed_names:
            enabled.append(name)
            enabled_reasons[name] = _ALLOW_REASONS.get(state, {}).get(
                name, f"Allowed in {state} state."
            )
        else:
            disabled.append(name)
            disabled_reasons[name] = _BLOCK_REASONS.get(state, {}).get(
                name, f"Not allowed in {state} state."
            )

    # Size multiplier: reduce size in HIGH_VOL, eliminate in NEWS_RISK
    if state == NEWS_RISK:
        size_multiplier = 0.0
    elif state == HIGH_VOL:
        size_multiplier = 0.5   # half size — volatility cuts both ways
    elif state == CHOPPY:
        size_multiplier = 0.75  # reduced — lower conviction in choppy
    else:
        size_multiplier = 1.0

    # Low-confidence state gets a further size cut
    if market_state_result.confidence < 0.4:
        size_multiplier *= 0.8

    n_enabled = len(enabled)
    summary = (
        f"{state} (conf {market_state_result.confidence:.0%}) — "
        f"{n_enabled} of {len(ALL_STRATEGY_NAMES)} strategies active, "
        f"size {size_multiplier:.0%}."
    )

    return RoutingDecision(
        enabled=enabled,
        disabled=disabled,
        enabled_reasons=enabled_reasons,
        disabled_reasons=disabled_reasons,
        market_state=state,
        size_multiplier=round(size_multiplier, 2),
        routing_summary=summary,
    )
