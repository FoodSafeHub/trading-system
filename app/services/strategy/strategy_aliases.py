from __future__ import annotations

"""
Strategy-type alias map — Phase 0 scaffolding (inert until Phase 1).

When the consolidated daily strategies land in Phase 1, several new type names
replace one or more old ones. This map lets a NEW type inherit the per-symbol
calibration (scanner_profiles.json, keyed ``old_type:SYMBOL``) saved under the
OLD name, so tuning isn't lost across the rename.

Contract:
  * Keys are NEW (Phase 1) type names; values are the OLD types they replace,
    PRIMARY first (the one whose calibration best matches the new behaviour).
  * The map is new->old ONLY. Never map an old type to another old type — that
    would let a stale profile leak into a still-live strategy.
  * In Phase 0 nothing references a new type, so every lookup resolves to the
    queried type itself and behaviour is unchanged.
"""

from typing import Dict, List

STRATEGY_ALIASES: Dict[str, List[str]] = {
    "rsi2_reversion":    ["rsi2_mean_reversion"],
    "trend_pullback":    ["pullback_ema50", "fib_pullback"],
    "squeeze_breakout":  ["bb_squeeze_breakout"],
    "momentum_breakout": ["ema_macd_crossover", "breakout"],
    "panic_reversal":    ["vix_spike_reversal"],
    "trend_follow":      ["supertrend", "ema_ribbon"],
    "rs_rotation":       [],  # net-new; no predecessor calibration
}

# Reverse index (old -> new), built once. Used by canonical() for Phase 2 cutover.
_OLD_TO_NEW: Dict[str, str] = {
    old: new for new, olds in STRATEGY_ALIASES.items() for old in olds
}


def resolve_alias(strategy_type: str) -> List[str]:
    """Return [strategy_type] + its predecessor old types (self first).

    For an existing (old) type this is just ``[strategy_type]`` — so callers
    that pass live types get unchanged behaviour. For a new type it appends the
    old names to try as calibration fallbacks, primary-first.
    """
    return [strategy_type] + STRATEGY_ALIASES.get(strategy_type, [])


def canonical(strategy_type: str) -> str:
    """Map an old type to its new canonical name; pass through anything that is
    already canonical or unknown. (Reserved for the Phase 2 cutover; harmless
    in Phase 0.)"""
    if strategy_type in STRATEGY_ALIASES:
        return strategy_type
    return _OLD_TO_NEW.get(strategy_type, strategy_type)
