"""
Day-trading strategy registry.

Active strategies (6 core):
    ORBBreakout, VWAPMeanReversion, EMAMomentum, OpeningGapFade,
    SupertrendTrend, NRSqueezeBreakout

Retired / disabled strategies (kept for import compatibility, not in ALL_STRATEGIES):
    VolumeSpikeReversal  → volume spike is now a confirmation filter inside other strategies
    BollingerMomentum    → merged into NRSqueezeBreakout
    EngulfingVolumeSurge → overlaps with EMAMomentum and SupertrendTrend in trending regimes
    NarrowRangeBreakout  → merged into NRSqueezeBreakout
    ThreeBarPush         → structural overlap with SupertrendTrend; lower R
    HammerShootingStar   → weaker form of VWAPMeanReversion without VWAP anchor

To re-enable a retired strategy: add it back to ALL_STRATEGIES below.
"""
from .orb_breakout import ORBBreakout
from .vwap_mean_reversion import VWAPMeanReversion
from .ema_momentum import EMAMomentum
from .opening_gap_fade import OpeningGapFade
from .supertrend_trend import SupertrendTrend
from .nr_squeeze_breakout import NRSqueezeBreakout

# Retired — imported for backward-compat with any code that references them by name
from .volume_spike_reversal import VolumeSpikeReversal
from .bollinger_momentum import BollingerMomentum
from .momentum_patterns import (
    EngulfingVolumeSurge,
    NarrowRangeBreakout,
    ThreeBarPush,
    HammerShootingStar,
)

# ── Active strategy set ────────────────────────────────────────────────────────
ALL_STRATEGIES = [
    ORBBreakout(),
    VWAPMeanReversion(),
    EMAMomentum(),
    OpeningGapFade(),
    SupertrendTrend(),
    NRSqueezeBreakout(),
]

STRATEGY_MAP = {s.name: s for s in ALL_STRATEGIES}

# ── Retired set (useful for backtesting comparisons) ──────────────────────────
RETIRED_STRATEGIES = [
    VolumeSpikeReversal(),
    BollingerMomentum(),
    EngulfingVolumeSurge(),
    NarrowRangeBreakout(),
    ThreeBarPush(),
    HammerShootingStar(),
]
