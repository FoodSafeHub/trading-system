from .orb_breakout import ORBBreakout
from .vwap_mean_reversion import VWAPMeanReversion
from .ema_momentum import EMAMomentum
from .opening_gap_fade import OpeningGapFade
from .volume_spike_reversal import VolumeSpikeReversal
from .bollinger_momentum import BollingerMomentum
from .supertrend_trend import SupertrendTrend

ALL_STRATEGIES = [
    ORBBreakout(),
    VWAPMeanReversion(),
    EMAMomentum(),
    OpeningGapFade(),
    VolumeSpikeReversal(),
    BollingerMomentum(),
    SupertrendTrend(),
]

STRATEGY_MAP = {s.name: s for s in ALL_STRATEGIES}
