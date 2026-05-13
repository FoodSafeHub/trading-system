from trading_bot.strategies.rsi2_mean_reversion import RSI2MeanReversion
from trading_bot.strategies.ema_macd_crossover import EMAMACDCrossover
from trading_bot.strategies.bb_squeeze_breakout import BBSqueezeBreakout
from trading_bot.strategies.pullback_ema50 import PullbackEMA50
from trading_bot.strategies.vix_spike_reversal import VIXSpikeReversal

NEW_STRATEGIES = [
    RSI2MeanReversion(),
    EMAMACDCrossover(),
    BBSqueezeBreakout(),
    PullbackEMA50(),
    VIXSpikeReversal(),
]

__all__ = [
    "RSI2MeanReversion",
    "EMAMACDCrossover",
    "BBSqueezeBreakout",
    "PullbackEMA50",
    "VIXSpikeReversal",
    "NEW_STRATEGIES",
]
