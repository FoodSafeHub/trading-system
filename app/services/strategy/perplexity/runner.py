from __future__ import annotations

"""
Runner for Perplexity strategies.
Evaluates all enabled strategies against live or historical OHLCV data.
"""

from typing import List

import pandas as pd

from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy
from app.services.strategy.perplexity.strategies import (
    BollingerReversionUptrend,
    BollingerSqueezeBreakout,
    EmaPullbackSupport,
    HighVolumeMomentumBreakout,
    MacdRsiMomentum,
)

PERPLEXITY_STRATEGIES: List[PerplexityStrategy] = [
    HighVolumeMomentumBreakout(),
    BollingerSqueezeBreakout(),
    MacdRsiMomentum(),
    EmaPullbackSupport(),
    BollingerReversionUptrend(),
]


def run_perplexity_signal(symbol: str, df: pd.DataFrame) -> List[PerplexitySignal]:
    """Run all enabled Perplexity strategies on the given OHLCV DataFrame."""
    results = []
    for strategy in PERPLEXITY_STRATEGIES:
        if not strategy.enabled:
            continue
        try:
            sig = strategy.run(symbol, df)
            results.append(sig)
        except Exception as exc:
            results.append(PerplexitySignal(
                symbol=symbol,
                strategy_name=strategy.name,
                direction="HOLD",
                reason=f"error: {exc}",
            ))
    return results
