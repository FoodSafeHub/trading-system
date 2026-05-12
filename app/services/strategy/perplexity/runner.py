from __future__ import annotations

"""
Runner for Perplexity strategies.
Evaluates all enabled strategies against live or historical OHLCV data.
"""

from typing import List

import pandas as pd

from app.services.market_regime import MarketRegime, get_current_regime
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy
from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend,
    MaCrossoverRsi,
    BreakoutConsolidation,
    BollingerMeanReversionUptrend,
    FibPullbackSupport,
)

PERPLEXITY_STRATEGIES: List[PerplexityStrategy] = [
    EmaMeanReversionUptrend(),
    MaCrossoverRsi(),
    BreakoutConsolidation(),
    BollingerMeanReversionUptrend(),
    FibPullbackSupport(),
]


def run_perplexity_signal(
    symbol: str,
    df: pd.DataFrame,
    regime: MarketRegime | None = None,
) -> List[PerplexitySignal]:
    """Run all enabled Perplexity strategies on the given OHLCV DataFrame."""
    results = []
    if regime is None and not df.empty:
        regime = get_current_regime(df.index[-1])
    regime = regime or MarketRegime.BULL

    for strategy in PERPLEXITY_STRATEGIES:
        if not strategy.enabled:
            continue
        try:
            sig = strategy.run(symbol, df, regime=regime)
            results.append(sig)
        except Exception as exc:
            results.append(PerplexitySignal(
                symbol=symbol,
                strategy_name=strategy.name,
                direction="HOLD",
                reason=f"error: {exc}",
            ))
    return results
