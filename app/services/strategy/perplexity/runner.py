from __future__ import annotations

"""
Runner for Perplexity strategies.
Evaluates all enabled strategies against live or historical OHLCV data.
"""

from typing import List

import pandas as pd

from app.services.market_regime import MarketRegime, get_current_regime
from app.services.performance_breakdown import bucket_atr_pct
from app.services.perplexity.suitability import is_strategy_suitable, load_suitability_config
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy
from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend,
    MaCrossoverRsi,
    BreakoutConsolidation,
    BollingerMeanReversionUptrend,
    FibPullbackSupport,
    RsiSwingReversal,
    SupertrendSwing,
    BollingerBandBreakout,
)

PERPLEXITY_STRATEGIES: List[PerplexityStrategy] = [
    EmaMeanReversionUptrend(),
    MaCrossoverRsi(),
    BreakoutConsolidation(),
    BollingerMeanReversionUptrend(),
    FibPullbackSupport(),
    RsiSwingReversal(),
    SupertrendSwing(),
    BollingerBandBreakout(),
]


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _current_atr(df: pd.DataFrame, period: int = 14) -> float:
    atr = _atr_series(df, period)
    v = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(df["Close"].iloc[-1]) * 0.02
    return v


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

    suitability_config = None
    try:
        suitability_config = load_suitability_config()
    except Exception:
        suitability_config = None

    volatility_bucket = bucket_atr_pct(_current_atr(df)) if not df.empty else "unknown"

    for strategy in PERPLEXITY_STRATEGIES:
        if not strategy.enabled:
            continue
        try:
            sig = strategy.run(
                symbol,
                df,
                regime=regime,
                volatility_bucket=volatility_bucket,
                suitability_config=suitability_config,
            )
            if sig.direction == "BUY" and suitability_config is not None:
                allowed, reason = is_strategy_suitable(
                    strategy.name,
                    symbol,
                    regime.value if regime else None,
                    volatility_bucket,
                    suitability_config,
                )
                if not allowed:
                    sig = PerplexitySignal(
                        symbol=symbol,
                        strategy_name=strategy.name,
                        direction="HOLD",
                        reason=f"Blocked by suitability: {reason}",
                        suitability_blocked=True,
                        suitability_reason=reason,
                        volatility_bucket=volatility_bucket,
                    )
            results.append(sig)
        except Exception as exc:
            results.append(PerplexitySignal(
                symbol=symbol,
                strategy_name=strategy.name,
                direction="HOLD",
                reason=f"error: {exc}",
                volatility_bucket=volatility_bucket,
            ))
    return results
