from __future__ import annotations

"""
Rule-backed Perplexity adapter — Phase 1 (behind a flag, default OFF).

When ``settings.use_unified_perplexity`` is enabled, the Perplexity runner swaps
its bespoke strategy classes for these thin adapters, which delegate to the
unified rule functions in ``rules.py`` and map the resulting StrategySignal to a
PerplexitySignal for the scanner / Perplexity page / recommendations. This is
how Engine B becomes a display layer over the single engine of record — WITHOUT
deleting the old classes yet. Default-off means scanner votes are byte-unchanged.
"""

from typing import Optional

import pandas as pd

from app.services.market_regime import MarketRegime
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy
from app.services.strategy.rules import evaluate_strategy


class RuleBackedPerplexityStrategy(PerplexityStrategy):
    """Adapts a unified rules.py strategy type to the PerplexityStrategy API."""

    def __init__(self, strategy_type: str, name: str, params: Optional[dict] = None):
        self.strategy_type = strategy_type
        self.name = name
        self.params = params or {}
        self.enabled = True

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: MarketRegime | None = None,
        volatility_bucket: Optional[str] = None,
        suitability_config: Optional[dict] = None,
    ) -> PerplexitySignal:
        if df is None or df.empty or "Close" not in df:
            return self._hold(symbol, "no data", volatility_bucket=volatility_bucket)
        try:
            sig = evaluate_strategy(self.strategy_type, symbol, df["Close"].dropna(),
                                    self.params, ohlcv=df)
        except Exception as exc:  # degrade like the runner does
            return self._hold(symbol, f"error: {exc}", volatility_bucket=volatility_bucket)

        if sig.direction == "HOLD":
            return self._hold(symbol, "no signal", volatility_bucket=volatility_bucket)
        return PerplexitySignal(
            symbol=symbol,
            strategy_name=self.name,
            direction=sig.direction,
            entry_price=sig.price_at_signal,
            stop_price=sig.stop_price,
            target_price=sig.target_price,
            confidence=float(sig.confidence) if sig.confidence is not None else 0.0,
            indicators=sig.indicators,
            reason=f"{self.strategy_type}: {sig.direction}",
            volatility_bucket=volatility_bucket,
        )


# Display names mirror the unified set. Used only when the flag is ON.
UNIFIED_ADAPTER_SPECS = [
    ("rsi2_reversion",    "Unified_RSI2_Reversion"),
    ("trend_pullback",    "Unified_Trend_Pullback"),
    ("squeeze_breakout",  "Unified_Squeeze_Breakout"),
    ("momentum_breakout", "Unified_Momentum_Breakout"),
    ("panic_reversal",    "Unified_Panic_Reversal"),
    ("trend_follow",      "Unified_Trend_Follow"),
]


def build_unified_adapters() -> list[RuleBackedPerplexityStrategy]:
    return [RuleBackedPerplexityStrategy(st, name) for st, name in UNIFIED_ADAPTER_SPECS]
