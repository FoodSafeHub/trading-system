from __future__ import annotations

"""
Base class for Perplexity swing trading strategies.
Each strategy receives full OHLCV data (not just close) and returns a PerplexitySignal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import pandas as pd


@dataclass
class PerplexitySignal:
    symbol: str
    strategy_name: str
    direction: Literal["BUY", "SELL", "HOLD"]
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    confidence: float = 0.0          # 0.0–1.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    suitability_blocked: bool = False
    suitability_reason: Optional[str] = None
    volatility_bucket: Optional[str] = None


from app.services.market_regime import MarketRegime

class PerplexityStrategy:
    name: str = "base"
    enabled: bool = True

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: MarketRegime | None = None,
        volatility_bucket: Optional[str] = None,
        suitability_config: Optional[dict] = None,
    ) -> PerplexitySignal:
        raise NotImplementedError

    def _hold(
        self,
        symbol: str,
        reason: str = "",
        suitability_blocked: bool = False,
        suitability_reason: Optional[str] = None,
        volatility_bucket: Optional[str] = None,
    ) -> PerplexitySignal:
        return PerplexitySignal(
            symbol=symbol,
            strategy_name=self.name,
            direction="HOLD",
            reason=reason,
            suitability_blocked=suitability_blocked,
            suitability_reason=suitability_reason,
            volatility_bucket=volatility_bucket,
        )
