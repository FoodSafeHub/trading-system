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
    # When True, the strategy code is kept and remains importable / runnable
    # in backtests, but the LIVE runner (`run_perplexity_signal`) skips it.
    # Used by the decision pass to hold strategies for further research
    # without deleting their implementation. See:
    #   reports/perplexity_strategy_decisions.md
    research_only: bool = False

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


class CeeiGatedStrategy:
    """Transparent wrapper adding the CEEI entry gate to any PerplexityStrategy.

    Used by the backtest route so a gated backtest exercises the exact same
    veto the live scheduler applies (post-signal, BUY-only, fail-open). Every
    attribute other than run() delegates to the wrapped strategy, so the
    backtest engine sees name/config/enabled etc. unchanged.

    gate_params is the same dict the assignment-level gate consumes:
      {"ceei_gate": "setup"|"trigger"|"score",
       "ceei_gate_threshold": ..., "ceei_gate_lookback": ...}
    """

    def __init__(self, inner: "PerplexityStrategy", gate_params: dict):
        self._inner = inner
        self._gate_params = dict(gate_params)

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def run(self, symbol: str, df: pd.DataFrame, *args, **kwargs) -> PerplexitySignal:
        sig = self._inner.run(symbol, df, *args, **kwargs)
        if sig.direction == "BUY" and df is not None and not df.empty:
            # Imported here to avoid a circular import (rules imports this module
            # via the adapter chain).
            from app.services.strategy.rules import apply_ceei_gate
            sig = apply_ceei_gate(sig, df["Close"].dropna(), df, self._gate_params)
        return sig
