from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class StrategySignal:
    symbol: str
    direction: Literal["BUY", "SELL", "HOLD"]
    strength: float = 1.0  # 0.0–1.0
    price_at_signal: Optional[float] = None
    indicators: Dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    # ── Phase 0 scaffolding (optional, default-inert) ──
    # Carried so a rule can advertise an initial stop / target / confidence to
    # sizing and the exit-policy layer. No Phase 0 rule sets these; they default
    # to None/None/1.0 so existing signals and their serialization are unchanged.
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    confidence: float = 1.0


@dataclass
class StrategyConfig:
    """Config-driven strategy definition. Load from YAML or dict."""
    name: str
    symbol: str
    type: str  # "sma_rsi" | "ema_crossover" | "macd" | "bollinger"
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
