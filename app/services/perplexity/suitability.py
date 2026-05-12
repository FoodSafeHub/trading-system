from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.performance_metrics import StrategyPerformance

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "perplexity_suitability.json"


@dataclass
class StrategySuitabilityConfig:
    allowed_regimes: List[str] = field(default_factory=list)
    blocked_regimes: List[str] = field(default_factory=list)
    allowed_symbols: List[str] = field(default_factory=list)
    blocked_symbols: List[str] = field(default_factory=list)
    allowed_volatility_buckets: List[str] = field(default_factory=list)
    blocked_volatility_buckets: List[str] = field(default_factory=list)


def _normalize_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).lower() for v in value if v is not None]
    return []


def load_suitability_config(path: Path | str | None = None) -> Dict[str, StrategySuitabilityConfig]:
    config_file = Path(path) if path else CONFIG_PATH
    if not config_file.exists():
        return {}
    raw = json.loads(config_file.read_text())
    normalized: Dict[str, StrategySuitabilityConfig] = {}
    for strategy_name, values in raw.items():
        normalized[strategy_name] = StrategySuitabilityConfig(
            allowed_regimes=_normalize_value(values.get("allowed_regimes", [])),
            blocked_regimes=_normalize_value(values.get("blocked_regimes", [])),
            allowed_symbols=[s.upper() for s in _normalize_value(values.get("allowed_symbols", []))],
            blocked_symbols=[s.upper() for s in _normalize_value(values.get("blocked_symbols", []))],
            allowed_volatility_buckets=_normalize_value(values.get("allowed_volatility_buckets", [])),
            blocked_volatility_buckets=_normalize_value(values.get("blocked_volatility_buckets", [])),
        )
    return normalized


def save_suitability_config(config: Dict[str, Any], path: Path | str | None = None) -> None:
    config_file = Path(path) if path else CONFIG_PATH
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))


def is_strategy_suitable(
    strategy_name: str,
    symbol: str,
    regime: str | None,
    volatility_bucket: str | None,
    config: Dict[str, StrategySuitabilityConfig] | None,
) -> Tuple[bool, str]:
    if not config:
        return True, ""
    strategy_cfg = config.get(strategy_name)
    if not strategy_cfg:
        return True, ""

    regime_value = regime.lower() if regime else ""
    volatility_value = volatility_bucket.lower() if volatility_bucket else ""
    symbol_value = symbol.upper() if symbol else ""

    if strategy_cfg.allowed_regimes and regime_value not in strategy_cfg.allowed_regimes:
        return False, f"regime '{regime_value}' not allowed"
    if regime_value in strategy_cfg.blocked_regimes:
        return False, f"regime '{regime_value}' blocked"
    if strategy_cfg.allowed_symbols and symbol_value not in strategy_cfg.allowed_symbols:
        return False, f"symbol '{symbol_value}' not allowed"
    if symbol_value in strategy_cfg.blocked_symbols:
        return False, f"symbol '{symbol_value}' blocked"
    if strategy_cfg.allowed_volatility_buckets and volatility_value not in strategy_cfg.allowed_volatility_buckets:
        return False, f"volatility bucket '{volatility_value}' not allowed"
    if volatility_value in strategy_cfg.blocked_volatility_buckets:
        return False, f"volatility bucket '{volatility_value}' blocked"
    return True, ""


def suggest_blocked_conditions(
    breakdowns: Dict[str, StrategyPerformance],
    min_trades: int = 6,
    profit_factor_threshold: float = 1.2,
    expectancy_threshold: float = 0.0,
) -> List[str]:
    suggestions: List[str] = []
    for key, perf in breakdowns.items():
        if perf.total_trades < min_trades:
            continue
        if perf.profit_factor is not None and perf.profit_factor < profit_factor_threshold:
            suggestions.append(
                f"Consider blocking {key} for {perf.strategy_name} — PF={perf.profit_factor:.2f}, trades={perf.total_trades}"
            )
            continue
        if perf.expectancy_pct < expectancy_threshold:
            suggestions.append(
                f"Consider blocking {key} for {perf.strategy_name} — expectancy {perf.expectancy_pct:.2f}%"
            )
    return suggestions
