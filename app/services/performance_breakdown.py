from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from app.services.performance_metrics import StrategyPerformance, calculate_performance_from_pairs


def bucket_atr_pct(atr_pct: float | None) -> str:
    if atr_pct is None:
        return "unknown"
    if atr_pct < 1.0:
        return "low"
    if atr_pct < 2.0:
        return "medium"
    return "high"


def r_bucket(r_multiple: float | None) -> str:
    if r_multiple is None:
        return "unknown"
    if r_multiple < 0:
        return "<0R"
    if r_multiple < 1:
        return "0–1R"
    if r_multiple < 2:
        return "1–2R"
    return ">=2R"


def _build_grouped_performance(
    trade_pairs: List[Dict[str, any]],
    group_key: str,
    strategy_name: str,
    symbol: str,
    period: str,
    initial_capital: float,
) -> Dict[str, StrategyPerformance]:
    groups: Dict[str, List[Dict[str, any]]] = defaultdict(list)
    for pair in trade_pairs:
        key = pair.get(group_key) or "unknown"
        groups[key].append(pair)

    return {
        key: calculate_performance_from_pairs(
            strategy_name=strategy_name,
            symbol=symbol,
            period=period,
            initial_capital=initial_capital,
            trade_pairs=group,
            equity_curve=None,
        )
        for key, group in groups.items()
    }


def breakdown_by_regime(
    trade_pairs: List[Dict[str, any]],
    strategy_name: str,
    symbol: str,
    period: str,
    initial_capital: float,
) -> Dict[str, StrategyPerformance]:
    return _build_grouped_performance(
        trade_pairs, "regime", strategy_name, symbol, period, initial_capital
    )


def breakdown_by_volatility(
    trade_pairs: List[Dict[str, any]],
    strategy_name: str,
    symbol: str,
    period: str,
    initial_capital: float,
) -> Dict[str, StrategyPerformance]:
    return _build_grouped_performance(
        trade_pairs, "volatility_bucket", strategy_name, symbol, period, initial_capital
    )


def breakdown_by_r_bucket(
    trade_pairs: List[Dict[str, any]],
    strategy_name: str,
    symbol: str,
    period: str,
    initial_capital: float,
) -> Dict[str, StrategyPerformance]:
    enhanced: List[Dict[str, any]] = []
    for pair in trade_pairs:
        bucket = r_bucket(pair.get("r_multiple"))
        pair_copy = pair.copy()
        pair_copy["r_bucket"] = bucket
        enhanced.append(pair_copy)
    return _build_grouped_performance(
        enhanced, "r_bucket", strategy_name, symbol, period, initial_capital
    )
