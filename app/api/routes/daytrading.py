from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.strategy.daytrading.market_open import (
    get_spy_regime,
    market_status,
)
from app.services.strategy.daytrading.runner import (
    get_session_symbols,
    run_backtest,
    run_backtest_all,
    run_scan,
    run_signals,
)
from app.services.strategy.daytrading.validation.walkforward import run_walkforward
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP

router = APIRouter(tags=["daytrading"])

# simple in-memory toggle store (resets on restart)
_disabled: set[str] = set()


@router.get("/signals/{symbol}")
def get_signals(symbol: str) -> dict[str, Any]:
    enabled = [s.name for s in ALL_STRATEGIES if s.name not in _disabled]
    return run_signals(symbol.upper(), enabled_strategies=enabled)


@router.get("/strategies")
def list_strategies() -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "timeframe": s.timeframe,
            "config": s.default_config,
            "enabled": s.name not in _disabled,
        }
        for s in ALL_STRATEGIES
    ]


@router.get("/backtest/{strategy}/{symbol}")
def backtest_strategy(
    strategy: str,
    symbol: str,
    period: str = Query("60d", description="60d | 90d | 730d"),
    initial_capital: float = Query(10_000.0),
    position_pct: float = Query(0.95),
) -> dict[str, Any]:
    if strategy not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    return run_backtest(
        symbol.upper(), strategy, period, initial_capital, position_pct
    )


@router.get("/backtest-all/{symbol}")
def backtest_all(
    symbol: str,
    period: str = Query("60d"),
    initial_capital: float = Query(10_000.0),
) -> list[dict[str, Any]]:
    return run_backtest_all(symbol.upper(), period, initial_capital)


@router.get("/scan")
def scan(
    symbols: str = Query("SPY,QQQ,AAPL,TSLA,NVDA"),
    strategies: str = Query("all"),
) -> list[dict[str, Any]]:
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    enabled = None if strategies == "all" else [s.strip() for s in strategies.split(",")]
    return run_scan(sym_list, enabled)


@router.get("/market-status")
def get_market_status() -> dict[str, Any]:
    status = market_status()
    regime, spy_vs_vwap, gap_pct, gap_type = get_spy_regime()
    return {
        **status,
        "regime": regime,
        "spy_vs_vwap_pct": spy_vs_vwap,
        "spy_gap_pct": gap_pct,
        "spy_gap_type": gap_type,
    }


@router.get("/scanner/watchlist")
def scanner_watchlist(
    max_symbols: int = Query(20, description="Max symbols to return"),
    universe: str = Query("", description="Comma-separated override universe; empty = default"),
    market_state: str = Query("", description="Force a market state (TREND_UP etc.); empty = auto"),
) -> list[dict[str, Any]]:
    """Pre-market scanner: ranked watchlist with scores, tags, and strategy buckets."""
    from app.services.strategy.daytrading.scanners import DayTradingScanner, DayTradingScannerConfig
    sym_list = [s.strip().upper() for s in universe.split(",") if s.strip()] or None
    scanner = DayTradingScanner(config=DayTradingScannerConfig(), universe=sym_list)
    state = market_state.strip() or None
    results = scanner.get_intraday_watchlist(max_symbols=max_symbols, market_state=state)
    return [r.to_dict() for r in results]


@router.get("/scanner/metrics/{symbol}")
def scanner_symbol_metrics(symbol: str) -> dict[str, Any]:
    """Fetch raw scan metrics for a single symbol."""
    from app.services.strategy.daytrading.scanners import DayTradingScanner, DayTradingScannerConfig
    scanner = DayTradingScanner(config=DayTradingScannerConfig())
    metrics = scanner.fetch_metrics(symbol.upper())
    result = scanner.score_symbol(metrics)
    return result.to_dict()


@router.get("/backtest/walkforward/{strategy}/{symbol}")
def walkforward_backtest(
    strategy: str,
    symbol: str,
    years: int = Query(2, description="Years of history (max 2 due to 5m data limits)"),
    step_months: int = Query(3, description="Window step size in months"),
    initial_capital: float = Query(10_000.0),
    position_pct: float = Query(0.95),
) -> dict[str, Any]:
    """Walk-forward validation: IS/OOS windows with WFE scoring."""
    if strategy not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    result = run_walkforward(
        symbol=symbol.upper(),
        strategy_name=strategy,
        years=years,
        step_months=step_months,
        initial_capital=initial_capital,
        position_pct=position_pct,
    )
    return result.to_dict()


@router.post("/strategies/{name}/toggle")
def toggle_strategy(name: str, enabled: bool = Query(...)) -> dict[str, Any]:
    if name not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    if enabled:
        _disabled.discard(name)
    else:
        _disabled.add(name)
    return {"name": name, "enabled": enabled}
