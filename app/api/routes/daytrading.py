from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.strategy.daytrading.autotrader import (
    AutoTraderConfig,
    get_manager,
)
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


# ── Auto-trader switch ────────────────────────────────────────────────────────
# Flip ON to have the system day-trade automatically against a list of symbols
# using the active intraday strategies. Each symbol gets its own SingleStockTrader
# (polls 1m/5m/15m bars, enters/exits with the risk governor active, flattens at
# 3:45 PM ET). The whole switch auto-stops at 4:00 PM ET.


class AutoTraderStartRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Tickers to trade, e.g. ['TSLA','NVDA']")
    direction_mode: Literal["long_only", "short_only", "both"] = "long_only"
    trail_mode: Literal["ema", "atr", "candle"] = "atr"
    partial_tp: bool = True
    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.05, description="Fraction of capital risked per trade")
    max_daily_loss_pct: float = Field(2.0, gt=0, le=20.0)
    max_trades_per_day: int = Field(6, ge=1, le=50)
    max_consecutive_losses: int = Field(3, ge=1, le=10)
    initial_capital: float = Field(10_000.0, gt=0)
    broker_name: Literal["paper", "alpaca"] = "paper"
    force: bool = Field(False, description="Arm even if outside regular trading hours")


@router.post("/autotrader/start")
def autotrader_start(req: AutoTraderStartRequest) -> dict[str, Any]:
    """Flip the day-trading switch ON. Spins up one trader per symbol."""
    cfg = AutoTraderConfig(
        symbols=req.symbols,
        direction_mode=req.direction_mode,
        trail_mode=req.trail_mode,
        partial_tp=req.partial_tp,
        risk_per_trade_pct=req.risk_per_trade_pct,
        max_daily_loss_pct=req.max_daily_loss_pct,
        max_trades_per_day=req.max_trades_per_day,
        max_consecutive_losses=req.max_consecutive_losses,
        initial_capital=req.initial_capital,
        broker_name=req.broker_name,
    )
    try:
        return get_manager().flip_on(cfg, force=req.force)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/autotrader/stop")
def autotrader_stop(
    flatten: bool = Query(False, description="Close open positions at market before stopping"),
) -> dict[str, Any]:
    """Flip the day-trading switch OFF. Pass flatten=true to also close open positions."""
    return get_manager().flip_off(flatten=flatten)


@router.get("/autotrader/status")
def autotrader_status() -> dict[str, Any]:
    """Snapshot of the switch, config, and every active trader."""
    return get_manager().status()


@router.post("/autotrader/flatten")
def autotrader_flatten(
    symbol: str | None = Query(None, description="Symbol to flatten; omit to flatten ALL"),
) -> dict[str, Any]:
    """Force-close one or all open positions without stopping the loop."""
    return {"flattened": get_manager().flatten(symbol)}
