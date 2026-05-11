from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.backtest.engine import run_backtest
from app.services.backtest.consensus_engine import run_consensus_backtest
from app.services.strategy.engine import load_strategies_from_config

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/run/{strategy_name}")
def backtest_strategy(
    strategy_name: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
    quantity: float = 0.0,
):
    """
    Run a backtest for a named strategy from strategies.json.
    period: 1mo, 3mo, 6mo, 1y, 2y
    """
    configs = load_strategies_from_config()
    config = next((c for c in configs if c.name == strategy_name), None)
    if not config:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    try:
        result = run_backtest(
            strategy_name=config.name,
            symbol=config.symbol,
            strategy_type=config.type,
            params=config.params,
            period=period,
            initial_capital=initial_capital,
            quantity=quantity,
        )
        return {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "period": result.period,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_pnl": result.total_pnl,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": [
                {"date": t.date, "side": t.side, "price": t.price,
                 "quantity": t.quantity, "value": round(t.value, 2)}
                for t in result.trades
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/consensus/{symbol}")
def backtest_consensus(
    symbol: str,
    min_agreement: int = 2,
    period: str = "1y",
    initial_capital: float = 100000.0,
):
    """
    Run a consensus backtest for a symbol using all its configured strategies.
    Only trades when min_agreement strategies agree on the same direction.
    """
    configs = load_strategies_from_config()
    symbol_configs = [c for c in configs if c.symbol == symbol.upper()]
    if not symbol_configs:
        raise HTTPException(status_code=404, detail=f"No strategies configured for symbol '{symbol}'")

    try:
        result = run_consensus_backtest(symbol.upper(), symbol_configs, min_agreement, period, initial_capital)
        return {
            "symbol": result.symbol,
            "period": result.period,
            "min_agreement": result.min_agreement,
            "strategies_used": result.strategies_used,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_pnl": result.total_pnl,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/consensus-symbols")
def list_consensus_symbols():
    """List symbols that have 2+ strategies configured (eligible for consensus backtest)."""
    configs = load_strategies_from_config()
    from collections import defaultdict
    by_symbol: dict = defaultdict(list)
    for c in configs:
        by_symbol[c.symbol].append(c.name)
    return [
        {"symbol": sym, "strategy_count": len(names), "strategies": names}
        for sym, names in sorted(by_symbol.items())
    ]


@router.get("/strategies")
def list_backtest_strategies():
    """List all strategies available for backtesting."""
    configs = load_strategies_from_config()
    return [{"name": c.name, "symbol": c.symbol, "type": c.type} for c in configs]
