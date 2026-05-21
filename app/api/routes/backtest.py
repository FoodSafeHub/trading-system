from __future__ import annotations

from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.backtest.engine import run_backtest
from app.services.backtest.consensus_engine import run_consensus_backtest
from app.services.strategy.engine import load_strategies_from_config

router = APIRouter(prefix="/backtest", tags=["backtest"])

# New strategy type names — used for consensus defaults
_NEW_STRATEGY_TYPES = {
    "rsi2_mean_reversion",
    "ema_macd_crossover",
    "bb_squeeze_breakout",
    "pullback_ema50",
    "vix_spike_reversal",
}

# Symbols that use the 5 new strategies by default in consensus mode
_NEW_STRATEGY_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "SPY"}


@router.get("/run/{strategy_name}")
def backtest_strategy(
    strategy_name: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
    quantity: float = 0.0,
    symbol: Optional[str] = None,
):
    """
    Run a backtest for a named strategy from strategies.json.

    symbol  : override the symbol configured in strategies.json (optional)
    period  : 1mo, 3mo, 6mo, 1y, 2y
    """
    configs = load_strategies_from_config()
    config = next((c for c in configs if c.name == strategy_name), None)
    if not config:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    # Allow caller to override the symbol
    effective_symbol = symbol.upper() if symbol else config.symbol

    try:
        result = run_backtest(
            strategy_name=config.name,
            symbol=effective_symbol,
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
    new_only: bool = True,
):
    """
    Run a consensus backtest for a symbol.

    new_only=True (default): uses only the 5 new strategies for AAPL/MSFT/GOOGL/SPY.
    new_only=False          : uses all configured strategies for the symbol.
    """
    sym = symbol.upper()
    configs = load_strategies_from_config()
    symbol_configs = [c for c in configs if c.symbol == sym]
    if not symbol_configs:
        raise HTTPException(status_code=404, detail=f"No strategies configured for symbol '{symbol}'")

    # For canonical symbols, filter to new strategies only by default
    if new_only and sym in _NEW_STRATEGY_SYMBOLS:
        new_configs = [c for c in symbol_configs if c.type in _NEW_STRATEGY_TYPES]
        if new_configs:
            symbol_configs = new_configs

    try:
        result = run_consensus_backtest(sym, symbol_configs, min_agreement, period, initial_capital)
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
def list_consensus_symbols(new_only: bool = False):
    """
    List symbols that have 2+ strategies configured.

    new_only=True : only counts the 5 new strategy types.
    """
    configs = load_strategies_from_config()
    by_symbol: dict = defaultdict(list)
    for c in configs:
        if new_only and c.type not in _NEW_STRATEGY_TYPES:
            continue
        by_symbol[c.symbol].append(c.name)
    return [
        {"symbol": sym, "strategy_count": len(names), "strategies": names}
        for sym, names in sorted(by_symbol.items())
        if len(names) >= 2
    ]


@router.get("/custom-consensus/{symbol}")
def backtest_custom_consensus(
    symbol: str,
    min_agreement: int = 2,
    period: str = "1y",
    initial_capital: float = 100000.0,
):
    """Consensus backtest for an arbitrary symbol using all 5 new strategy types.

    This is what the scanner runs under the hood, so a candidate flagged on the
    scanner page (e.g. BRK-B_Pullback_EMA50) can be re-tested here with the same
    factory defaults. Falls back to the regular /consensus endpoint when the
    symbol has explicit configs in strategies.json.
    """
    from app.services.scanner.scanner_service import _make_generic_configs
    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")
    configs = _make_generic_configs(sym)
    try:
        result = run_consensus_backtest(sym, configs, min_agreement, period, initial_capital)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
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


@router.get("/strategies")
def list_backtest_strategies(new_only: bool = False):
    """
    List all strategies available for backtesting.

    new_only=True : only the 5 new strategies.
    """
    configs = load_strategies_from_config()
    if new_only:
        configs = [c for c in configs if c.type in _NEW_STRATEGY_TYPES]
    return [{"name": c.name, "symbol": c.symbol, "type": c.type} for c in configs]
