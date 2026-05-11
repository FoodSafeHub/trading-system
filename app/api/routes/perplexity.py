from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.market_data.provider import get_ohlcv
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES, run_perplexity_signal

router = APIRouter(prefix="/perplexity", tags=["perplexity"])

_STRATEGY_MAP = {s.name: s for s in PERPLEXITY_STRATEGIES}


@router.get("/strategies")
def list_strategies():
    return [{"name": s.name, "enabled": s.enabled} for s in PERPLEXITY_STRATEGIES]


@router.post("/strategies/{name}/toggle")
def toggle_strategy(name: str, enabled: bool = True):
    s = _STRATEGY_MAP.get(name)
    if not s:
        raise HTTPException(404, f"Strategy '{name}' not found")
    s.enabled = enabled
    return {"name": name, "enabled": s.enabled}


@router.get("/signals/{symbol}")
def get_signals(symbol: str):
    """
    Run all Perplexity strategies on the latest data for a symbol.
    BUY signals include position sizing based on account settings.
    """
    settings = get_settings()
    try:
        df = get_ohlcv(symbol.upper(), period="1y")
        if df.empty or len(df) < 60:
            raise HTTPException(400, f"Not enough data for {symbol} (need at least 60 bars)")
        signals = run_perplexity_signal(symbol.upper(), df)
        results = []
        for s in signals:
            item = {
                "strategy": s.strategy_name,
                "direction": s.direction,
                "entry_price": s.entry_price,
                "stop_price": s.stop_price,
                "target_price": s.target_price,
                "confidence": round(s.confidence, 2),
                "reason": s.reason,
                "indicators": s.indicators,
                "position_size": None,
            }
            # Attach position sizing for BUY signals that have a stop price
            if s.direction == "BUY" and s.entry_price and s.stop_price and settings.position_sizing_enabled:
                sz = calculate_position_size(
                    symbol=symbol.upper(),
                    entry_price=s.entry_price,
                    stop_price=s.stop_price,
                    account_value=settings.account_value,
                    risk_pct_per_trade=settings.risk_pct_per_trade,
                    max_position_size_usd=settings.max_position_size_usd,
                    max_account_risk_pct=settings.max_account_risk_pct,
                )
                item["position_size"] = {
                    "shares": sz.shares,
                    "position_value": sz.position_value,
                    "risk_amount": sz.risk_amount,
                    "risk_pct_of_account": sz.risk_pct_of_account,
                    "stop_distance": sz.stop_distance,
                    "stop_distance_pct": sz.stop_distance_pct,
                    "capped": sz.capped,
                    "cap_reason": sz.cap_reason,
                    "viable": sz.viable,
                    "skip_reason": sz.skip_reason,
                }
            results.append(item)
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/atr/{symbol}")
def get_atr_stops(symbol: str):
    """
    Return current price + ATR(14)-based stop suggestions for the position sizer.
    Provides 1×, 1.5×, and 2× ATR stop levels so the user doesn't have to guess.
    """
    try:
        df = get_ohlcv(symbol.upper(), period="3mo")
        if df.empty or len(df) < 15:
            raise HTTPException(400, f"Not enough data for {symbol}")

        high = df["High"]
        low  = df["Low"]
        close = df["Close"]

        # True Range = max of (H-L, |H-prevC|, |L-prevC|)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        current_price = float(close.iloc[-1])
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 4),
            "atr14": round(atr14, 4),
            "stop_1x_atr":   round(current_price - 1.0 * atr14, 4),
            "stop_1_5x_atr": round(current_price - 1.5 * atr14, 4),
            "stop_2x_atr":   round(current_price - 2.0 * atr14, 4),
            "stop_pct_1x":   round(1.0 * atr14 / current_price * 100, 2),
            "stop_pct_1_5x": round(1.5 * atr14 / current_price * 100, 2),
            "stop_pct_2x":   round(2.0 * atr14 / current_price * 100, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/size")
def calculate_size(
    symbol: str,
    entry_price: float,
    stop_price: float,
    account_value: Optional[float] = None,
    risk_pct: Optional[float] = None,
    max_position_size_usd: Optional[float] = None,
):
    """Calculate position size for a given entry/stop. Uses account settings by default."""
    settings = get_settings()
    sz = calculate_position_size(
        symbol=symbol.upper(),
        entry_price=entry_price,
        stop_price=stop_price,
        account_value=account_value or settings.account_value,
        risk_pct_per_trade=risk_pct or settings.risk_pct_per_trade,
        max_position_size_usd=max_position_size_usd or settings.max_position_size_usd,
        max_account_risk_pct=settings.max_account_risk_pct,
    )
    return {
        "symbol": sz.symbol,
        "entry_price": sz.entry_price,
        "stop_price": sz.stop_price,
        "shares": sz.shares,
        "position_value": sz.position_value,
        "risk_amount": sz.risk_amount,
        "risk_pct_of_account": sz.risk_pct_of_account,
        "stop_distance": sz.stop_distance,
        "stop_distance_pct": sz.stop_distance_pct,
        "capped": sz.capped,
        "cap_reason": sz.cap_reason,
        "viable": sz.viable,
        "skip_reason": sz.skip_reason,
    }


@router.get("/backtest/{strategy_name}/{symbol}")
def backtest(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
):
    """Run a single Perplexity strategy backtest."""
    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    try:
        result = run_perplexity_backtest(strategy, symbol.upper(), period, initial_capital)
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
            "profit_factor": result.profit_factor,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/backtest-all/{symbol}")
def backtest_all(
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
):
    """Run all 5 Perplexity strategies on one symbol and return a comparison table."""
    results = []
    for strategy in PERPLEXITY_STRATEGIES:
        try:
            r = run_perplexity_backtest(strategy, symbol.upper(), period, initial_capital)
            results.append({
                "strategy_name": r.strategy_name,
                "total_trades": r.total_trades,
                "win_rate_pct": r.win_rate_pct,
                "profit_factor": r.profit_factor,
                "total_return_pct": r.total_return_pct,
                "total_pnl": r.total_pnl,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
            })
        except Exception as exc:
            results.append({"strategy_name": strategy.name, "error": str(exc)})
    return results
