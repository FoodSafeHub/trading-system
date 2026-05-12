from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StrategyPerformance:
    strategy_name: str
    symbol: str
    period: str
    initial_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    expectancy_r: Optional[float]
    profit_factor: Optional[float]
    total_return_pct: float
    cagr: Optional[float]
    max_drawdown_pct: Optional[float]
    sharpe_ratio: Optional[float]
    average_holding_days: float
    average_r_multiple: Optional[float]
    trade_pairs: List[Dict[str, Any]] = field(default_factory=list)


def calc_total_return(equity_curve: list[dict]) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e_initial = equity_curve[0]["equity"]
    e_final = equity_curve[-1]["equity"]
    if e_initial == 0:
        return 0.0
    return round((e_final / e_initial - 1) * 100, 2)


def calc_cagr(equity_curve: list[dict]) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e_initial = equity_curve[0]["equity"]
    e_final = equity_curve[-1]["equity"]
    if e_initial <= 0 or e_final <= 0:
        return 0.0
    years = len(equity_curve) / 252
    if years <= 0:
        return 0.0
    return round(((e_final / e_initial) ** (1.0 / years) - 1) * 100, 2)


def calc_max_drawdown(equity_curve: list[dict]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for point in equity_curve:
        equity = point.get("equity", 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return round(max_dd, 2)


def calc_daily_returns(equity_curve: list[dict]) -> List[float]:
    returns = []
    prev = None
    for point in equity_curve:
        equity = point.get("equity", 0.0)
        if prev is not None and prev > 0:
            returns.append((equity - prev) / prev)
        prev = equity
    return returns


def calc_sharpe_ratio(equity_curve: list[dict]) -> Optional[float]:
    returns = calc_daily_returns(equity_curve)
    if len(returns) < 2:
        return None
    avg = statistics.mean(returns)
    std = statistics.stdev(returns)
    if std <= 0:
        return None
    return round((avg / std) * (252 ** 0.5), 2)


def pair_trade_records(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    active_buy: Optional[Dict[str, Any]] = None

    for trade in trades:
        if trade.get("side") == "BUY":
            active_buy = trade.copy()
            active_buy["entry_date"] = trade.get("date")
            active_buy["entry_price"] = trade.get("price")
            active_buy["entry_stop"] = trade.get("stop")
            active_buy["entry_target"] = trade.get("target")
            active_buy["entry_regime"] = trade.get("regime")
            active_buy["entry_atr_pct"] = trade.get("atr_pct")
            active_buy["volatility_bucket"] = trade.get("volatility_bucket")
        elif active_buy is not None and trade.get("side", "").startswith("SELL"):
            exit_date = trade.get("date")
            entry_date = active_buy.get("entry_date")
            entry_price = active_buy.get("entry_price") or 0.0
            exit_price = trade.get("price") or 0.0
            pnl = trade.get("pnl") or 0.0
            risk_usd = active_buy.get("risk_usd") or 0.0
            r_multiple = round(pnl / risk_usd, 2) if risk_usd and risk_usd != 0 else None
            hold_days = 0
            try:
                exit_ts = pd.to_datetime(exit_date)
                entry_ts = pd.to_datetime(entry_date)
                hold_days = max(1, int((exit_ts - entry_ts).days))
            except Exception:
                hold_days = 0
            pairs.append({
                "strategy_name": active_buy.get("strategy_name"),
                "symbol": active_buy.get("symbol"),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": active_buy.get("quantity"),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / entry_price) * 100, 2) if entry_price else 0.0,
                "risk_usd": round(risk_usd, 2),
                "r_multiple": r_multiple,
                "hold_bars": hold_days,
                "regime": active_buy.get("entry_regime") or "unknown",
                "atr_pct": active_buy.get("entry_atr_pct"),
                "volatility_bucket": active_buy.get("volatility_bucket") or "unknown",
                "entry_stop": active_buy.get("entry_stop"),
                "entry_target": active_buy.get("entry_target"),
                "exit_reason": trade.get("side"),
                "reason": active_buy.get("reason"),
            })
            active_buy = None
    return pairs


def calculate_performance_from_pairs(
    strategy_name: str,
    symbol: str,
    period: str,
    initial_capital: float,
    trade_pairs: List[Dict[str, Any]],
    equity_curve: Optional[List[Dict[str, Any]]] = None,
) -> StrategyPerformance:
    total_trades = len(trade_pairs)
    wins = [t for t in trade_pairs if t["pnl"] > 0]
    losses = [t for t in trade_pairs if t["pnl"] <= 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0.0
    avg_win = round(statistics.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0.0
    avg_loss = round(statistics.mean([t["pnl_pct"] for t in losses]), 2) if losses else 0.0
    expectancy = round(statistics.mean([t["pnl_pct"] for t in trade_pairs]), 2) if trade_pairs else 0.0
    valid_r = [t["r_multiple"] for t in trade_pairs if t.get("r_multiple") is not None]
    expectancy_r = round(statistics.mean(valid_r), 2) if valid_r else None
    avg_r = round(statistics.mean(valid_r), 2) if valid_r else None
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None
    total_return = (
        calc_total_return(equity_curve)
        if equity_curve
        else round(sum(t["pnl"] for t in trade_pairs) / initial_capital * 100, 2)
    )
    cagr = calc_cagr(equity_curve) if equity_curve else None
    max_drawdown = calc_max_drawdown(equity_curve) if equity_curve else None
    sharpe = calc_sharpe_ratio(equity_curve) if equity_curve else None
    avg_hold = round(statistics.mean([t["hold_bars"] for t in trade_pairs]), 2) if trade_pairs else 0.0

    return StrategyPerformance(
        strategy_name=strategy_name,
        symbol=symbol,
        period=period,
        initial_capital=initial_capital,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=round(win_rate, 2),
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy_pct=expectancy,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        total_return_pct=total_return,
        cagr=cagr,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        average_holding_days=avg_hold,
        average_r_multiple=avg_r,
        trade_pairs=trade_pairs,
    )


def calculate_performance_from_backtest_result(result: Any) -> StrategyPerformance:
    trade_pairs = pair_trade_records(result.trades)
    return calculate_performance_from_pairs(
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        period=result.period,
        initial_capital=result.initial_capital,
        trade_pairs=trade_pairs,
        equity_curve=result.equity_curve,
    )
