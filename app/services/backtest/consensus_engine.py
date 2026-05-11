from __future__ import annotations

"""
Consensus backtesting engine.

Runs multiple strategies on the same symbol simultaneously.
A trade is only placed when min_agreement strategies agree on the same direction.
This mirrors exactly how the live scheduler works.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional
import statistics

from app.services.backtest.engine import BacktestTrade
from app.services.market_data.provider import get_ohlcv
from app.services.strategy.rules import evaluate_strategy
from app.services.strategy.models import StrategyConfig


@dataclass
class ConsensusBacktestResult:
    symbol: str
    period: str
    min_agreement: int
    strategies_used: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    trades: List[dict]
    equity_curve: List[dict]


def run_consensus_backtest(
    symbol: str,
    configs: List[StrategyConfig],
    min_agreement: int = 2,
    period: str = "1y",
    initial_capital: float = 100_000.0,
) -> ConsensusBacktestResult:
    """
    Backtest a symbol using multiple strategies with consensus filtering.
    Only trades when min_agreement strategies agree on the same direction.
    """
    df = get_ohlcv(symbol, period=period)
    if df.empty or len(df) < 35:
        raise ValueError(f"Not enough data for {symbol}")

    closes = df["Close"].dropna()
    opens  = df["Open"].dropna()
    dates  = [str(d)[:10] for d in df.index]

    strategy_names = [c.name for c in configs]
    capital = initial_capital
    position = 0.0
    position_cost = 0.0
    trades: List[dict] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital
    lookback = 35

    for i in range(lookback, len(df)):
        price_series = closes.iloc[:i]
        current_close = float(closes.iloc[i])
        fill_price = float(opens.iloc[i]) if i < len(opens) else current_close
        today = dates[i]

        # Collect votes from all strategies
        votes: dict = defaultdict(list)
        for config in configs:
            try:
                signal = evaluate_strategy(config.type, symbol, price_series, config.params)
                if signal.direction != "HOLD":
                    votes[signal.direction].append(config.name)
            except Exception:
                continue

        # Check consensus
        action = None
        agreeing = []
        for direction, names in votes.items():
            if len(names) >= min_agreement:
                action = direction
                agreeing = names
                break

        # Execute trade based on consensus
        if action == "BUY" and position == 0:
            affordable_qty = (capital * 0.95) / fill_price if fill_price > 0 else 0
            qty = min(1.0, affordable_qty) if affordable_qty >= 0.01 else 0
            if qty > 0:
                cost = fill_price * qty
                capital -= cost
                position = qty
                position_cost = cost
                trades.append({
                    "date": today, "side": "BUY", "price": round(fill_price, 2),
                    "quantity": qty, "value": round(cost, 2),
                    "agreeing": agreeing, "pnl": None,
                })

        elif action == "SELL" and position > 0:
            proceeds = fill_price * position
            pnl = proceeds - position_cost
            capital += proceeds
            trades.append({
                "date": today, "side": "SELL", "price": round(fill_price, 2),
                "quantity": position, "value": round(proceeds, 2),
                "agreeing": agreeing, "pnl": round(pnl, 2),
            })
            position = 0.0
            position_cost = 0.0

        # Mark-to-market
        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

    # Close open position
    if position > 0:
        proceeds = float(closes.iloc[-1]) * position
        pnl = proceeds - position_cost
        capital += proceeds
        trades.append({
            "date": dates[-1], "side": "SELL (close)",
            "price": round(float(closes.iloc[-1]), 2),
            "quantity": position, "value": round(proceeds, 2),
            "agreeing": [], "pnl": round(pnl, 2),
        })

    final_capital = capital
    total_pnl = final_capital - initial_capital
    total_return = total_pnl / initial_capital * 100

    buy_trades  = [t for t in trades if t["side"] == "BUY"]
    sell_trades = [t for t in trades if "SELL" in t["side"]]
    winning = sum(1 for t in sell_trades if (t.get("pnl") or 0) > 0)
    losing  = sum(1 for t in sell_trades if (t.get("pnl") or 0) <= 0)
    win_rate = winning / len(sell_trades) * 100 if sell_trades else 0.0

    sharpe = None
    if len(daily_returns) > 1:
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    return ConsensusBacktestResult(
        symbol=symbol,
        period=period,
        min_agreement=min_agreement,
        strategies_used=strategy_names,
        start_date=dates[lookback] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_pnl=round(total_pnl, 2),
        total_return_pct=round(total_return, 2),
        total_trades=len(trades),
        winning_trades=winning,
        losing_trades=losing,
        win_rate_pct=round(win_rate, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )
