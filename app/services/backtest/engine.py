from __future__ import annotations

"""
Backtesting engine — simulates strategy execution against historical OHLCV data.

For each bar, runs the strategy on all data up to that point (no lookahead),
simulates fills at next-bar open price, and tracks P&L.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.strategy.rules import evaluate_strategy, PositionState


@dataclass
class BacktestTrade:
    date: str
    symbol: str
    side: str           # BUY or SELL
    price: float
    quantity: float
    value: float        # price * quantity
    signal_from: str    # strategy name


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    trades: List[BacktestTrade]
    equity_curve: List[dict]   # [{date, equity}]
    total_pnl: float


def run_backtest(
    strategy_name: str,
    symbol: str,
    strategy_type: str,
    params: dict,
    period: str = "1y",
    initial_capital: float = 100_000.0,
    quantity: float = 1.0,
    df: pd.DataFrame | None = None,
) -> BacktestResult:
    """
    Simulate a strategy over historical data.
    Uses next-bar open as the fill price to avoid lookahead bias.

    df: optional pre-fetched OHLCV. When provided, skips the get_ohlcv call —
    used by grid-search calibration to reuse one fetch across many param combos.
    """
    if df is None:
        df = get_ohlcv(symbol, period=period)
    if df.empty or len(df) < 30:
        # Distinguish "no data anywhere" from "stock too newly listed". A short
        # series means the fetch worked but the stock simply hasn't traded long
        # enough to backtest — common for recent IPOs/relistings on NSE.
        if df.empty:
            raise ValueError(f"No data returned for {symbol} over period {period}")
        listed = str(df.index[0])[:10]
        raise ValueError(
            f"{symbol} has only {len(df)} trading days of history (listed ~{listed}); "
            f"backtests need at least 30 bars. Pick a longer-established stock or "
            f"a shorter period."
        )

    closes = df["Close"].dropna()
    opens  = df["Open"].dropna()
    dates  = [str(d)[:10] for d in df.index]

    capital = initial_capital
    position = 0.0          # shares held
    position_cost = 0.0     # total cost basis
    entry_price = 0.0       # fill price of the open position (for trailing-stop overlay)
    highest_close = 0.0     # highest close seen since entry (Chandelier trail)
    trades: List[BacktestTrade] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital

    # Need at least 30 bars of history before we start signalling
    lookback = 35

    for i in range(lookback, len(df)):
        price_series = closes.iloc[:i]
        current_close = float(closes.iloc[i])
        # Fill at next bar open if available, else current close
        fill_price = float(opens.iloc[i]) if i < len(opens) else current_close
        today = dates[i]

        df_slice = df.iloc[:i]
        # The rule decides on data through bar i-1 (price_series/df_slice end at
        # i-1) and we fill at bar i's open — no lookahead. The trailing overlay
        # must see the SAME frame, so trail against the peak close through i-1,
        # not current_close (= close[i], which the rule cannot see yet).
        if position > 0:
            highest_close = max(highest_close, float(closes.iloc[i - 1]))
        pos_state = (
            PositionState(entry_price=entry_price, highest_close=highest_close)
            if position > 0 else None
        )
        signal = evaluate_strategy(
            strategy_type, symbol, price_series, params, ohlcv=df_slice, position=pos_state
        )

        direction = signal.direction

        # Execute simulated trade
        if direction == "BUY" and position == 0:
            # Use up to 95% of available capital
            affordable_qty = (capital * 0.95) / fill_price if fill_price > 0 else 0
            # quantity<=0 means "use all capital"; quantity>0 is a fixed share count cap
            actual_qty = affordable_qty if quantity <= 0 else min(quantity, affordable_qty)
            actual_qty = actual_qty if affordable_qty >= 0.01 else 0
            if actual_qty > 0:
                cost = fill_price * actual_qty
                capital -= cost
                position = actual_qty
                position_cost = cost
                entry_price = fill_price
                highest_close = current_close
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="BUY",
                    price=fill_price, quantity=actual_qty, value=cost,
                    signal_from=strategy_name,
                ))

        elif direction == "SELL" and position > 0:
            proceeds = fill_price * position
            pnl = proceeds - position_cost
            capital += proceeds
            trades.append(BacktestTrade(
                date=today, symbol=symbol, side="SELL",
                price=fill_price, quantity=position, value=proceeds,
                signal_from=strategy_name,
            ))
            position = 0.0
            position_cost = 0.0
            entry_price = 0.0
            highest_close = 0.0

        # Mark-to-market equity
        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        # Drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        # Daily return
        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

    # Close any open position at last price
    final_close = float(closes.iloc[-1])
    if position > 0:
        proceeds = final_close * position
        trades.append(BacktestTrade(
            date=dates[-1], symbol=symbol, side="SELL (close)",
            price=final_close, quantity=position, value=proceeds,
            signal_from=strategy_name,
        ))
        capital += proceeds
        position = 0.0

    final_capital = capital

    # P&L per round trip
    buy_trades  = [t for t in trades if t.side == "BUY"]
    sell_trades = [t for t in trades if "SELL" in t.side]
    round_trips = min(len(buy_trades), len(sell_trades))

    winning = losing = 0
    for b, s in zip(buy_trades[:round_trips], sell_trades[:round_trips]):
        if s.value > b.value:
            winning += 1
        else:
            losing += 1

    win_rate = (winning / round_trips * 100) if round_trips > 0 else 0.0
    total_return = (final_capital - initial_capital) / initial_capital * 100
    total_pnl = final_capital - initial_capital

    # Sharpe ratio (annualised, risk-free = 0)
    sharpe = None
    if len(daily_returns) > 1:
        import statistics
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        period=period,
        start_date=dates[lookback] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_return_pct=round(total_return, 2),
        total_pnl=round(total_pnl, 2),
        total_trades=len(trades),
        winning_trades=winning,
        losing_trades=losing,
        win_rate_pct=round(win_rate, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )
