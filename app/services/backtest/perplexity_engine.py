from __future__ import annotations

"""
Backtester for Perplexity swing trading strategies.
Runs each strategy bar-by-bar with no lookahead bias.
Fills at the next bar's open price.
Respects stop_price and target_price if set.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.perplexity.base import PerplexityStrategy


@dataclass
class PerplexityBacktestResult:
    strategy_name: str
    symbol: str
    period: str
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
    profit_factor: float        # gross wins / gross losses
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


def run_perplexity_backtest(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    risk_pct_per_trade: float = 0.01,
    max_position_pct: float = 0.20,        # never put more than 20% of capital in one trade
) -> PerplexityBacktestResult:
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty or len(df_full) < 60:
        raise ValueError(f"Not enough data for {symbol} (need at least 60 bars)")

    # Use 210-bar lookback when we have enough data (needed for SMA200).
    # For shorter periods (6mo/1y) fall back to 60 bars — SMA200 will return
    # HOLD on most signals but the backtest still runs and shows real results.
    lookback = min(210, max(60, len(df_full) // 3))

    dates = [str(d)[:10] for d in df_full.index]
    capital = initial_capital
    position = 0.0
    position_cost = 0.0
    entry_stop: Optional[float] = None
    entry_target: Optional[float] = None
    open_risk_usd: float = 0.0          # dollars currently at risk in open position
    trades: List[dict] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital

    for i in range(lookback, len(df_full)):
        df_slice = df_full.iloc[:i]
        current_close = float(df_full["Close"].iloc[i])
        fill_price    = float(df_full["Open"].iloc[i]) if i < len(df_full) else current_close
        today = dates[i]

        # ── Check stop / target on open if in position ──────────
        if position > 0 and entry_stop is not None:
            open_price = float(df_full["Open"].iloc[i])
            # Gap down through stop
            if open_price <= entry_stop:
                proceeds = open_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append(_trade("SELL (stop)", today, open_price, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0
            # Gap up through target
            elif entry_target and open_price >= entry_target:
                proceeds = open_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append(_trade("SELL (target)", today, open_price, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        # ── Intraday stop / target on close ─────────────────────
        if position > 0 and entry_stop is not None:
            low_today = float(df_full["Low"].iloc[i])
            high_today = float(df_full["High"].iloc[i])
            if low_today <= entry_stop:
                pnl = entry_stop * position - position_cost
                capital += entry_stop * position
                trades.append(_trade("SELL (stop)", today, entry_stop, position,
                                     entry_stop * position, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0
            elif entry_target and high_today >= entry_target:
                pnl = entry_target * position - position_cost
                capital += entry_target * position
                trades.append(_trade("SELL (target)", today, entry_target, position,
                                     entry_target * position, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        # ── Run strategy signal ──────────────────────────────────
        if position == 0:
            try:
                sig = strategy.run(symbol, df_slice)
            except Exception:
                sig = None

            if sig and sig.direction == "BUY":
                # Position sizing: risk fixed % of current capital by stop distance
                if sig.stop_price and sig.stop_price < fill_price:
                    sz = calculate_position_size(
                        symbol=symbol,
                        entry_price=fill_price,
                        stop_price=sig.stop_price,
                        account_value=capital,
                        risk_pct_per_trade=risk_pct_per_trade,
                        max_position_size_usd=capital * max_position_pct,
                        current_open_risk_usd=open_risk_usd,
                    )
                    qty = sz.shares if sz.viable else 0.0
                    trade_risk = sz.risk_amount if sz.viable else 0.0
                else:
                    # No stop defined — fall back to investing 10% of capital
                    qty = (capital * 0.10) / fill_price if fill_price > 0 else 0
                    trade_risk = 0.0

                qty = round(qty, 6)
                if qty >= 0.001 and fill_price * qty <= capital:
                    cost = fill_price * qty
                    capital -= cost
                    position = qty
                    position_cost = cost
                    open_risk_usd += trade_risk
                    entry_stop   = sig.stop_price
                    entry_target = sig.target_price
                    trades.append({
                        "date": today, "side": "BUY",
                        "price": round(fill_price, 2), "quantity": round(qty, 4),
                        "value": round(cost, 2), "pnl": None,
                        "stop": round(sig.stop_price, 2) if sig.stop_price else None,
                        "target": round(sig.target_price, 2) if sig.target_price else None,
                        "confidence": sig.confidence,
                        "reason": sig.reason,
                        "risk_usd": round(trade_risk, 2),
                    })

        elif position > 0:
            try:
                sig = strategy.run(symbol, df_slice)
            except Exception:
                sig = None

            if sig and sig.direction == "SELL":
                proceeds = fill_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append({
                    "date": today, "side": "SELL",
                    "price": round(fill_price, 2), "quantity": position,
                    "value": round(proceeds, 2), "pnl": round(pnl, 2),
                    "stop": None, "target": None,
                    "confidence": sig.confidence if sig else None,
                    "reason": sig.reason if sig else "",
                    "risk_usd": None,
                })
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        # ── Mark-to-market ───────────────────────────────────────
        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

    # Close any open position at last close
    if position > 0:
        last_price = float(df_full["Close"].iloc[-1])
        proceeds = last_price * position
        pnl = proceeds - position_cost
        capital += proceeds
        trades.append(_trade("SELL (close)", dates[-1], last_price, position, proceeds, pnl))

    final_capital = capital
    total_pnl = final_capital - initial_capital
    total_return = total_pnl / initial_capital * 100

    sell_trades = [t for t in trades if "SELL" in t["side"]]
    winning = [t for t in sell_trades if (t.get("pnl") or 0) > 0]
    losing  = [t for t in sell_trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0.0
    # total_trades = round trips (sell count), not raw trade records (buy+sell)
    total_completed_trades = len(sell_trades)

    gross_wins   = sum(t["pnl"] for t in winning)
    gross_losses = abs(sum(t["pnl"] for t in losing))
    # None = no losing trades at all (perfect record) — displayed as "—" in UI
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    sharpe = None
    if len(daily_returns) > 1:
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    return PerplexityBacktestResult(
        strategy_name=strategy.name,
        symbol=symbol,
        period=period,
        start_date=dates[lookback] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_pnl=round(total_pnl, 2),
        total_return_pct=round(total_return, 2),
        total_trades=total_completed_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate_pct=round(win_rate, 2),
        profit_factor=profit_factor,
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )


def _trade(side: str, date: str, price: float, qty: float, value: float, pnl: float) -> dict:
    return {
        "date": date, "side": side,
        "price": round(price, 2), "quantity": qty,
        "value": round(value, 2), "pnl": round(pnl, 2),
        "stop": None, "target": None, "confidence": None, "reason": "", "risk_usd": None,
    }
