from __future__ import annotations

"""
Portfolio backtester for Perplexity strategies.

Runs one strategy across multiple symbols simultaneously, sharing a single
capital pool. Positions across symbols are sized as a fixed % of current
portfolio equity so capital is always deployed. Maintains a full daily
equity curve across the entire portfolio.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from app.services.backtest.costs import CostModel
from app.services.backtest.perplexity_engine import calc_cagr, calc_total_return
from app.services.market_data.provider import get_ohlcv
from app.services.strategy.perplexity.base import PerplexityStrategy

# Engine-level default when a strategy doesn't declare a budget.
_DEFAULT_MAX_HOLD_BARS = 60


@dataclass
class PortfolioTrade:
    date: str
    symbol: str
    side: str
    price: float
    quantity: float
    value: float
    pnl: Optional[float]
    strategy: str
    reason: str = ""


@dataclass
class PortfolioBacktestResult:
    strategy_name: str
    symbols: List[str]
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float
    cagr: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: Optional[float]
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    capital_utilisation_pct: float   # avg % of capital deployed on any given day
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


def run_portfolio_backtest(
    strategy: PerplexityStrategy,
    symbols: List[str],
    period: str = "5y",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.20,      # % of portfolio equity per position
    max_open_positions: int = 5,     # never hold more than this many symbols at once
    cost_model: Optional[CostModel] = None,  # opt-in slippage/commission; None == identity (baseline byte-equivalent)
) -> PortfolioBacktestResult:
    """
    Run one strategy over a portfolio of symbols with shared capital.

    Each bar, for every symbol not currently held, the strategy is evaluated.
    If a BUY signal fires and we have capacity (<max_open_positions), we open
    a position sized at position_pct of current portfolio equity.
    Exits are triggered by the strategy SELL signal, or stop/target breach.
    """
    # ── Load all data ─────────────────────────────────────────
    raw: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_ohlcv(sym, period=period)
            if not df.empty and len(df) >= 60:
                raw[sym] = df
        except Exception:
            pass

    if not raw:
        raise ValueError("No data loaded for any symbol")

    # Build a unified date index (union of all trading days)
    all_dates = sorted(set(
        str(d)[:10] for df in raw.values() for d in df.index
    ))

    lookback = 210  # need SMA200

    # Read the strategy's own time-budget; same convention as perplexity_engine.
    try:
        max_hold_bars = int(getattr(strategy, "config", {}).get("max_hold_bars", _DEFAULT_MAX_HOLD_BARS))
    except Exception:
        max_hold_bars = _DEFAULT_MAX_HOLD_BARS
    if max_hold_bars <= 0:
        max_hold_bars = _DEFAULT_MAX_HOLD_BARS

    # ── State ─────────────────────────────────────────────────
    capital = initial_capital
    # positions[sym] = {qty, cost, stop, target, bars_held}
    positions: Dict[str, dict] = {}

    all_trades: List[dict] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital
    daily_deployed: List[float] = []

    # Pre-build per-symbol date→iloc mapping for fast lookup
    sym_iloc: Dict[str, Dict[str, int]] = {}
    for sym, df in raw.items():
        sym_iloc[sym] = {str(d)[:10]: i for i, d in enumerate(df.index)}

    for date_idx, today in enumerate(all_dates):
        if date_idx < lookback:
            equity_curve.append({"date": today, "equity": round(initial_capital, 2)})
            daily_returns.append(0.0)
            prev_equity = initial_capital
            continue

        # Mark-to-market current equity
        position_value = 0.0
        for sym, pos in list(positions.items()):
            df = raw.get(sym)
            if df is None:
                continue
            iloc = sym_iloc[sym].get(today)
            if iloc is None:
                # Use last known close
                position_value += pos["qty"] * pos["last_price"]
                continue
            close = float(df["Close"].iloc[iloc])
            pos["last_price"] = close
            position_value += pos["qty"] * close

            # Check stop / target intraday
            low  = float(df["Low"].iloc[iloc])
            high = float(df["High"].iloc[iloc])

            # Count a holding day BEFORE intra-bar exit checks so the bar a
            # stop/target/time-exit fires on counts towards the budget.
            pos["bars_held"] = pos.get("bars_held", 0) + 1

            if pos["stop"] and low <= pos["stop"]:
                sell_px = pos["stop"] if cost_model is None else cost_model.apply_sell(pos["stop"])
                proceeds = sell_px * pos["qty"]
                commission = 0.0 if cost_model is None else cost_model.exit_commission(pos["qty"], proceeds)
                pnl = proceeds - commission - pos["cost"]
                capital += proceeds - commission
                position_value -= pos["qty"] * close  # already counted above
                all_trades.append({
                    "date": today, "symbol": sym, "side": "SELL (stop)",
                    "price": round(sell_px, 2), "quantity": round(pos["qty"], 4),
                    "value": round(proceeds, 2), "pnl": round(pnl, 2), "reason": "stop hit",
                    "commission": round(commission, 4),
                })
                del positions[sym]
                continue

            if pos["target"] and high >= pos["target"]:
                sell_px = pos["target"] if cost_model is None else cost_model.apply_sell(pos["target"])
                proceeds = sell_px * pos["qty"]
                commission = 0.0 if cost_model is None else cost_model.exit_commission(pos["qty"], proceeds)
                pnl = proceeds - commission - pos["cost"]
                capital += proceeds - commission
                all_trades.append({
                    "date": today, "symbol": sym, "side": "SELL (target)",
                    "price": round(sell_px, 2), "quantity": round(pos["qty"], 4),
                    "value": round(proceeds, 2), "pnl": round(pnl, 2), "reason": "target hit",
                    "commission": round(commission, 4),
                })
                del positions[sym]
                continue

            # Time exit: this bar's CLOSE if max_hold_bars budget exceeded.
            # Mirrors perplexity_engine's "SELL (time)" convention so the
            # downstream metrics aggregator treats it consistently.
            if pos["bars_held"] >= max_hold_bars:
                sell_px = close if cost_model is None else cost_model.apply_sell(close)
                proceeds = sell_px * pos["qty"]
                commission = 0.0 if cost_model is None else cost_model.exit_commission(pos["qty"], proceeds)
                pnl = proceeds - commission - pos["cost"]
                capital += proceeds - commission
                all_trades.append({
                    "date": today, "symbol": sym, "side": "SELL (time)",
                    "price": round(sell_px, 2), "quantity": round(pos["qty"], 4),
                    "value": round(proceeds, 2), "pnl": round(pnl, 2),
                    "reason": f"max_hold_bars={max_hold_bars} reached",
                    "commission": round(commission, 4),
                })
                del positions[sym]
                continue

        # Run strategy signals on all symbols
        for sym, df in raw.items():
            iloc = sym_iloc[sym].get(today)
            if iloc is None or iloc < lookback:
                continue

            df_slice = df.iloc[:iloc]
            fill_price = float(df["Open"].iloc[iloc])

            if sym not in positions:
                # Only enter if we have capacity
                if len(positions) >= max_open_positions:
                    continue
                try:
                    sig = strategy.run(sym, df_slice)
                except Exception:
                    continue

                if sig and sig.direction == "BUY":
                    equity_now = capital + sum(
                        p["qty"] * p.get("last_price", p["cost"] / p["qty"])
                        for p in positions.values()
                    )
                    alloc = equity_now * position_pct
                    if alloc > capital:
                        alloc = capital  # can't spend more than free cash
                    # cost-adjusted buy price: cost_model=None → identity
                    buy_px = fill_price if cost_model is None else cost_model.apply_buy(fill_price)
                    qty = alloc / buy_px if buy_px > 0 else 0.0
                    qty = round(qty, 6)
                    cost = buy_px * qty
                    commission = 0.0 if cost_model is None else cost_model.entry_commission(qty, cost)
                    if qty >= 0.001 and cost + commission <= capital:
                        capital -= cost + commission
                        positions[sym] = {
                            "qty": qty,
                            "cost": cost + commission,   # include entry commission in cost basis
                            "stop": sig.stop_price,
                            "target": sig.target_price,
                            "last_price": buy_px,
                            "bars_held": 0,
                        }
                        all_trades.append({
                            "date": today, "symbol": sym, "side": "BUY",
                            "price": round(buy_px, 2), "quantity": round(qty, 4),
                            "value": round(cost, 2), "pnl": None,
                            "reason": sig.reason or "",
                            "commission": round(commission, 4),
                        })
            else:
                # Check for SELL signal
                try:
                    sig = strategy.run(sym, df_slice)
                except Exception:
                    continue

                if sig and sig.direction == "SELL" and sym in positions:
                    pos = positions[sym]
                    sell_px = fill_price if cost_model is None else cost_model.apply_sell(fill_price)
                    proceeds = sell_px * pos["qty"]
                    commission = 0.0 if cost_model is None else cost_model.exit_commission(pos["qty"], proceeds)
                    pnl = proceeds - commission - pos["cost"]
                    capital += proceeds - commission
                    all_trades.append({
                        "date": today, "symbol": sym, "side": "SELL",
                        "price": round(sell_px, 2), "quantity": round(pos["qty"], 4),
                        "value": round(proceeds, 2), "pnl": round(pnl, 2),
                        "reason": sig.reason or "",
                        "commission": round(commission, 4),
                    })
                    del positions[sym]

        # End-of-day equity
        pos_val = sum(
            p["qty"] * p.get("last_price", p["cost"] / p["qty"])
            for p in positions.values()
        )
        equity = capital + pos_val
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

        # Track utilisation
        daily_deployed.append(pos_val / equity * 100 if equity > 0 else 0)

    # Close remaining positions at last price
    last_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        df = raw.get(sym)
        if df is not None:
            last_close = float(df["Close"].iloc[-1])
            sell_px = last_close if cost_model is None else cost_model.apply_sell(last_close)
            proceeds = sell_px * pos["qty"]
            commission = 0.0 if cost_model is None else cost_model.exit_commission(pos["qty"], proceeds)
            pnl = proceeds - commission - pos["cost"]
            capital += proceeds - commission
            all_trades.append({
                "date": last_date, "symbol": sym, "side": "SELL (close)",
                "price": round(sell_px, 2), "quantity": round(pos["qty"], 4),
                "value": round(proceeds, 2), "pnl": round(pnl, 2), "reason": "end of backtest",
                "commission": round(commission, 4),
            })

    final_capital = capital
    total_pnl = final_capital - initial_capital

    sell_trades = [t for t in all_trades if "SELL" in t["side"] and t.get("pnl") is not None]
    winning = [t for t in sell_trades if (t.get("pnl") or 0) > 0]
    losing  = [t for t in sell_trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0.0

    gross_wins   = sum(t["pnl"] for t in winning)
    gross_losses = abs(sum(t["pnl"] for t in losing))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    sharpe = None
    non_zero = [r for r in daily_returns if r != 0]
    if len(non_zero) > 1:
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    cap_util = round(sum(daily_deployed) / len(daily_deployed), 1) if daily_deployed else 0.0

    return PortfolioBacktestResult(
        strategy_name=strategy.name,
        symbols=list(raw.keys()),
        period=period,
        start_date=all_dates[lookback] if len(all_dates) > lookback else all_dates[0],
        end_date=all_dates[-1],
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_pnl=round(total_pnl, 2),
        total_return_pct=calc_total_return(equity_curve),
        cagr=calc_cagr(equity_curve),
        total_trades=len(sell_trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate_pct=round(win_rate, 2),
        profit_factor=profit_factor,
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        capital_utilisation_pct=cap_util,
        trades=all_trades,
        equity_curve=equity_curve,
    )
