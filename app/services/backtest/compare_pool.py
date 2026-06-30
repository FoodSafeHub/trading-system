"""Process-pool helpers for the Compare-All backtests.

The per-strategy backtest is CPU-bound (pandas/numpy over a 10y daily frame),
so a ThreadPoolExecutor barely helps — the GIL serializes the heavy work. A
ProcessPoolExecutor runs each strategy on its own core for a near-linear
speedup.

To keep the (large) shared OHLCV frames out of every task's pickle payload, we
hand them to each worker process ONCE via a pool initializer that stashes them
in module-level globals. Tasks then carry only the lightweight strategy name.
Strategy objects are module-level singletons rebuilt inside the worker via
PERPLEXITY_STRATEGIES, so they never cross the process boundary either.
"""
from __future__ import annotations

import pandas as pd

# Per-worker shared state, populated by _pool_init / _custom_pool_init.
_SHARED: dict = {}


def _custom_pool_init(df_shared, symbol, period, initial_capital):
    _SHARED["df_shared"] = df_shared
    _SHARED["symbol"] = symbol
    _SHARED["period"] = period
    _SHARED["initial_capital"] = initial_capital


def run_one_custom(cfg_name: str, cfg_type: str, cfg_params: dict) -> dict:
    """Run one scanner-strategy config against the worker's shared frame.

    Mirrors the per-row body of /backtest/custom-compare-all. Picklable so it can
    execute in a ProcessPoolExecutor (the backtest is CPU-bound; threads are GIL-
    bound). The (large) OHLCV frame is shared via the pool initializer.
    """
    from app.services.backtest.engine import run_backtest

    sym = _SHARED["symbol"]
    try:
        r = run_backtest(
            strategy_name=cfg_name,
            symbol=sym,
            strategy_type=cfg_type,
            params=cfg_params,
            period=_SHARED["period"],
            initial_capital=_SHARED["initial_capital"],
            quantity=0,
            df=_SHARED["df_shared"],
        )
    except Exception as exc:
        return {"strategy_name": cfg_name, "error": str(exc)}

    buys = [t for t in r.trades if t.side == "BUY"]
    sells = [t for t in r.trades if "SELL" in t.side]
    n_rt = min(len(buys), len(sells))
    wins_pnl: list = []
    losses_pnl: list = []
    win_pct_list: list = []
    loss_pct_list: list = []
    for b, s in zip(buys[:n_rt], sells[:n_rt]):
        pnl = s.value - b.value
        pct = (pnl / b.value * 100) if b.value > 0 else 0.0
        if pnl > 0:
            wins_pnl.append(pnl)
            win_pct_list.append(pct)
        else:
            losses_pnl.append(pnl)
            loss_pct_list.append(pct)

    gross_win = sum(wins_pnl)
    gross_loss = abs(sum(losses_pnl))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (None if not wins_pnl else float("inf"))
    avg_win_pct = (sum(win_pct_list) / len(win_pct_list)) if win_pct_list else 0.0
    avg_loss_pct = (sum(loss_pct_list) / len(loss_pct_list)) if loss_pct_list else 0.0
    wr_frac = (r.win_rate_pct / 100.0) if r.win_rate_pct else 0.0
    expectancy_pct = wr_frac * avg_win_pct + (1 - wr_frac) * avg_loss_pct

    cagr = 0.0
    try:
        from datetime import date as _date
        if r.start_date and r.end_date:
            d0 = _date.fromisoformat(r.start_date)
            d1 = _date.fromisoformat(r.end_date)
            years = max((d1 - d0).days / 365.25, 1e-6)
            if r.initial_capital > 0 and r.final_capital > 0:
                cagr = ((r.final_capital / r.initial_capital) ** (1 / years) - 1) * 100
    except Exception:
        cagr = 0.0

    pf_out = profit_factor
    if pf_out is not None and pf_out == float("inf"):
        pf_out = None

    return {
        "strategy_name": r.strategy_name,
        "symbol": r.symbol,
        "strategy_type": cfg_type,
        "params": cfg_params,
        "period": r.period,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "initial_capital": r.initial_capital,
        "final_capital": r.final_capital,
        "total_pnl": r.total_pnl,
        "total_return_pct": r.total_return_pct,
        "total_trades": r.total_trades,
        "winning_trades": r.winning_trades,
        "losing_trades": r.losing_trades,
        "win_rate_pct": r.win_rate_pct,
        "profit_factor": round(pf_out, 2) if pf_out is not None else None,
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "expectancy_pct": round(expectancy_pct, 2),
        "cagr": round(cagr, 2),
        "max_drawdown_pct": r.max_drawdown_pct,
        "sharpe_ratio": r.sharpe_ratio,
        "equity_curve": r.equity_curve,
        "trades": [
            {"date": t.date, "side": t.side, "price": t.price,
             "quantity": t.quantity, "value": round(t.value, 2)}
            for t in r.trades
        ],
    }


def _pool_init(df_full, spy_close, mom_index_close, mom_vix_close,
               period, initial_capital, position_pct):
    _SHARED["df_full"] = df_full
    _SHARED["spy_close"] = spy_close
    _SHARED["mom_index_close"] = mom_index_close
    _SHARED["mom_vix_close"] = mom_vix_close
    _SHARED["period"] = period
    _SHARED["initial_capital"] = initial_capital
    _SHARED["position_pct"] = position_pct


def run_one(strategy_name: str, symbol: str) -> dict:
    """Run a single strategy against the worker's shared frames. Picklable."""
    from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES
    from app.services.backtest.perplexity_engine import run_perplexity_backtest

    strategy = next((s for s in PERPLEXITY_STRATEGIES if s.name == strategy_name), None)
    if strategy is None:
        return {"strategy_name": strategy_name, "error": "unknown strategy"}
    try:
        r = run_perplexity_backtest(
            strategy, symbol, _SHARED["period"], _SHARED["initial_capital"],
            position_pct=_SHARED["position_pct"],
            df_full=_SHARED["df_full"], spy_close=_SHARED["spy_close"],
            mom_index_close=_SHARED["mom_index_close"],
            mom_vix_close=_SHARED["mom_vix_close"],
        )
        return {
            "strategy_name": r.strategy_name,
            "total_trades": r.total_trades,
            "win_rate_pct": r.win_rate_pct,
            "profit_factor": r.profit_factor,
            "avg_win_pct": r.avg_win_pct,
            "avg_loss_pct": r.avg_loss_pct,
            "expectancy_pct": r.expectancy_pct,
            "expectancy_r": r.expectancy_r,
            "average_holding_days": r.average_holding_days,
            "average_r_multiple": r.average_r_multiple,
            "total_return_pct": r.total_return_pct,
            "cagr": r.cagr,
            "total_pnl": r.total_pnl,
            "capital_employed": r.capital_employed,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
        }
    except Exception as exc:
        return {"strategy_name": strategy_name, "error": str(exc)}
