from __future__ import annotations

"""
Relative-strength rotation — Phase 1 (NEW capability, separate from the
per-symbol rule registry).

This is a CROSS-SECTIONAL strategy: it ranks a basket of symbols by their
return relative to a benchmark and holds the leaders. It therefore does NOT fit
``evaluate_strategy(symbol, prices, params)`` and is deliberately kept out of
``rules._RULE_REGISTRY``. It is a standalone service with its own backtest
harness, surfaced via an additive API route. Nothing here changes existing
scanner/scheduler/recommendation behaviour.

Offline-testable: pass ``data={symbol: ohlcv_df}`` to skip network fetches.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RankedSymbol:
    symbol: str
    rs_score: float          # blended return minus benchmark blended return
    ret_3m: float            # fractional return over lookback_3m bars
    ret_6m: float            # fractional return over lookback_6m bars
    above_sma200: bool
    last_close: float


def _get_df(symbol: str, period: str, data: Optional[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    if data is not None:
        return data.get(symbol, pd.DataFrame())
    from app.services.market_data.provider import get_ohlcv
    return get_ohlcv(symbol, period=period)


def _blended(ret_3m: float, ret_6m: float) -> float:
    return 0.5 * ret_3m + 0.5 * ret_6m


def _ret_over(close: pd.Series, lookback: int) -> Optional[float]:
    if len(close) <= lookback:
        return None
    past = float(close.iloc[-lookback - 1])
    now = float(close.iloc[-1])
    return (now / past - 1.0) if past > 0 else None


def rank_relative_strength(
    symbols: List[str],
    benchmark_symbol: str = "SPY",
    period: str = "2y",
    lookback_3m: int = 63,
    lookback_6m: int = 126,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[RankedSymbol]:
    """Rank each symbol by blended 3m/6m return RELATIVE to the benchmark.

    Returns all rankable symbols sorted best-first. ``above_sma200`` is a flag
    the caller (or the harness) uses to gate buys — names below their 200-SMA
    are still ranked but should not be held long.
    """
    bench_df = _get_df(benchmark_symbol, period, data)
    bench_blended = 0.0
    if not bench_df.empty and "Close" in bench_df:
        b3 = _ret_over(bench_df["Close"], lookback_3m)
        b6 = _ret_over(bench_df["Close"], lookback_6m)
        if b3 is not None and b6 is not None:
            bench_blended = _blended(b3, b6)

    ranked: List[RankedSymbol] = []
    for sym in symbols:
        df = _get_df(sym, period, data)
        if df.empty or "Close" not in df or len(df) <= lookback_6m + 1:
            continue
        close = df["Close"]
        r3 = _ret_over(close, lookback_3m)
        r6 = _ret_over(close, lookback_6m)
        if r3 is None or r6 is None:
            continue
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
        last = float(close.iloc[-1])
        above = (not math.isnan(sma200)) and last > sma200
        ranked.append(RankedSymbol(
            symbol=sym,
            rs_score=round(_blended(r3, r6) - bench_blended, 6),
            ret_3m=round(r3, 6), ret_6m=round(r6, 6),
            above_sma200=above, last_close=last,
        ))
    ranked.sort(key=lambda r: r.rs_score, reverse=True)
    return ranked


# ── Portfolio backtest harness ─────────────────────────────────────────────────

@dataclass
class RsRotationResult:
    benchmark: str
    symbols: List[str]
    period: str
    top_n: int
    rebalance_days: int
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    sharpe: Optional[float]
    n_rebalances: int
    equity_curve: List[dict] = field(default_factory=list)
    holdings_log: List[dict] = field(default_factory=list)


def backtest_rs_rotation(
    symbols: List[str],
    benchmark_symbol: str = "SPY",
    period: str = "5y",
    top_n: int = 5,
    rebalance_days: int = 21,
    lookback_3m: int = 63,
    lookback_6m: int = 126,
    initial_capital: float = 100_000.0,
    cost_model=None,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> RsRotationResult:
    """Equal-weight rotation into the top-N RS names above their 200-SMA,
    rebalanced every ``rebalance_days``. No lookahead: each rebalance ranks on
    data through that date and trades at that day's close. Optional cost_model
    charges slippage+commission on turnover.
    """
    # Build a common close panel aligned on shared dates.
    closes: Dict[str, pd.Series] = {}
    for sym in symbols:
        df = _get_df(sym, period, data)
        if not df.empty and "Close" in df:
            closes[sym] = df["Close"]
    if not closes:
        raise ValueError("no usable symbol data for rs_rotation backtest")
    panel = pd.DataFrame(closes).dropna(how="all")
    panel = panel.ffill()
    dates = list(panel.index)
    if len(dates) <= lookback_6m + rebalance_days:
        raise ValueError("not enough history for rs_rotation backtest")

    cash = initial_capital
    shares: Dict[str, float] = {}
    equity_curve: List[dict] = []
    holdings_log: List[dict] = []
    peak = initial_capital
    max_dd = 0.0
    prev_equity = initial_capital
    daily_rets: List[float] = []
    n_rebal = 0

    start_i = lookback_6m + 1
    for i in range(start_i, len(dates)):
        d = dates[i]
        px_now = {s: float(panel[s].iloc[i]) for s in panel.columns if not pd.isna(panel[s].iloc[i])}

        # Rebalance on cadence (and on the very first eligible bar).
        if (i - start_i) % rebalance_days == 0:
            window = {s: panel[s].iloc[: i + 1] for s in panel.columns}
            ranked = rank_relative_strength(
                list(window.keys()), benchmark_symbol, period,
                lookback_3m, lookback_6m, data={**(data or {}),
                **{s: pd.DataFrame({"Close": w}) for s, w in window.items()}},
            )
            target = [r.symbol for r in ranked if r.above_sma200 and r.symbol in px_now][:top_n]
            # Liquidate everything (mark to current px), then equal-weight into targets.
            for s, q in list(shares.items()):
                if q > 0 and s in px_now:
                    proceeds = q * px_now[s]
                    if cost_model is not None:
                        proceeds = q * cost_model.apply_sell(px_now[s])
                        proceeds -= cost_model.exit_commission(q, proceeds)
                    cash += proceeds
            shares = {}
            if target:
                alloc = cash / len(target)
                for s in target:
                    buy_px = px_now[s] if cost_model is None else cost_model.apply_buy(px_now[s])
                    q = (alloc * 0.99) / buy_px if buy_px > 0 else 0.0
                    if q > 0:
                        cost = q * buy_px
                        if cost_model is not None:
                            cost += cost_model.entry_commission(q, cost)
                        cash -= cost
                        shares[s] = q
            n_rebal += 1
            holdings_log.append({"date": str(d)[:10], "holdings": target})

        equity = cash + sum(q * px_now.get(s, 0.0) for s, q in shares.items())
        equity_curve.append({"date": str(d)[:10], "equity": round(equity, 2)})
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        if prev_equity > 0:
            daily_rets.append((equity - prev_equity) / prev_equity)
        prev_equity = equity

    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100
    years = max((dates[-1] - dates[start_i]).days / 365.25, 1e-6)
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if final_equity > 0 else 0.0
    sharpe = None
    if len(daily_rets) > 1:
        sd = statistics.stdev(daily_rets)
        if sd > 0:
            sharpe = round(statistics.mean(daily_rets) / sd * (252 ** 0.5), 2)

    return RsRotationResult(
        benchmark=benchmark_symbol, symbols=list(panel.columns), period=period,
        top_n=top_n, rebalance_days=rebalance_days, initial_capital=initial_capital,
        final_capital=round(final_equity, 2), total_return_pct=round(total_return, 2),
        cagr=round(cagr, 2), max_drawdown_pct=round(max_dd, 2), sharpe=sharpe,
        n_rebalances=n_rebal, equity_curve=equity_curve, holdings_log=holdings_log,
    )
