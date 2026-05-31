from __future__ import annotations

"""
Walk-Forward out-of-sample validation for the v2 (generic) strategies.

The original walkforward_engine.py is hardwired to the PerplexityStrategy
interface (strategy.run(symbol, df)). The v2 strategies are driven instead
through the canonical run_backtest engine via evaluate_strategy(type, ...),
which is the same no-lookahead path the scheduler and Backtest page use —
including the Chandelier trailing-stop overlay. Re-running that one engine on
time slices keeps a single source of truth rather than a parallel simulator.

Two modes (mirrors walkforward_engine):
  - simple  : one IS / one OOS split by bar count.
  - rolling : sliding IS/OOS windows; OOS curves stitched into a composite.

Key metric: Walk-Forward Efficiency (WFE) = OOS_CAGR / IS_CAGR. The WFE bands
and the CAGR/return helpers are reused from walkforward_engine so the two
engines report identical, comparable numbers.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pandas as pd

from app.services.backtest.costs import CostModel
from app.services.backtest.engine import run_backtest
from app.services.backtest.walkforward_engine import (
    _TRADING_DAYS_PER_YEAR,
    _wfe,
    _wfe_label,
    calc_cagr,
    calc_total_return,
)
from app.services.market_data.provider import get_ohlcv


@dataclass
class V2Segment:
    """Metrics for one IS or OOS window, from a run_backtest result."""
    start: str
    end: str
    bars: int
    total_return_pct: float
    cagr: float
    win_rate_pct: float
    max_drawdown_pct: float
    trades: int
    sharpe: Optional[float]
    equity_curve: List[dict] = field(default_factory=list)


@dataclass
class V2SimpleSplitResult:
    mode: str                       # "simple"
    strategy_type: str
    symbol: str
    full_period: str
    train_pct: float
    is_segment: V2Segment
    oos_segment: V2Segment
    wfe: Optional[float]
    wfe_label: str


@dataclass
class V2RollingResult:
    mode: str                       # "rolling"
    strategy_type: str
    symbol: str
    full_period: str
    train_years: float
    test_years: float
    step_years: float
    segments: List[dict]
    oos_composite_curve: List[dict]
    global_oos_total_return_pct: float
    global_oos_cagr: float
    avg_is_cagr: float
    global_wfe: Optional[float]
    global_wfe_label: str


def _run_slice(
    strategy_type: str,
    symbol: str,
    params: dict,
    df_slice: pd.DataFrame,
    warmup_df: Optional[pd.DataFrame],
    initial_capital: float,
    cost_model: Optional[CostModel] = None,
) -> V2Segment:
    """Backtest one slice. warmup_df (IS history) is prepended so indicators
    have lookback, then trimmed out of the reported window so only OOS bars
    contribute trades/returns.

    run_backtest itself enforces a ~35-bar warmup internally, so when warmup_df
    is supplied we feed the concatenated frame and rebase the reported segment
    onto just the OOS span via its equity curve dates.
    """
    if warmup_df is not None and not warmup_df.empty:
        df_full = pd.concat([warmup_df, df_slice])
    else:
        df_full = df_slice

    r = run_backtest(
        strategy_name=f"wf:{strategy_type}",
        symbol=symbol,
        strategy_type=strategy_type,
        params=params,
        period="slice",
        initial_capital=initial_capital,
        quantity=0,
        df=df_full,
        cost_model=cost_model,
    )

    # Default: report the full backtest result as-is (correct for the IS case
    # where df_full == df_slice — no warmup, so r already covers only the IS span).
    curve = r.equity_curve
    trades_count = r.total_trades
    win_rate = r.win_rate_pct
    max_dd = r.max_drawdown_pct
    sharpe = r.sharpe_ratio

    # When warmed up (OOS case): trim the equity curve to the OOS span AND
    # recompute trade-derived sub-metrics on OOS-only data. Without this, r
    # reflects trades / drawdown / sharpe across IS+OOS (the warmup span is
    # traded too) and contaminates the OOS view.
    if warmup_df is not None and not warmup_df.empty and not df_slice.empty:
        oos_start = str(df_slice.index[0])[:10]
        trimmed = [pt for pt in curve if pt["date"] >= oos_start]
        if len(trimmed) >= 2:
            curve = trimmed

        # Trades that fall on/after OOS start. A leading SELL is a carry-over
        # close of an IS-opened position — dropped from OOS round-trip pairing
        # so we count only round-trips entirely within the held-out window.
        oos_events = [t for t in r.trades if t.date >= oos_start]
        while oos_events and "SELL" in oos_events[0].side:
            oos_events.pop(0)
        trades_count = len(oos_events)
        buys = [t for t in oos_events if t.side == "BUY"]
        sells = [t for t in oos_events if "SELL" in t.side]
        rt = min(len(buys), len(sells))
        wins = sum(1 for b, s in zip(buys[:rt], sells[:rt]) if s.value > b.value)
        win_rate = round(wins / rt * 100, 2) if rt > 0 else 0.0

        # Max drawdown with the peak reset at OOS start.
        if curve:
            peak = curve[0]["equity"]
            mdd = 0.0
            for pt in curve:
                if pt["equity"] > peak:
                    peak = pt["equity"]
                if peak > 0:
                    dd = (peak - pt["equity"]) / peak * 100
                    if dd > mdd:
                        mdd = dd
            max_dd = round(mdd, 2)

        # Sharpe from the trimmed curve's daily returns (rf = 0, annualised).
        if len(curve) >= 2:
            import statistics
            rets = []
            for i in range(1, len(curve)):
                prev = curve[i - 1]["equity"]
                cur_eq = curve[i]["equity"]
                if prev > 0:
                    rets.append((cur_eq - prev) / prev)
            if len(rets) >= 2:
                avg = statistics.mean(rets)
                sd = statistics.stdev(rets)
                sharpe = round(avg / sd * (252 ** 0.5), 2) if sd > 0 else None
            else:
                sharpe = None

    return V2Segment(
        start=curve[0]["date"] if curve else r.start_date,
        end=curve[-1]["date"] if curve else r.end_date,
        bars=len(df_slice),
        total_return_pct=calc_total_return(curve),
        cagr=calc_cagr(curve),
        win_rate_pct=win_rate,
        max_drawdown_pct=max_dd,
        trades=trades_count,
        sharpe=sharpe,
        equity_curve=curve,
    )


def run_simple_split_v2(
    strategy_type: str,
    symbol: str,
    params: dict,
    period: str = "5y",
    train_pct: float = 0.70,
    initial_capital: float = 100_000.0,
    cost_model: Optional[CostModel] = None,
) -> V2SimpleSplitResult:
    """One IS (train_pct) / one OOS (rest) split by bar count, both seeded with
    the same initial capital. OOS uses IS bars for indicator warmup."""
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty or len(df_full) < 120:
        raise ValueError(f"Not enough data for {symbol} over {period} (need >= 120 bars)")

    n = len(df_full)
    split = int(n * train_pct)
    split = max(60, min(split, n - 60))

    df_train = df_full.iloc[:split]
    df_test = df_full.iloc[split:]

    is_seg = _run_slice(strategy_type, symbol, params, df_train, None, initial_capital, cost_model)
    oos_seg = _run_slice(strategy_type, symbol, params, df_test, df_train, initial_capital, cost_model)

    wfe_val = _wfe(oos_seg.cagr, is_seg.cagr)
    return V2SimpleSplitResult(
        mode="simple",
        strategy_type=strategy_type,
        symbol=symbol,
        full_period=period,
        train_pct=train_pct,
        is_segment=is_seg,
        oos_segment=oos_seg,
        wfe=wfe_val,
        wfe_label=_wfe_label(wfe_val),
    )


def run_rolling_v2(
    strategy_type: str,
    symbol: str,
    params: dict,
    period: str = "10y",
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 100_000.0,
    cost_model: Optional[CostModel] = None,
) -> V2RollingResult:
    """Slide IS/OOS windows across full history; stitch OOS curves into one
    composite (rebased at each join) and compute a global WFE."""
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty:
        raise ValueError(f"No data for {symbol}")

    n = len(df_full)
    train_len = max(120, int(train_years * _TRADING_DAYS_PER_YEAR))
    test_len = max(40, int(test_years * _TRADING_DAYS_PER_YEAR))
    step_len = max(20, int(step_years * _TRADING_DAYS_PER_YEAR))

    if train_len + test_len > n:
        raise ValueError(
            f"Not enough data: need {train_len + test_len} bars "
            f"(train {train_len} + test {test_len}) but only have {n}."
        )

    segments: list = []
    composite_equity = initial_capital
    composite_curve: list = []

    t0 = 0
    while t0 + train_len + test_len <= n:
        df_is = df_full.iloc[t0:t0 + train_len]
        df_oos = df_full.iloc[t0 + train_len:t0 + train_len + test_len]

        is_seg = _run_slice(strategy_type, symbol, params, df_is, None, initial_capital, cost_model)
        oos_seg = _run_slice(strategy_type, symbol, params, df_oos, df_is, initial_capital, cost_model)

        oos_curve = oos_seg.equity_curve
        if oos_curve:
            oos_e0 = oos_curve[0]["equity"]
            scale = composite_equity / oos_e0 if oos_e0 > 0 else 1.0
            for pt in oos_curve:
                composite_curve.append(
                    {"date": pt["date"], "equity": round(pt["equity"] * scale, 2)}
                )
            composite_equity = composite_curve[-1]["equity"]

        w = _wfe(oos_seg.cagr, is_seg.cagr)
        segments.append({
            "window": len(segments) + 1,
            "is_start": is_seg.start, "is_end": is_seg.end, "is_bars": is_seg.bars,
            "is_total_return_pct": is_seg.total_return_pct, "is_cagr": is_seg.cagr,
            "is_win_rate_pct": is_seg.win_rate_pct, "is_trades": is_seg.trades,
            "oos_start": oos_seg.start, "oos_end": oos_seg.end, "oos_bars": oos_seg.bars,
            "oos_total_return_pct": oos_seg.total_return_pct, "oos_cagr": oos_seg.cagr,
            "oos_win_rate_pct": oos_seg.win_rate_pct, "oos_trades": oos_seg.trades,
            "oos_max_drawdown_pct": oos_seg.max_drawdown_pct,
            "wfe": w, "wfe_label": _wfe_label(w),
        })
        t0 += step_len

    if not segments:
        raise ValueError("No complete IS/OOS windows fit in the available data.")

    global_oos_tr = calc_total_return(composite_curve)
    global_oos_cagr = calc_cagr(composite_curve)
    is_cagrs = [s["is_cagr"] for s in segments]
    avg_is_cagr = sum(is_cagrs) / len(is_cagrs) if is_cagrs else 0.0
    global_wfe = _wfe(global_oos_cagr, avg_is_cagr)

    return V2RollingResult(
        mode="rolling",
        strategy_type=strategy_type,
        symbol=symbol,
        full_period=period,
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
        segments=segments,
        oos_composite_curve=composite_curve,
        global_oos_total_return_pct=global_oos_tr,
        global_oos_cagr=global_oos_cagr,
        avg_is_cagr=round(avg_is_cagr, 4),
        global_wfe=global_wfe,
        global_wfe_label=_wfe_label(global_wfe),
    )


def to_dict(result) -> dict:
    """Flatten a V2 result dataclass (and its nested segments) to JSON-safe dict."""
    return asdict(result)
