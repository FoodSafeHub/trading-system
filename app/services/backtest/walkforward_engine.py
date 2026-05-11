from __future__ import annotations

"""
Walk-Forward Validation engine for Perplexity strategies.

Two modes
---------
1. Simple 70/30 split  — one IS window, one OOS window.
2. Rolling walk-forward — multiple IS/OOS windows sliding across full history;
   OOS equity curves are stitched end-to-end into a single composite curve.

Key metric: Walk-Forward Efficiency (WFE)
    WFE = OOS_CAGR / IS_CAGR

Interpretation
    ~1.0   : OOS matches IS — excellent robustness
    0.7–1.0: Slight degradation — acceptable for real strategies
    0.5–0.7: Yellow zone — investigate but not an automatic rejection
    0.3–0.5: Large degradation — possible overfitting or regime change
    <  0.3 : Red flag — IS performance unlikely to repeat going forward
"""

import statistics as _stats
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from app.services.backtest.perplexity_engine import PerplexityBacktestResult
from app.services.market_data.provider import get_ohlcv
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.perplexity.base import PerplexityStrategy

_TRADING_DAYS_PER_YEAR = 252


# ── Metric helpers ────────────────────────────────────────────────────────────

def calc_total_return(equity_curve: list) -> float:
    """(E_final / E_initial) - 1, expressed as a percentage."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e0 = equity_curve[0]["equity"]
    e1 = equity_curve[-1]["equity"]
    if e0 <= 0:
        return 0.0
    return round((e1 / e0 - 1) * 100, 4)


def calc_cagr(equity_curve: list) -> float:
    """
    Compound annual growth rate (%), annualised over 252 trading days/year.
    Returns 0.0 for flat or empty curves.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e0 = equity_curve[0]["equity"]
    e1 = equity_curve[-1]["equity"]
    if e0 <= 0 or e1 <= 0:
        return 0.0
    years = len(equity_curve) / _TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return round(((e1 / e0) ** (1.0 / years) - 1) * 100, 4)


def _wfe(oos_cagr: float, is_cagr: float) -> Optional[float]:
    """
    Walk-Forward Efficiency = OOS_CAGR / IS_CAGR.
    Returns None when IS_CAGR <= 0 (no meaningful IS edge to compare against).
    """
    if is_cagr <= 0:
        return None
    return round(oos_cagr / is_cagr, 4)


def _wfe_label(wfe: Optional[float]) -> str:
    if wfe is None:
        return "N/A (no IS edge)"
    if wfe >= 1.0:
        return "Excellent"
    if wfe >= 0.7:
        return "Acceptable"
    if wfe >= 0.5:
        return "Yellow — investigate"
    if wfe >= 0.3:
        return "Degraded — possible overfit"
    return "Red flag — likely overfit"


# ── Segment result ────────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """Metrics for one IS or OOS window."""
    start: str
    end: str
    bars: int
    total_return_pct: float
    cagr: float
    win_rate_pct: float
    profit_factor: Optional[float]
    max_drawdown_pct: float
    trades: int
    sharpe: Optional[float]
    equity_curve: List[dict] = field(default_factory=list)


# ── Top-level result objects ──────────────────────────────────────────────────

@dataclass
class SimpleSplitResult:
    mode: str                       # "simple"
    strategy_name: str
    symbol: str
    full_period: str
    train_pct: float

    is_segment: SegmentResult
    oos_segment: SegmentResult

    wfe: Optional[float]            # OOS_CAGR / IS_CAGR
    wfe_label: str
    oos_pf_ratio: Optional[float]   # OOS_PF / IS_PF


@dataclass
class RollingWalkForwardResult:
    mode: str                       # "rolling"
    strategy_name: str
    symbol: str
    full_period: str
    train_years: float
    test_years: float
    step_years: float

    segments: List[dict]            # per-window IS+OOS metrics

    # Composite OOS (all OOS windows stitched end-to-end, rebased at each join)
    oos_composite_curve: List[dict]
    global_oos_total_return_pct: float
    global_oos_cagr: float
    avg_is_cagr: float
    global_wfe: Optional[float]
    global_wfe_label: str


# ── Core backtest-on-slice ─────────────────────────────────────────────────────

def _backtest_on_slice(
    strategy: PerplexityStrategy,
    symbol: str,
    df_slice: pd.DataFrame,
    initial_capital: float,
    position_pct: float,
    warmup_df: Optional[pd.DataFrame] = None,
) -> PerplexityBacktestResult:
    """
    Run the strategy bar-by-bar on df_slice.
    warmup_df, if given, is prepended so indicators have history but no trades
    fire during the warmup bars.
    """
    if warmup_df is not None and not warmup_df.empty:
        df_full = pd.concat([warmup_df, df_slice])
        start_bar = len(warmup_df)
    else:
        df_full = df_slice
        start_bar = 0

    lookback = min(210, max(60, len(df_full) // 3))
    effective_start = max(lookback, start_bar)

    dates = [str(d)[:10] for d in df_full.index]
    capital = initial_capital
    position = 0.0
    position_cost = 0.0
    entry_stop: Optional[float] = None
    entry_target: Optional[float] = None
    open_risk_usd = 0.0
    trades: list = []
    equity_curve: list = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: list = []
    prev_equity = initial_capital

    for i in range(effective_start, len(df_full)):
        df_bar = df_full.iloc[:i]
        current_close = float(df_full["Close"].iloc[i])
        fill_price    = float(df_full["Open"].iloc[i])
        today = dates[i]

        # ── Gap stop / target check on open ──
        if position > 0 and entry_stop is not None:
            op = float(df_full["Open"].iloc[i])
            if op <= entry_stop:
                proceeds = op * position; pnl = proceeds - position_cost
                capital += proceeds
                trades.append({"date": today, "side": "SELL (stop)", "price": round(op, 2),
                                "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0
            elif entry_target and op >= entry_target:
                proceeds = op * position; pnl = proceeds - position_cost
                capital += proceeds
                trades.append({"date": today, "side": "SELL (target)", "price": round(op, 2),
                                "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        # ── Intraday stop / target check on H/L ──
        if position > 0 and entry_stop is not None:
            lo = float(df_full["Low"].iloc[i])
            hi = float(df_full["High"].iloc[i])
            if lo <= entry_stop:
                proceeds = entry_stop * position; pnl = proceeds - position_cost
                capital += proceeds
                trades.append({"date": today, "side": "SELL (stop)", "price": round(entry_stop, 2),
                                "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0
            elif entry_target and hi >= entry_target:
                proceeds = entry_target * position; pnl = proceeds - position_cost
                capital += proceeds
                trades.append({"date": today, "side": "SELL (target)", "price": round(entry_target, 2),
                                "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        # ── Strategy signal ──
        if position == 0:
            try:
                sig = strategy.run(symbol, df_bar)
            except Exception:
                sig = None
            if sig and sig.direction == "BUY":
                if position_pct > 0:
                    alloc = capital * min(position_pct, 0.95)
                    qty = alloc / fill_price if fill_price > 0 else 0.0
                    trade_risk = (fill_price - sig.stop_price) * qty if sig.stop_price else 0.0
                elif sig.stop_price and sig.stop_price < fill_price:
                    sz = calculate_position_size(
                        symbol=symbol, entry_price=fill_price, stop_price=sig.stop_price,
                        account_value=capital, risk_pct_per_trade=0.01,
                        max_position_size_usd=capital * 0.20,
                        current_open_risk_usd=open_risk_usd,
                    )
                    qty = sz.shares if sz.viable else 0.0
                    trade_risk = sz.risk_amount if sz.viable else 0.0
                else:
                    qty = (capital * 0.10) / fill_price if fill_price > 0 else 0.0
                    trade_risk = 0.0
                qty = round(qty, 6)
                cost = fill_price * qty
                if qty >= 0.001 and cost <= capital:
                    capital -= cost; position = qty; position_cost = cost
                    open_risk_usd += trade_risk
                    entry_stop = sig.stop_price; entry_target = sig.target_price
                    trades.append({"date": today, "side": "BUY", "price": round(fill_price, 2),
                                   "quantity": round(qty, 4), "value": round(cost, 2), "pnl": None})
        elif position > 0:
            try:
                sig = strategy.run(symbol, df_bar)
            except Exception:
                sig = None
            if sig and sig.direction == "SELL":
                proceeds = fill_price * position; pnl = proceeds - position_cost
                capital += proceeds
                trades.append({"date": today, "side": "SELL", "price": round(fill_price, 2),
                               "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0

        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        daily_returns.append(ret)
        prev_equity = equity

    # Close open position at last bar
    if position > 0:
        lp = float(df_full["Close"].iloc[-1])
        proceeds = lp * position; pnl = proceeds - position_cost
        capital += proceeds
        trades.append({"date": dates[-1], "side": "SELL (close)", "price": round(lp, 2),
                       "quantity": position, "value": round(proceeds, 2), "pnl": round(pnl, 2)})

    final_capital = capital
    total_pnl = final_capital - initial_capital
    sell_trades = [t for t in trades if "SELL" in t["side"] and t.get("pnl") is not None]
    winning = [t for t in sell_trades if (t.get("pnl") or 0) > 0]
    losing  = [t for t in sell_trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0.0
    gross_wins   = sum(t["pnl"] for t in winning)
    gross_losses = abs(sum(t["pnl"] for t in losing))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    sharpe = None
    if len(daily_returns) > 1:
        avg = _stats.mean(daily_returns)
        std = _stats.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (_TRADING_DAYS_PER_YEAR ** 0.5), 2)

    from app.services.backtest.perplexity_engine import calc_total_return as _tr, calc_cagr as _cagr
    return PerplexityBacktestResult(
        strategy_name=strategy.name, symbol=symbol, period="slice",
        start_date=dates[effective_start] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_pnl=round(total_pnl, 2),
        capital_employed=sum(t["value"] for t in trades if t["side"] == "BUY"),
        total_return_pct=_tr(equity_curve),
        cagr=_cagr(equity_curve),
        total_trades=len(sell_trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate_pct=round(win_rate, 2),
        profit_factor=profit_factor,
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )


def _to_segment(result: PerplexityBacktestResult, bars: int) -> SegmentResult:
    return SegmentResult(
        start=result.start_date,
        end=result.end_date,
        bars=bars,
        total_return_pct=result.total_return_pct,
        cagr=result.cagr,
        win_rate_pct=result.win_rate_pct,
        profit_factor=result.profit_factor,
        max_drawdown_pct=result.max_drawdown_pct,
        trades=result.total_trades,
        sharpe=result.sharpe_ratio,
        equity_curve=result.equity_curve,
    )


# ── Mode 1: Simple split ──────────────────────────────────────────────────────

def run_simple_split(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "10y",
    train_pct: float = 0.70,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
) -> SimpleSplitResult:
    """
    Split full history into IS (train_pct) and OOS (1 - train_pct) by bar count.
    Both windows start with the same initial_capital so metrics are comparable.
    OOS backtest uses IS data for indicator warmup (no lookahead).
    """
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty or len(df_full) < 120:
        raise ValueError(f"Not enough data for {symbol} over {period} (need ≥ 120 bars)")

    n = len(df_full)
    split = int(n * train_pct)
    split = max(split, 60)
    split = min(split, n - 60)

    df_train = df_full.iloc[:split]
    df_test  = df_full.iloc[split:]

    is_result  = _backtest_on_slice(strategy, symbol, df_train, initial_capital, position_pct)
    oos_result = _backtest_on_slice(strategy, symbol, df_test,  initial_capital, position_pct,
                                    warmup_df=df_train)

    is_seg  = _to_segment(is_result,  len(df_train))
    oos_seg = _to_segment(oos_result, len(df_test))

    wfe_val = _wfe(oos_seg.cagr, is_seg.cagr)

    def _pf_ratio(oos_pf, is_pf):
        if oos_pf is None or is_pf is None or is_pf == 0:
            return None
        return round(oos_pf / is_pf, 4)

    return SimpleSplitResult(
        mode="simple",
        strategy_name=strategy.name,
        symbol=symbol,
        full_period=period,
        train_pct=train_pct,
        is_segment=is_seg,
        oos_segment=oos_seg,
        wfe=wfe_val,
        wfe_label=_wfe_label(wfe_val),
        oos_pf_ratio=_pf_ratio(oos_seg.profit_factor, is_seg.profit_factor),
    )


# ── Mode 2: Rolling walk-forward ──────────────────────────────────────────────

def run_rolling_walk_forward(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "10y",
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
) -> RollingWalkForwardResult:
    """
    Slide IS/OOS windows across the full history.

    Window layout (bars):
        IS  = [t0 : t0 + train_len)
        OOS = [t0 + train_len : t0 + train_len + test_len)
    Step forward by step_len bars each iteration.

    OOS equity curves are rebased and stitched end-to-end into one composite
    curve. Global OOS CAGR is computed from that composite curve.
    WFE = global_OOS_CAGR / avg_IS_CAGR across all segments.
    """
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty:
        raise ValueError(f"No data for {symbol}")

    n = len(df_full)
    train_len = max(60, int(train_years * _TRADING_DAYS_PER_YEAR))
    test_len  = max(20, int(test_years  * _TRADING_DAYS_PER_YEAR))
    step_len  = max(10, int(step_years  * _TRADING_DAYS_PER_YEAR))

    if train_len + test_len > n:
        raise ValueError(
            f"Not enough data: need {train_len + test_len} bars "
            f"(train {train_len} + test {test_len}) but only have {n}."
        )

    segments: list = []
    # Composite OOS: list of equity values rebased so each segment starts
    # where the previous one ended (capital carried forward).
    composite_equity: float = initial_capital
    composite_curve: list = []

    t0 = 0
    while t0 + train_len + test_len <= n:
        df_is  = df_full.iloc[t0 : t0 + train_len]
        df_oos = df_full.iloc[t0 + train_len : t0 + train_len + test_len]

        is_result  = _backtest_on_slice(strategy, symbol, df_is,  initial_capital, position_pct)
        oos_result = _backtest_on_slice(strategy, symbol, df_oos, initial_capital, position_pct,
                                        warmup_df=df_is)

        is_seg  = _to_segment(is_result,  len(df_is))
        oos_seg = _to_segment(oos_result, len(df_oos))

        # Rebase OOS equity onto composite_equity
        oos_curve = oos_seg.equity_curve
        if oos_curve:
            oos_e0 = oos_curve[0]["equity"]
            scale = composite_equity / oos_e0 if oos_e0 > 0 else 1.0
            for pt in oos_curve:
                rebased = round(pt["equity"] * scale, 2)
                composite_curve.append({"date": pt["date"], "equity": rebased})
            composite_equity = composite_curve[-1]["equity"]

        segments.append({
            "window": len(segments) + 1,
            "is_start": is_seg.start, "is_end": is_seg.end, "is_bars": is_seg.bars,
            "is_total_return_pct": is_seg.total_return_pct, "is_cagr": is_seg.cagr,
            "is_win_rate_pct": is_seg.win_rate_pct, "is_profit_factor": is_seg.profit_factor,
            "is_max_drawdown_pct": is_seg.max_drawdown_pct, "is_trades": is_seg.trades,
            "oos_start": oos_seg.start, "oos_end": oos_seg.end, "oos_bars": oos_seg.bars,
            "oos_total_return_pct": oos_seg.total_return_pct, "oos_cagr": oos_seg.cagr,
            "oos_win_rate_pct": oos_seg.win_rate_pct, "oos_profit_factor": oos_seg.profit_factor,
            "oos_max_drawdown_pct": oos_seg.max_drawdown_pct, "oos_trades": oos_seg.trades,
            "wfe": _wfe(oos_seg.cagr, is_seg.cagr),
            "wfe_label": _wfe_label(_wfe(oos_seg.cagr, is_seg.cagr)),
        })

        t0 += step_len

    if not segments:
        raise ValueError("No complete IS/OOS windows fit in the available data.")

    global_oos_tr   = calc_total_return(composite_curve)
    global_oos_cagr = calc_cagr(composite_curve)

    is_cagrs = [s["is_cagr"] for s in segments]
    avg_is_cagr = sum(is_cagrs) / len(is_cagrs) if is_cagrs else 0.0

    global_wfe = _wfe(global_oos_cagr, avg_is_cagr)

    return RollingWalkForwardResult(
        mode="rolling",
        strategy_name=strategy.name,
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


# ── Multi-strategy comparison on one symbol ──────────────────────────────────

@dataclass
class StrategyWFESummary:
    strategy_name: str
    symbol: str
    n_windows: int
    global_wfe: Optional[float]
    global_wfe_label: str
    global_oos_cagr: float
    global_oos_total_return_pct: float
    avg_is_cagr: float
    # Most recent window IS/OOS CAGR for "what's working now" read
    latest_is_cagr: float
    latest_oos_cagr: float
    latest_wfe: Optional[float]
    latest_wfe_label: str
    # Second-to-last window OOS (the most recent truly held-out period)
    prev_oos_cagr: Optional[float]
    error: Optional[str] = None


def run_walkforward_all(
    strategies: list,
    symbol: str,
    period: str = "10y",
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
) -> List[StrategyWFESummary]:
    """
    Run rolling walk-forward for every strategy in `strategies` on the same
    symbol. Returns one StrategyWFESummary per strategy — the key "which
    strategy is working NOW" comparison table.
    """
    results: List[StrategyWFESummary] = []
    for strategy in strategies:
        try:
            r = run_rolling_walk_forward(
                strategy, symbol, period=period,
                train_years=train_years, test_years=test_years, step_years=step_years,
                initial_capital=initial_capital, position_pct=position_pct,
            )
            segs = r.segments
            latest = segs[-1] if segs else None
            prev   = segs[-2] if len(segs) >= 2 else None
            results.append(StrategyWFESummary(
                strategy_name=r.strategy_name,
                symbol=r.symbol,
                n_windows=len(segs),
                global_wfe=r.global_wfe,
                global_wfe_label=r.global_wfe_label,
                global_oos_cagr=r.global_oos_cagr,
                global_oos_total_return_pct=r.global_oos_total_return_pct,
                avg_is_cagr=r.avg_is_cagr,
                latest_is_cagr=latest["is_cagr"] if latest else 0.0,
                latest_oos_cagr=latest["oos_cagr"] if latest else 0.0,
                latest_wfe=latest["wfe"] if latest else None,
                latest_wfe_label=latest["wfe_label"] if latest else "N/A",
                prev_oos_cagr=prev["oos_cagr"] if prev else None,
            ))
        except Exception as exc:
            results.append(StrategyWFESummary(
                strategy_name=strategy.name, symbol=symbol,
                n_windows=0, global_wfe=None, global_wfe_label="N/A",
                global_oos_cagr=0.0, global_oos_total_return_pct=0.0,
                avg_is_cagr=0.0, latest_is_cagr=0.0, latest_oos_cagr=0.0,
                latest_wfe=None, latest_wfe_label="N/A", prev_oos_cagr=None,
                error=str(exc),
            ))
    return results


# ── Unified entry point (keeps old API compatible) ────────────────────────────

def run_walk_forward(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "10y",
    train_pct: float = 0.70,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
) -> SimpleSplitResult:
    """Backward-compatible wrapper — runs simple split mode."""
    return run_simple_split(strategy, symbol, period, train_pct, initial_capital, position_pct)
