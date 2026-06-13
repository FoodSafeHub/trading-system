from __future__ import annotations

"""
Target-allocation backtest for the M1 advisor.

Question answered: does periodically REBALANCING the portfolio toward the
capped, momentum-weighted target weights beat (a) holding the current M1 weights
and (b) naive equal weight?

Method
------
Lump-sum portfolio (initial_capital), rebalanced every `rebalance_days`. At each
rebalance date the target weights are recomputed FROM PRICE DATA UP TO THAT DATE
(no lookahead) using the SAME scoring spirit as targets.compute_targets:
  score = signal_factor x momentum_factor x risk_factor
A lightweight, backtest-safe signal proxy (price vs SMA50/SMA200) stands in for
the live strategy panel so we can score thousands of (symbol, date) cells fast.
Caps are applied via the same _cap_and_redistribute. Within-pie then across-pie,
exactly like the live engine.

Benchmarks:
  - target  : rebalance to computed capped momentum targets.
  - current : rebalance back to the starting (M1) weights each period.
  - equal   : rebalance to equal weight across all holdings each period.

Costs not modeled (M1 is commission-free, fractional). This is a relative test:
the point is whether the target rule's curve beats the others, not absolute return.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.m1.analyzer import load_pies, PIES_PATH
from app.services.m1.targets import _cap_and_redistribute, MAX_STOCK_WEIGHT, MAX_PIE_WEIGHT
from app.services.strategy.rs_rotation import _blended, _ret_over

logger = logging.getLogger(__name__)

_MIN_BARS = 210  # need SMA200 + momentum lookback


@dataclass
class StrategyCurve:
    name: str
    final_value: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    sharpe: Optional[float]


@dataclass
class TargetBacktestResult:
    period: str
    rebalance_days: int
    initial_capital: float
    symbols_used: int
    symbols_skipped: int
    rebalances: int
    results: List[StrategyCurve] = field(default_factory=list)
    winner: str = ""
    note: str = (
        "Lump-sum, periodic rebalance, no lookahead. Costs not modeled "
        "(M1 is commission-free). Relative comparison — advisory, not financial advice."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _signal_factor(close: pd.Series) -> float:
    """Backtest-safe trend proxy → BUY/HOLD/SELL-like factor from price vs SMAs."""
    if len(close) < 200:
        return 1.0
    last = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean())
    if last > sma50 > sma200:
        return 1.5          # clean uptrend (BUY-like)
    if last < sma50 < sma200:
        return 0.25         # clean downtrend (SELL-like)
    return 1.0              # mixed (HOLD-like)


def _momentum_factor(close: pd.Series, bench_blended: float) -> float:
    r3, r6 = _ret_over(close, 63), _ret_over(close, 126)
    if r3 is None or r6 is None:
        return 1.0
    rs = _blended(r3, r6) - bench_blended
    mf = max(0.5, min(2.0, 1.0 + rs))
    if len(close) >= 200 and float(close.iloc[-1]) < float(close.tail(200).mean()):
        mf *= 0.6
    return mf


def _risk_factor(close: pd.Series, lookback: int = 63) -> float:
    if len(close) < lookback + 2:
        return 1.0
    rets = close.pct_change().dropna().tail(lookback)
    sd = float(rets.std())
    if sd <= 0:
        return 1.0
    vol = sd * (252 ** 0.5)
    return max(0.4, min(1.6, 0.30 / vol))


def _targets_asof(
    pies_raw: List[dict],
    data: Dict[str, pd.DataFrame],
    day: pd.Timestamp,
    max_stock: float,
    max_pie: float,
) -> Dict[str, float]:
    """Recompute per-SYMBOL portfolio target weights as of `day` (no lookahead).

    Returns {symbol: portfolio_weight}. A symbol in multiple pies sums its
    per-pie target * that pie's weight (so the live two-level structure holds).
    """
    # Benchmark blended return as of `day`.
    bench_blended = 0.0
    bdf = data.get("SPY")
    if bdf is not None:
        bc = bdf.loc[:day, "Close"].dropna()
        b3, b6 = _ret_over(bc, 63), _ret_over(bc, 126)
        if b3 is not None and b6 is not None:
            bench_blended = _blended(b3, b6)

    # Score each unique symbol once.
    score_cache: Dict[str, float] = {}

    def score(sym: str) -> float:
        if sym in score_cache:
            return score_cache[sym]
        df = data.get(sym)
        sc = 0.0
        if df is not None:
            close = df.loc[:day, "Close"].dropna()
            if len(close) >= _MIN_BARS:
                sc = _signal_factor(close) * _momentum_factor(close, bench_blended) * _risk_factor(close)
        score_cache[sym] = sc
        return sc

    # Within-pie capped weights, then across-pie capped weights.
    pie_weights_raw: Dict[str, float] = {}
    pie_slice_w: Dict[str, Dict[str, float]] = {}
    pie_value_proxy: Dict[str, float] = {}
    for pie in pies_raw:
        name = pie.get("name", "Pie")
        scores = {}
        for sl in pie.get("slices", []):
            sym = sl["symbol"].upper()
            scores[sym] = score(sym)
        capped = _cap_and_redistribute(scores, max_stock)
        pie_slice_w[name] = capped
        # Pie score = mean stock score (equal proxy; live uses value-weighted, but
        # historical pie $ isn't known, so use mean — keeps it lookahead-free).
        vals = [v for v in scores.values() if v > 0]
        pie_weights_raw[name] = (sum(vals) / len(vals)) if vals else 0.0

    pie_w = _cap_and_redistribute(pie_weights_raw, max_pie)

    # Combine: symbol portfolio weight = sum over pies of (slice_w * pie_w).
    sym_w: Dict[str, float] = {}
    for pie in pies_raw:
        name = pie.get("name", "Pie")
        for sym, w in pie_slice_w[name].items():
            sym_w[sym] = sym_w.get(sym, 0.0) + w * pie_w.get(name, 0.0)
    total = sum(sym_w.values())
    if total > 0:
        sym_w = {k: v / total for k, v in sym_w.items()}
    return sym_w


def _curve_stats(name: str, equity: List[float], dates: List[pd.Timestamp], periods_per_year: float) -> StrategyCurve:
    final = equity[-1]
    init = equity[0]
    total_ret = (final / init - 1.0) * 100.0 if init > 0 else 0.0
    n_years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    cagr = ((final / init) ** (1.0 / n_years) - 1.0) * 100.0 if init > 0 else 0.0
    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, v / peak - 1.0)
    # Sharpe from period returns
    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity)) if equity[i - 1] > 0]
    sharpe = None
    if len(rets) > 2:
        import statistics
        sd = statistics.pstdev(rets)
        if sd > 0:
            sharpe = round(statistics.mean(rets) / sd * (periods_per_year ** 0.5), 2)
    return StrategyCurve(
        name=name, final_value=round(final, 2),
        total_return_pct=round(total_ret, 2), cagr=round(cagr, 2),
        max_drawdown_pct=round(max_dd * 100.0, 2), sharpe=sharpe,
    )


def run_target_backtest(
    period: str = "3y",
    rebalance_days: int = 21,
    initial_capital: float = 100_000.0,
    max_stock_weight: float = MAX_STOCK_WEIGHT,
    max_pie_weight: float = MAX_PIE_WEIGHT,
    path: str = PIES_PATH,
) -> TargetBacktestResult:
    _as_of, pies_raw = load_pies(path)
    symbols = sorted({sl["symbol"].upper() for pie in pies_raw for sl in pie.get("slices", [])})

    # Load data (+SPY benchmark). tz-naive, normalized (see backtest.py note).
    data: Dict[str, pd.DataFrame] = {}
    skipped = 0
    for sym in symbols + ["SPY"]:
        try:
            df = get_ohlcv(sym, period=period, interval="1d")
            if df is not None and len(df.dropna()) >= _MIN_BARS:
                df = df.copy()
                idx = pd.to_datetime(df.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                df.index = idx.normalize()
                data[sym] = df
            elif sym != "SPY":
                skipped += 1
        except Exception:
            if sym != "SPY":
                skipped += 1

    used = [s for s in symbols if s in data]
    if not used:
        raise ValueError("No symbols had enough history for the target backtest.")

    cal = pd.DatetimeIndex(sorted(set().union(*[set(data[s].index) for s in used])))
    start_i = _MIN_BARS
    if start_i >= len(cal):
        raise ValueError("Not enough history after warmup.")
    rebal_idx = list(range(start_i, len(cal), rebalance_days))
    rebal_dates = [cal[i] for i in rebal_idx]

    # Starting (M1 current) weights from pies.json slice values → portfolio weight.
    cur_sym_val: Dict[str, float] = {}
    for pie in pies_raw:
        for sl in pie.get("slices", []):
            cur_sym_val[sl["symbol"].upper()] = cur_sym_val.get(sl["symbol"].upper(), 0.0) + float(sl.get("value", 0) or 0)
    cur_total = sum(cur_sym_val.get(s, 0.0) for s in used) or 1.0
    current_w = {s: cur_sym_val.get(s, 0.0) / cur_total for s in used}
    equal_w = {s: 1.0 / len(used) for s in used}

    def price_asof(sym: str, day: pd.Timestamp) -> Optional[float]:
        sub = data[sym].loc[:day, "Close"].dropna()
        return float(sub.iloc[-1]) if not sub.empty else None

    # Run each strategy: hold weights between rebalances, mark-to-market.
    def simulate(weight_fn) -> tuple[List[float], List[pd.Timestamp]]:
        equity = initial_capital
        shares: Dict[str, float] = {}
        curve: List[float] = []
        cdates: List[pd.Timestamp] = []
        for k, day in enumerate(rebal_dates):
            # Mark-to-market current holdings at today's prices.
            if shares:
                mv = sum(sh * (price_asof(s, day) or 0.0) for s, sh in shares.items())
                equity = mv
            # Compute target weights for this period.
            w = weight_fn(day)
            wsum = sum(w.values()) or 1.0
            w = {s: v / wsum for s, v in w.items()}
            # Rebalance: convert equity into shares at today's price.
            shares = {}
            for s, wt in w.items():
                px = price_asof(s, day)
                if px and px > 0 and wt > 0:
                    shares[s] = (equity * wt) / px
            curve.append(equity)
            cdates.append(day)
        # Final mark at last calendar day.
        last = cal[-1]
        if shares:
            equity = sum(sh * (price_asof(s, last) or 0.0) for s, sh in shares.items())
        curve.append(equity)
        cdates.append(last)
        return curve, cdates

    ppy = 252.0 / rebalance_days
    tgt_curve, tgt_dates = simulate(
        lambda day: _targets_asof(pies_raw, data, day, max_stock_weight, max_pie_weight)
    )
    cur_curve, _ = simulate(lambda day: current_w)
    eq_curve, _ = simulate(lambda day: equal_w)

    results = [
        _curve_stats("target", tgt_curve, tgt_dates, ppy),
        _curve_stats("current", cur_curve, tgt_dates, ppy),
        _curve_stats("equal", eq_curve, tgt_dates, ppy),
    ]
    results.sort(key=lambda r: r.final_value, reverse=True)
    return TargetBacktestResult(
        period=period, rebalance_days=rebalance_days, initial_capital=initial_capital,
        symbols_used=len(used), symbols_skipped=skipped,
        rebalances=len(rebal_dates), results=results,
        winner=results[0].name if results else "",
    )
