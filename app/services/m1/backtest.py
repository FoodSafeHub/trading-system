from __future__ import annotations

"""
Contribution-tilt backtest for the M1 advisor.

Question answered: when you add a fixed amount of new money on a schedule and
you NEVER sell (long-term pies), does steering each contribution toward
BUY-signal names ("aggressive tilt") beat naive splits?

Benchmarks:
  - aggressive : the live rule — fund only net-BUY names, weighted by conviction.
  - equal      : split each contribution equally across all holdings.
  - weight     : split each contribution by current portfolio $ weight.

No lookahead: at each contribution date the strategy panel sees only prices up
to and including that date. Shares bought are held to the end (buy-and-accumulate;
no selling — matches how M1 pies are actually used).

This reuses the SAME panel + consensus + conviction as the live analyzer, so a
favorable result here is evidence for the live rule, not a different model.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.strategy.rules import evaluate_strategy
from app.services.m1.analyzer import (
    DEFAULT_PANEL, HOLDINGS_PATH, load_holdings, _consensus, _conviction,
)

logger = logging.getLogger(__name__)

_MIN_BARS = 120  # need enough history for the trend panel (SMA/ATR warmup)


@dataclass
class StrategyResult:
    name: str
    contributed: float
    final_value: float
    profit: float
    return_pct: float
    contributions: int


@dataclass
class BacktestResult:
    period: str
    cadence: str
    contribution_per_period: float
    symbols_used: int
    symbols_skipped: int
    contribution_dates: int
    results: List[StrategyResult] = field(default_factory=list)
    winner: str = ""
    note: str = (
        "Buy-and-accumulate (never sells). No lookahead. Costs not modeled. "
        "Advisory backtest — not financial advice."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _panel_consensus(df_slice: pd.DataFrame, panel) -> tuple[str, float]:
    """Run the panel on a price slice (no lookahead) → (direction, conviction)."""
    prices = df_slice["Close"].dropna()
    per: Dict[str, str] = {}
    for stype, params in panel:
        try:
            sig = evaluate_strategy(stype, "BT", prices, params, ohlcv=df_slice)
            per[stype] = sig.direction
        except Exception:
            per[stype] = "HOLD"
    direction, buys, sells, _ = _consensus(per)
    conviction = _conviction(direction, buys, sells, len(per))
    return direction, conviction


def _tilt_weights(mode: str, syms, signals, values) -> Dict[str, float]:
    """Allocation weights for a single contribution. Mirrors analyzer.compute_tilt."""
    raw: Dict[str, float] = {}
    for s in syms:
        direction, conviction = signals[s]
        if mode == "aggressive":
            raw[s] = (conviction + 10.0) if direction == "BUY" else 0.0
        elif mode == "equal":
            raw[s] = 1.0
        elif mode == "weight":
            raw[s] = max(values[s], 1e-9)
        else:
            raw[s] = 0.0
    total = sum(raw.values())
    if total <= 0:
        # No BUYs this period (aggressive) → fall back to equal so cash isn't lost.
        n = len(syms)
        return {s: 1.0 / n for s in syms} if n else {}
    return {s: raw[s] / total for s in syms}


def run_tilt_backtest(
    contribution: float = 100.0,
    cadence: str = "weekly",          # "weekly" | "monthly"
    period: str = "2y",
    panel=None,
    path: str = HOLDINGS_PATH,
    modes: Optional[List[str]] = None,
) -> BacktestResult:
    panel = panel or DEFAULT_PANEL
    modes = modes or ["aggressive", "equal", "weight"]
    _as_of, holdings = load_holdings(path)
    symbols = [h["symbol"].upper() for h in holdings]

    # Load data once per symbol; keep those with enough history.
    data: Dict[str, pd.DataFrame] = {}
    skipped = 0
    for sym in symbols:
        try:
            df = get_ohlcv(sym, period=period, interval="1d")
            if df is not None and len(df.dropna()) >= _MIN_BARS:
                df = df.copy()
                idx = pd.to_datetime(df.index)
                # Normalize to tz-naive so cross-symbol calendars unify cleanly
                # (yfinance returns tz-aware; ADRs/ETFs can differ).
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                df.index = idx.normalize()
                data[sym] = df
            else:
                skipped += 1
        except Exception as exc:
            logger.debug("[m1bt] %s skipped: %s", sym, exc)
            skipped += 1

    if not data:
        raise ValueError("No symbols had enough history for the backtest.")

    used = sorted(data.keys())

    # Unified calendar from the union of trading days.
    all_days = sorted(set().union(*[set(df.index) for df in data.values()]))
    cal = pd.DatetimeIndex(all_days)

    # Contribution dates: first trading day of each week/month, after warmup.
    warmup_cutoff = cal[min(_MIN_BARS, len(cal) - 1)]
    freq = "W" if cadence == "weekly" else "MS"
    buckets = pd.Series(cal, index=cal).groupby(pd.Grouper(freq=freq)).first().dropna()
    contrib_dates = [d for d in buckets.values if pd.Timestamp(d) >= warmup_cutoff]
    contrib_dates = [pd.Timestamp(d) for d in contrib_dates]

    # shares[mode][sym] accumulates.
    shares: Dict[str, Dict[str, float]] = {m: {s: 0.0 for s in used} for m in modes}

    def price_asof(sym: str, day: pd.Timestamp) -> Optional[float]:
        df = data[sym]
        sub = df.loc[:day]
        if sub.empty:
            return None
        return float(sub["Close"].iloc[-1])

    for day in contrib_dates:
        # Per-symbol signal + current price, as of `day` (no lookahead).
        signals: Dict[str, tuple[str, float]] = {}
        prices_today: Dict[str, float] = {}
        values_today: Dict[str, float] = {}
        eligible: List[str] = []
        for sym in used:
            df = data[sym]
            sl = df.loc[:day]
            if len(sl.dropna()) < _MIN_BARS:
                continue
            px = float(sl["Close"].iloc[-1])
            prices_today[sym] = px
            signals[sym] = _panel_consensus(sl, panel)
            # current value of accumulated shares (use aggressive book for the
            # weight benchmark's reference — but weights are per-mode below)
            eligible.append(sym)
        if not eligible:
            continue

        for mode in modes:
            # value map for the "weight" mode = this mode's own current holdings $
            values_today = {
                s: shares[mode][s] * prices_today[s] for s in eligible
            }
            # if all-zero (first contribution), weight mode degrades to equal
            if mode == "weight" and sum(values_today.values()) <= 0:
                w = {s: 1.0 / len(eligible) for s in eligible}
            else:
                w = _tilt_weights(mode, eligible, signals, values_today)
            for s in eligible:
                dollars = contribution * w.get(s, 0.0)
                if dollars > 0 and prices_today[s] > 0:
                    shares[mode][s] += dollars / prices_today[s]

    # Final valuation at the last calendar day.
    last_day = cal[-1]
    n_contrib = len(contrib_dates)
    total_contributed = contribution * n_contrib

    results: List[StrategyResult] = []
    for mode in modes:
        final_val = 0.0
        for s in used:
            px = price_asof(s, last_day)
            if px:
                final_val += shares[mode][s] * px
        profit = final_val - total_contributed
        ret = (profit / total_contributed * 100.0) if total_contributed > 0 else 0.0
        results.append(StrategyResult(
            name=mode, contributed=round(total_contributed, 2),
            final_value=round(final_val, 2), profit=round(profit, 2),
            return_pct=round(ret, 2), contributions=n_contrib,
        ))

    results.sort(key=lambda r: r.final_value, reverse=True)
    return BacktestResult(
        period=period, cadence=cadence, contribution_per_period=contribution,
        symbols_used=len(used), symbols_skipped=skipped,
        contribution_dates=n_contrib, results=results,
        winner=results[0].name if results else "",
    )
