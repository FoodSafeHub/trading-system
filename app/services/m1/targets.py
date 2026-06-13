from __future__ import annotations

"""
M1 target-allocation engine — ADVISORY ONLY.

Answers: "how much SHOULD I hold in each pie and each stock?" — i.e. recommended
TARGET weights, not just where the next deposit goes.

Per-stock target_score = signal_factor x momentum_factor x risk_factor
  - signal_factor   : trend panel (BUY > HOLD > SELL; laggard/SELL floored low)
  - momentum_factor : relative strength vs SPY (blended 3m/6m, gated by SMA200)
  - risk_factor     : inverse volatility (choppier name -> smaller raw weight)

Weights are normalized WITHIN each pie (capped per stock), and ACROSS pies
(capped per pie), via iterative cap-and-redistribute so caps hold without
leaving probability mass unassigned.

Current weights are derived from M1 slice $ values. Drift = target - current.
The contribution (add-only) is steered to close the largest UNDER-weight gaps
first — never sells. Reuses the same panel + dip/laggard signals as the rest of
the advisor (analyzer.analyze_holding) and the relative-strength primitive from
rs_rotation.
"""

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.m1.analyzer import (
    DEFAULT_PANEL, PIES_PATH, analyze_holding, load_pies,
)

logger = logging.getLogger(__name__)

# Caps (user-chosen "moderate"): no stock > 10% of its pie, no pie > 35% overall.
MAX_STOCK_WEIGHT = 0.10
MAX_PIE_WEIGHT = 0.35

_SIGNAL_FACTOR = {"BUY": 1.5, "HOLD": 1.0, "SELL": 0.25}


@dataclass
class TargetSlice:
    symbol: str
    current_value: float
    current_weight: float        # within-pie, from $ value
    target_weight: float         # within-pie, computed
    drift: float                 # target - current (within-pie)
    score: float                 # raw target_score (pre-normalization)
    direction: str
    momentum_rs: Optional[float]
    volatility: Optional[float]
    laggard: bool
    suggested_dollars: float = 0.0   # share of contribution to this slice
    note: str = ""


@dataclass
class TargetPie:
    name: str
    current_value: float
    current_weight: float        # pie share of whole portfolio, from $ value
    target_weight: float         # computed pie share
    drift: float
    score: float
    suggested_dollars: float = 0.0
    slices: List[TargetSlice] = field(default_factory=list)


@dataclass
class TargetAllocation:
    as_of: Optional[str]
    contribution: float
    max_stock_weight: float
    max_pie_weight: float
    pies: List[TargetPie]
    advisory_note: str = (
        "Advisory only. Target weights are a recommended destination to set in "
        "M1's pie editor; the contribution figures move you toward them without "
        "selling. Not financial advice."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _volatility(prices: pd.Series, lookback: int = 63) -> Optional[float]:
    """Annualized daily-return volatility over the lookback (~3mo)."""
    if len(prices) < lookback + 2:
        return None
    rets = prices.pct_change().dropna().tail(lookback)
    if rets.empty:
        return None
    sd = float(rets.std())
    return sd * math.sqrt(252) if sd > 0 else None


def _cap_and_redistribute(weights: Dict[str, float], cap: float, iters: int = 50) -> Dict[str, float]:
    """Normalize ``weights`` to sum 1.0 with no element exceeding ``cap``.

    Iteratively: normalize, clamp anything over cap, then redistribute the freed
    mass across the still-uncapped elements (proportional to their weight).
    Converges quickly. If cap*n < 1 the cap is mathematically infeasible (n items
    each <= cap can't sum to 1) — we fall back to EQUAL weights, which is the best
    achievable spread even though each equal share then exceeds the cap. In real
    pies (20+ slices at a 10% cap) this branch never triggers; it only guards
    against a pathological caller.
    """
    keys = [k for k, v in weights.items() if v > 0]
    if not keys:
        return {k: 0.0 for k in weights}
    if cap * len(keys) <= 1.0 + 1e-9:
        eq = 1.0 / len(keys)
        return {k: (eq if k in keys else 0.0) for k in weights}

    w = {k: max(weights.get(k, 0.0), 0.0) for k in weights}
    for _ in range(iters):
        total = sum(w.values())
        if total <= 0:
            break
        w = {k: v / total for k, v in w.items()}
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        capped_mass = sum(cap for _ in over)
        free_mass = 1.0 - capped_mass
        uncapped = {k: v for k, v in w.items() if k not in over and v > 0}
        un_total = sum(uncapped.values())
        new = {}
        for k in w:
            if k in over:
                new[k] = cap
            elif un_total > 0 and k in uncapped:
                new[k] = free_mass * (uncapped[k] / un_total)
            else:
                new[k] = 0.0
        w = new
    return w


def compute_targets(
    contribution: float = 0.0,
    panel: Optional[List[tuple[str, Dict[str, Any]]]] = None,
    period: str = "1y",
    benchmark: str = "SPY",
    path: str = PIES_PATH,
    max_stock_weight: float = MAX_STOCK_WEIGHT,
    max_pie_weight: float = MAX_PIE_WEIGHT,
) -> TargetAllocation:
    panel = panel or DEFAULT_PANEL
    as_of, pies_raw = load_pies(path)

    # 0) Benchmark blended return for relative strength.
    from app.services.strategy.rs_rotation import _blended, _ret_over
    bench_blended = 0.0
    try:
        bdf = get_ohlcv(benchmark, period="2y")
        b3, b6 = _ret_over(bdf["Close"], 63), _ret_over(bdf["Close"], 126)
        if b3 is not None and b6 is not None:
            bench_blended = _blended(b3, b6)
    except Exception:
        pass

    # 1) Per-unique-symbol signal + momentum + volatility (computed once).
    uniq: Dict[str, Dict[str, Any]] = {}
    for pie in pies_raw:
        for sl in pie.get("slices", []):
            sym = sl["symbol"].upper()
            if sym in uniq:
                continue
            sig = analyze_holding({"symbol": sym, "name": sym, "value": 0.0}, panel, period=period)
            rs = None
            vol = None
            try:
                df = get_ohlcv(sym, period="2y")
                close = df["Close"].dropna()
                r3, r6 = _ret_over(close, 63), _ret_over(close, 126)
                if r3 is not None and r6 is not None:
                    rs = round(_blended(r3, r6) - bench_blended, 4)
                vol = _volatility(close)
            except Exception:
                pass
            uniq[sym] = {"sig": sig, "rs": rs, "vol": vol}

    # 2) Score each unique symbol.
    def score_for(sym: str) -> float:
        d = uniq[sym]
        sig = d["sig"]
        if sig.error is not None:
            return 0.0
        signal_factor = _SIGNAL_FACTOR.get(sig.direction, 1.0)
        if sig.laggard:
            signal_factor = min(signal_factor, 0.15)   # broken-down name: shrink, don't zero
        # Momentum: map RS (can be negative) into a positive multiplier ~[0.5, 2.0].
        rs = d["rs"]
        if rs is None:
            momentum_factor = 1.0
        else:
            momentum_factor = max(0.5, min(2.0, 1.0 + rs))   # +50% RS -> 1.5x
        if sig.above_sma200 is False:
            momentum_factor *= 0.6                            # downtrend damp
        # Risk: inverse vol, normalized to a ~1.0 baseline (30% vol).
        vol = d["vol"]
        risk_factor = 1.0 if not vol else max(0.4, min(1.6, 0.30 / vol))
        return signal_factor * momentum_factor * risk_factor

    # 3) Build pies with within-pie capped target weights.
    target_pies: List[TargetPie] = []
    for pie in pies_raw:
        raw_scores: Dict[str, float] = {}
        slice_meta: Dict[str, Dict[str, Any]] = {}
        for sl in pie.get("slices", []):
            sym = sl["symbol"].upper()
            val = float(sl.get("value", 0.0) or 0.0)
            sc = score_for(sym)
            raw_scores[sym] = sc
            slice_meta[sym] = {"value": val, "score": sc}

        pie_value = sum(m["value"] for m in slice_meta.values())
        target_w = _cap_and_redistribute(raw_scores, max_stock_weight)

        slices: List[TargetSlice] = []
        for sym, m in slice_meta.items():
            d = uniq[sym]
            sig = d["sig"]
            cur_w = (m["value"] / pie_value) if pie_value > 0 else 0.0
            tgt_w = target_w.get(sym, 0.0)
            slices.append(TargetSlice(
                symbol=sym, current_value=round(m["value"], 2),
                current_weight=round(cur_w, 4), target_weight=round(tgt_w, 4),
                drift=round(tgt_w - cur_w, 4), score=round(m["score"], 4),
                direction=sig.direction, momentum_rs=d["rs"],
                volatility=round(d["vol"], 3) if d["vol"] else None,
                laggard=sig.laggard, note=sig.note,
            ))
        # Pie score = value-weighted stock score (a pie of strong names scores high).
        pie_score = sum(slice_meta[s]["score"] * slice_meta[s]["value"] for s in slice_meta)
        pie_score = (pie_score / pie_value) if pie_value > 0 else 0.0
        slices.sort(key=lambda s: s.target_weight, reverse=True)
        target_pies.append(TargetPie(
            name=pie.get("name", "Pie"),
            current_value=round(pie_value, 2),
            current_weight=0.0, target_weight=0.0, drift=0.0,
            score=round(pie_score, 4), slices=slices,
        ))

    # 4) Across-pie capped target weights + current weights.
    grand_total = sum(p.current_value for p in target_pies) or 1.0
    pie_scores = {p.name: p.score for p in target_pies}
    pie_target = _cap_and_redistribute(pie_scores, max_pie_weight)
    for p in target_pies:
        p.current_weight = round(p.current_value / grand_total, 4)
        p.target_weight = round(pie_target.get(p.name, 0.0), 4)
        p.drift = round(p.target_weight - p.current_weight, 4)

    # 5) Add-only contribution: close the largest UNDER-weight gaps first.
    #    Work in absolute portfolio dollars: desired_$ = target_weight * (port + contribution).
    _allocate_contribution(target_pies, grand_total, contribution, max_stock_weight)

    target_pies.sort(key=lambda p: p.target_weight, reverse=True)
    return TargetAllocation(
        as_of=as_of, contribution=contribution,
        max_stock_weight=max_stock_weight, max_pie_weight=max_pie_weight,
        pies=target_pies,
    )


def _allocate_contribution(
    pies: List[TargetPie], port_total: float, contribution: float, max_stock_weight: float
) -> None:
    """Distribute the contribution to reduce the biggest under-weight gaps,
    never selling. Two levels: first across pies, then within each funded pie.
    """
    if contribution <= 0:
        return
    new_total = port_total + contribution

    # Pie level: desired add = max(0, target_$ - current_$); fund proportional to gap.
    pie_gap = {}
    for p in pies:
        desired = p.target_weight * new_total
        gap = max(0.0, desired - p.current_value)
        pie_gap[p.name] = gap
    total_gap = sum(pie_gap.values())
    for p in pies:
        if total_gap > 0:
            p.suggested_dollars = round(contribution * pie_gap[p.name] / total_gap, 2)
        else:
            p.suggested_dollars = round(contribution / len(pies), 2)

        # Within pie: same gap logic on the slices, using this pie's new dollars.
        pie_new_total = p.current_value + p.suggested_dollars
        slice_gap = {}
        for s in p.slices:
            desired_s = s.target_weight * pie_new_total
            slice_gap[s.symbol] = max(0.0, desired_s - s.current_value)
        sg_total = sum(slice_gap.values())
        for s in p.slices:
            if sg_total > 0:
                s.suggested_dollars = round(p.suggested_dollars * slice_gap[s.symbol] / sg_total, 2)
            else:
                s.suggested_dollars = 0.0
        p.slices.sort(key=lambda s: s.suggested_dollars, reverse=True)
