from __future__ import annotations

"""
M1 portfolio analyzer — ADVISORY ONLY.

Runs a panel of the existing daily strategies over each M1 holding, builds a
per-stock consensus signal + conviction score, and computes a signal-driven
"where to put your next contribution" tilt across the holdings.

This module places NO orders. M1 Finance has no per-order trading API; the
output is guidance the operator funds manually in M1.

Design notes
------------
- Reuses ``evaluate_strategy`` and ``get_ohlcv`` — the same engine + data path
  the scanner trusts. No new trading logic is introduced here.
- Strategy panel and tilt aggressiveness are chosen to suit long-term pie
  investing: trend-biased panel, "aggressive" tilt (only BUY-signal names are
  funded this cycle; HOLD/SELL get nothing). See ``DEFAULT_PANEL`` / ``TiltMode``.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.strategy.rules import evaluate_strategy

logger = logging.getLogger(__name__)

HOLDINGS_PATH = "m1_holdings.json"
PIES_PATH = "pies.json"

# Trend-biased panel — favors riding strength, which fits buy-and-hold pies.
# Each entry is (strategy_type, params). Params left default-ish; the rules
# carry sensible internal defaults.
DEFAULT_PANEL: List[tuple[str, Dict[str, Any]]] = [
    ("trend_follow", {}),
    ("momentum_breakout", {}),
    ("trend_pullback", {}),
    ("squeeze_breakout", {}),
]

# Tilt modes — how hard new money is concentrated toward strong signals.
#   aggressive: fund only net-BUY names this cycle; HOLD/SELL get 0.
#   moderate:   strong up to ~2x share, SELL ~0.
#   gentle:     strong up to ~1.5x, weak/SELL down to ~0.5x.
#   dip:        SIP-style — fund names that are oversold/pulled-back within an
#               uptrend (buy-the-dip), weighted by dip_score. Never sits in cash.
TiltMode = str  # "aggressive" | "moderate" | "gentle" | "dip"


@dataclass
class HoldingSignal:
    symbol: str
    name: str
    value: float                      # current $ value in M1
    price: Optional[float]            # latest close
    direction: str                    # BUY | HOLD | SELL (consensus)
    conviction: float                 # 0..100
    votes_buy: int
    votes_sell: int
    votes_hold: int
    per_strategy: Dict[str, str] = field(default_factory=dict)  # type -> direction
    tilt_weight: float = 0.0          # fraction of new contribution (0..1)
    suggested_dollars: float = 0.0    # tilt_weight * contribution
    # ── Dip / SIP-timing fields ──
    rsi: Optional[float] = None       # latest RSI(14)
    pct_from_high: Optional[float] = None  # % below trailing 6mo high (negative)
    above_sma200: Optional[bool] = None    # still in a long-term uptrend?
    dip_score: float = 0.0            # 0..100 — higher = better dip-buy
    # ── Reshuffle (add-only) ──
    laggard: bool = False             # weak name worth reviewing (never auto-sold)
    note: str = ""                    # short human-readable reason
    error: Optional[str] = None


@dataclass
class PortfolioAnalysis:
    as_of: Optional[str]
    contribution: float
    holdings: List[HoldingSignal]
    analyzed: int
    failed: int
    advisory_note: str = (
        "Advisory only. M1 has no trading API — fund these suggestions manually. "
        "Signals are not financial advice."
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def load_holdings(path: str = HOLDINGS_PATH) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Load the M1 holdings file. Returns (as_of, holdings list)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"M1 holdings file not found: {path!r}")
    with open(path) as f:
        raw = json.load(f)
    return raw.get("as_of"), list(raw.get("holdings", []))


def _consensus(per_strategy: Dict[str, str]) -> tuple[str, int, int, int]:
    """Majority vote → (direction, buys, sells, holds). Ties resolve to HOLD."""
    buys = sum(1 for d in per_strategy.values() if d == "BUY")
    sells = sum(1 for d in per_strategy.values() if d == "SELL")
    holds = sum(1 for d in per_strategy.values() if d == "HOLD")
    if buys > sells and buys >= holds:
        direction = "BUY"
    elif sells > buys and sells >= holds:
        direction = "SELL"
    else:
        direction = "HOLD"
    return direction, buys, sells, holds


def _conviction(direction: str, buys: int, sells: int, total: int) -> float:
    """0..100. Net agreement of the winning side scaled by panel size."""
    if total == 0:
        return 0.0
    if direction == "BUY":
        net = buys - sells
    elif direction == "SELL":
        net = sells - buys
    else:
        return 0.0
    return max(0.0, min(100.0, (net / total) * 100.0))


def _dip_metrics(prices: pd.Series) -> tuple[Optional[float], Optional[float], Optional[bool], float]:
    """Compute (rsi, pct_from_high, above_sma200, dip_score) for SIP dip-timing.

    dip_score (0..100) rewards buying a pullback WITHIN a healthy long-term
    uptrend, not catching a falling knife:
      - low RSI (oversold) raises it,
      - being further below the trailing 6mo high raises it (a real dip),
      - but it's gated on price > SMA200 (still structurally up). A name below
        its 200-day gets a damped score — that's a downtrend, not a dip.
    """
    from app.services.indicators.rsi import compute_rsi

    if len(prices) < 30:
        return None, None, None, 0.0

    # RSI(14)
    try:
        rsi_val = float(compute_rsi(prices, 14).values.iloc[-1])
    except Exception:
        rsi_val = None

    # % from trailing high (~6mo = 126 trading days)
    window = prices.tail(126)
    hi = float(window.max())
    last = float(prices.iloc[-1])
    pct_from_high = (last / hi - 1.0) * 100.0 if hi > 0 else None  # <= 0

    # SMA200 regime
    above_sma200: Optional[bool] = None
    if len(prices) >= 200:
        sma200 = float(prices.tail(200).mean())
        above_sma200 = last > sma200

    # Score: blend oversold + pullback depth, gated by uptrend.
    rsi_component = 0.0
    if rsi_val is not None:
        # RSI 30 -> 100, RSI 70 -> 0 (linear, clamped)
        rsi_component = max(0.0, min(100.0, (70.0 - rsi_val) / 40.0 * 100.0))
    depth_component = 0.0
    if pct_from_high is not None:
        # 0% off high -> 0, -15% or deeper -> 100
        depth_component = max(0.0, min(100.0, (-pct_from_high) / 15.0 * 100.0))

    raw = 0.55 * rsi_component + 0.45 * depth_component
    if above_sma200 is False:
        raw *= 0.35  # below 200-day: likely a downtrend, damp the "dip" appeal
    dip_score = round(max(0.0, min(100.0, raw)), 1)
    return rsi_val, pct_from_high, above_sma200, dip_score


def analyze_holding(
    holding: Dict[str, Any],
    panel: List[tuple[str, Dict[str, Any]]],
    period: str = "1y",
) -> HoldingSignal:
    """Run the strategy panel over one holding and build its consensus signal."""
    symbol = holding["symbol"].upper()
    name = holding.get("name", symbol)
    value = float(holding.get("value", 0.0) or 0.0)

    try:
        df = get_ohlcv(symbol, period=period, interval="1d")
        prices = df["Close"].dropna()
        if prices.empty:
            raise ValueError("no close data")
        price = float(prices.iloc[-1])
    except Exception as exc:  # data failure for this name — skip, don't crash the run
        logger.warning("[m1] %s data fetch failed: %s", symbol, exc)
        return HoldingSignal(
            symbol=symbol, name=name, value=value, price=None,
            direction="HOLD", conviction=0.0, votes_buy=0, votes_sell=0,
            votes_hold=0, error=str(exc),
        )

    per_strategy: Dict[str, str] = {}
    for stype, params in panel:
        try:
            sig = evaluate_strategy(stype, symbol, prices, params, ohlcv=df)
            per_strategy[stype] = sig.direction
        except Exception as exc:
            logger.debug("[m1] %s strategy %s failed: %s", symbol, stype, exc)
            per_strategy[stype] = "HOLD"

    direction, buys, sells, holds = _consensus(per_strategy)
    conviction = _conviction(direction, buys, sells, len(per_strategy))
    rsi_val, pct_from_high, above_sma200, dip_score = _dip_metrics(prices)

    # Add-only reshuffle: a laggard is a SELL-consensus name that's also broken
    # down (below its 200-day). We FLAG it for review — never auto-sell.
    laggard = (direction == "SELL") and (above_sma200 is False)

    note_bits = []
    if direction == "BUY":
        note_bits.append("trend BUY")
    if dip_score >= 60 and above_sma200 is not False:
        note_bits.append("dip-buy zone")
    if laggard:
        note_bits.append("laggard — review")

    return HoldingSignal(
        symbol=symbol, name=name, value=value, price=price,
        direction=direction, conviction=conviction,
        votes_buy=buys, votes_sell=sells, votes_hold=holds,
        per_strategy=per_strategy,
        rsi=round(rsi_val, 1) if rsi_val is not None else None,
        pct_from_high=round(pct_from_high, 1) if pct_from_high is not None else None,
        above_sma200=above_sma200, dip_score=dip_score,
        laggard=laggard, note=" · ".join(note_bits),
    )


def compute_tilt(
    signals: List[HoldingSignal],
    contribution: float,
    mode: TiltMode = "aggressive",
) -> None:
    """Assign tilt_weight (sums to 1.0 across fundable names) + suggested_dollars.

    Mutates the signals in place. ``mode`` controls aggressiveness:
      - aggressive: only BUY names get weight, proportional to conviction.
      - moderate:   BUY ~2x, HOLD ~1x, SELL ~0 (by current value share).
      - gentle:     BUY ~1.5x, HOLD ~1x, SELL ~0.5x (by current value share).
      - dip:        SIP buy-the-dip — weight by dip_score, but never fund a
                    laggard (broken-down SELL). Always deploys (never all-cash)
                    as long as any non-laggard name exists.
    """
    usable = [s for s in signals if s.error is None]

    raw: Dict[str, float] = {}
    for s in usable:
        share = max(s.value, 0.0)
        if mode == "aggressive":
            # Only fund BUY signals; weight by conviction (floor so a 1-vote
            # BUY still gets something). HOLD/SELL → 0.
            raw[s.symbol] = (s.conviction + 10.0) if s.direction == "BUY" else 0.0
        elif mode == "moderate":
            mult = {"BUY": 2.0, "HOLD": 1.0, "SELL": 0.0}[s.direction]
            raw[s.symbol] = share * mult
        elif mode == "dip":
            # Buy-the-dip SIP: weight by dip_score (oversold pullback in uptrend),
            # with a small base so every healthy name still accumulates. A BUY
            # signal adds a bonus. Laggards (broken-down SELL) get nothing.
            if s.laggard:
                raw[s.symbol] = 0.0
            else:
                base = 10.0
                buy_bonus = 25.0 if s.direction == "BUY" else 0.0
                raw[s.symbol] = base + s.dip_score + buy_bonus
        else:  # gentle
            mult = {"BUY": 1.5, "HOLD": 1.0, "SELL": 0.5}[s.direction]
            raw[s.symbol] = share * mult

    total = sum(raw.values())
    for s in signals:
        if s.error is not None or total <= 0:
            s.tilt_weight = 0.0
            s.suggested_dollars = 0.0
        else:
            s.tilt_weight = raw.get(s.symbol, 0.0) / total
            s.suggested_dollars = round(s.tilt_weight * contribution, 2)


def analyze_portfolio(
    contribution: float = 0.0,
    panel: Optional[List[tuple[str, Dict[str, Any]]]] = None,
    tilt_mode: TiltMode = "aggressive",
    period: str = "1y",
    path: str = HOLDINGS_PATH,
) -> PortfolioAnalysis:
    """Full pipeline: load holdings → per-stock signals → contribution tilt."""
    panel = panel or DEFAULT_PANEL
    as_of, holdings = load_holdings(path)

    signals = [analyze_holding(h, panel, period=period) for h in holdings]
    compute_tilt(signals, contribution, mode=tilt_mode)

    # Sort: fundable BUYs first (by suggested $), then by conviction.
    signals.sort(key=lambda s: (s.suggested_dollars, s.conviction), reverse=True)

    failed = sum(1 for s in signals if s.error is not None)
    return PortfolioAnalysis(
        as_of=as_of,
        contribution=contribution,
        holdings=signals,
        analyzed=len(signals) - failed,
        failed=failed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pie-level (two-level) analysis
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PieSlice:
    symbol: str
    value: float                  # slice $ within this pie
    direction: str
    conviction: float
    tilt_weight: float = 0.0      # within-pie share of THIS pie's dollars
    suggested_dollars: float = 0.0
    dip_score: float = 0.0
    rsi: Optional[float] = None
    pct_from_high: Optional[float] = None
    laggard: bool = False
    note: str = ""
    error: Optional[str] = None


@dataclass
class PieResult:
    name: str
    value: float                  # total $ of the pie
    conviction: float             # value-weighted BUY conviction of the pie (0..100)
    buy_slices: int
    sell_slices: int
    hold_slices: int
    pie_weight: float = 0.0       # share of the whole contribution this pie gets
    suggested_dollars: float = 0.0
    slices: List[PieSlice] = field(default_factory=list)


@dataclass
class PiePortfolioAnalysis:
    as_of: Optional[str]
    contribution: float
    pie_split_mode: str
    tilt_mode: str
    pies: List[PieResult]
    analyzed_symbols: int
    failed_symbols: int
    advisory_note: str = (
        "Advisory only. M1 has no trading API — fund these suggestions manually. "
        "Pies share tickers; each pie is funded independently. Not financial advice."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_pies(path: str = PIES_PATH) -> tuple[Optional[str], List[Dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pies file not found: {path!r}")
    with open(path) as f:
        raw = json.load(f)
    return raw.get("as_of"), list(raw.get("pies", []))


def _pie_conviction(slices: List[PieSlice]) -> float:
    """Value-weighted NET-BUY conviction for a pie (0..100).

    BUY slices contribute +conviction, SELL slices contribute -conviction,
    HOLD contributes 0, each weighted by slice value. Clamped at 0 so a pie
    dominated by SELLs gets no contribution (aligns with the aggressive tilt).
    """
    num = 0.0
    den = 0.0
    for s in slices:
        if s.error is not None:
            continue
        w = max(s.value, 0.0)
        den += w
        if s.direction == "BUY":
            num += w * s.conviction
        elif s.direction == "SELL":
            num -= w * s.conviction
    if den <= 0:
        return 0.0
    return max(0.0, min(100.0, num / den))


def analyze_pies(
    contribution: float = 0.0,
    panel: Optional[List[tuple[str, Dict[str, Any]]]] = None,
    tilt_mode: TiltMode = "aggressive",
    pie_split_mode: str = "conviction",   # "conviction" | "equal" | "value"
    period: str = "1y",
    path: str = PIES_PATH,
) -> PiePortfolioAnalysis:
    """Two-level funding plan: split the contribution across pies, then tilt
    within each pie. A ticker appearing in multiple pies is signalled ONCE
    (cached) but funded independently per pie.
    """
    panel = panel or DEFAULT_PANEL
    as_of, pies_raw = load_pies(path)

    # 1) Signal each UNIQUE symbol once.
    uniq: Dict[str, HoldingSignal] = {}
    for pie in pies_raw:
        for sl in pie.get("slices", []):
            sym = sl["symbol"].upper()
            if sym not in uniq:
                uniq[sym] = analyze_holding(
                    {"symbol": sym, "name": sym, "value": 0.0}, panel, period=period
                )
    failed = sum(1 for s in uniq.values() if s.error is not None)

    # 2) Build per-pie slice signals + pie conviction.
    pie_results: List[PieResult] = []
    for pie in pies_raw:
        slices: List[PieSlice] = []
        for sl in pie.get("slices", []):
            sym = sl["symbol"].upper()
            sig = uniq[sym]
            slices.append(PieSlice(
                symbol=sym, value=float(sl.get("value", 0.0) or 0.0),
                direction=sig.direction, conviction=sig.conviction,
                dip_score=sig.dip_score, rsi=sig.rsi,
                pct_from_high=sig.pct_from_high, laggard=sig.laggard,
                note=sig.note, error=sig.error,
            ))
        conv = _pie_conviction(slices)
        pie_results.append(PieResult(
            name=pie.get("name", "Pie"),
            value=round(sum(s.value for s in slices), 2),
            conviction=round(conv, 1),
            buy_slices=sum(1 for s in slices if s.direction == "BUY"),
            sell_slices=sum(1 for s in slices if s.direction == "SELL"),
            hold_slices=sum(1 for s in slices if s.direction == "HOLD"),
            slices=slices,
        ))

    # 3) Top-level split across pies.
    if pie_split_mode == "equal":
        raw_pie = {p.name: 1.0 for p in pie_results}
    elif pie_split_mode == "value":
        raw_pie = {p.name: max(p.value, 0.0) for p in pie_results}
    else:  # conviction
        raw_pie = {p.name: p.conviction for p in pie_results}
    total_pie = sum(raw_pie.values())
    if total_pie <= 0:
        # No conviction anywhere → fall back to equal so cash isn't stranded.
        n = len(pie_results) or 1
        for p in pie_results:
            p.pie_weight = 1.0 / n
    else:
        for p in pie_results:
            p.pie_weight = raw_pie[p.name] / total_pie

    # 4) Within-pie tilt, scaled by each pie's dollars.
    for p in pie_results:
        p.suggested_dollars = round(p.pie_weight * contribution, 2)
        # Reuse the holding-level tilt math on this pie's slices.
        as_holdings = [
            HoldingSignal(
                symbol=s.symbol, name=s.symbol, value=s.value, price=None,
                direction=s.direction, conviction=s.conviction,
                votes_buy=0, votes_sell=0, votes_hold=0,
                dip_score=s.dip_score, laggard=s.laggard, error=s.error,
            ) for s in p.slices
        ]
        compute_tilt(as_holdings, p.suggested_dollars, mode=tilt_mode)
        by = {h.symbol: h for h in as_holdings}
        for s in p.slices:
            h = by[s.symbol]
            s.tilt_weight = h.tilt_weight
            s.suggested_dollars = h.suggested_dollars
        # Sort slices: funded first, then conviction.
        p.slices.sort(key=lambda s: (s.suggested_dollars, s.conviction), reverse=True)

    # Sort pies by suggested dollars.
    pie_results.sort(key=lambda p: (p.suggested_dollars, p.conviction), reverse=True)

    return PiePortfolioAnalysis(
        as_of=as_of, contribution=contribution,
        pie_split_mode=pie_split_mode, tilt_mode=tilt_mode,
        pies=pie_results,
        analyzed_symbols=len(uniq) - failed, failed_symbols=failed,
    )
