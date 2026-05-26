from __future__ import annotations

"""
Trade Analyzer — snapshot entry-bar indicators for every completed trade,
then compare winning vs losing distributions to surface actionable patterns.

Supports EMA_Mean_Reversion primarily but is generic enough for any strategy
that uses close/RSI/ATR/SMA200/EMA20/volume indicators.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bb(s: pd.Series, period: int = 20, std: float = 2.0):
    mid = s.rolling(period).mean()
    sigma = s.rolling(period).std()
    return mid - std * sigma, mid, mid + std * sigma


# ── Per-trade snapshot ────────────────────────────────────────────────────────

@dataclass
class TradeSnapshot:
    date: str
    outcome: str          # "win" | "loss"
    pnl: float
    pnl_pct: float        # pnl as % of entry value

    # Price context
    entry_price: float
    exit_price: float
    hold_bars: int

    # Indicator values AT ENTRY BAR
    rsi: float
    atr_pct: float        # ATR as % of price — measures volatility regime
    ema_dist_pct: float   # % distance from EMA20 (negative = below EMA)
    sma200_dist_pct: float  # % distance above SMA200 (trend strength)
    bb_pct: float         # where in BB bands: 0=lower, 0.5=mid, 1=upper
    volume_ratio: float   # today's volume / 20-day avg volume
    body_pct: float       # candle body as % of ATR (0=doji, 1=strong candle)
    upper_wick_pct: float # upper wick as % of range
    lower_wick_pct: float # lower wick as % of range (big = rejection)

    # Time features
    month: int
    day_of_week: int      # 0=Mon, 4=Fri
    quarter: int

    # Sequence context
    prior_trade_won: Optional[bool]   # did the trade BEFORE this one win?
    bars_since_last_trade: Optional[int]

    # Strategy-specific indicators (None when not applicable)
    ema_spread_pct: Optional[float] = None   # MA_Crossover: fast/slow EMA spread % at crossover
    range_atr_ratio: Optional[float] = None  # Breakout: consolidation range / ATR (tightness)
    bb_depth_pct: Optional[float] = None     # BB_MeanRev: how far below lower band (% of band width)
    fib_level: Optional[float] = None        # Fib_Pullback: which Fib level was hit (0.382/0.5/0.618)


# ── Indicator snapshot ────────────────────────────────────────────────────────

def _snapshot_indicators(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Compute all indicators at bar `idx` using only df.iloc[:idx]."""
    window = df.iloc[: idx + 1]
    close = window["Close"]

    ema20 = float(_ema(close, 20).iloc[-1])
    sma200 = float(_sma(close, 200).iloc[-1]) if len(close) >= 200 else float(close.mean())
    rsi14 = float(_rsi(close, 14).iloc[-1])

    atr_series = _atr(window, 14)
    atr_val = float(atr_series.iloc[-1])

    bb_lower, bb_mid, bb_upper = _bb(close, 20, 2.0)
    bb_l = float(bb_lower.iloc[-1])
    bb_u = float(bb_upper.iloc[-1])
    c_now = float(close.iloc[-1])
    bb_width = bb_u - bb_l
    bb_pct = (c_now - bb_l) / bb_width if bb_width > 0 else 0.5

    o = float(window["Open"].iloc[-1])
    h = float(window["High"].iloc[-1])
    l = float(window["Low"].iloc[-1])
    bar_range = h - l

    body = abs(c_now - o)
    upper_wick = h - max(c_now, o)
    lower_wick = min(c_now, o) - l

    vol_ratio = 1.0
    if "Volume" in window.columns:
        avg_vol = float(window["Volume"].iloc[-21:-1].mean()) if len(window) > 21 else 1.0
        cur_vol = float(window["Volume"].iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

    return {
        "rsi": rsi14,
        "atr_pct": atr_val / c_now * 100 if c_now > 0 else 0.0,
        "ema_dist_pct": (c_now - ema20) / ema20 * 100 if ema20 > 0 else 0.0,
        "sma200_dist_pct": (c_now - sma200) / sma200 * 100 if sma200 > 0 else 0.0,
        "bb_pct": bb_pct,
        "volume_ratio": vol_ratio,
        "body_pct": body / atr_val if atr_val > 0 else 0.0,
        "upper_wick_pct": upper_wick / bar_range if bar_range > 0 else 0.0,
        "lower_wick_pct": lower_wick / bar_range if bar_range > 0 else 0.0,
    }


# ── Strategy-specific indicators ─────────────────────────────────────────────

def _strategy_indicators(df: pd.DataFrame, idx: int, strategy_name: str, buy_trade: dict) -> Dict[str, Optional[float]]:
    """
    Compute indicators that only make sense for a specific strategy.
    buy_trade contains the recorded stop/target/reason from the backtest.
    """
    result: Dict[str, Optional[float]] = {
        "ema_spread_pct": None,
        "range_atr_ratio": None,
        "bb_depth_pct": None,
        "fib_level": None,
    }
    window = df.iloc[: idx + 1]
    close = window["Close"]

    if strategy_name == "MA_Crossover_RSI":
        # Fast/slow EMA spread % — how separated were the EMAs at crossover
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        ef = float(ema20.iloc[-1])
        es = float(ema50.iloc[-1])
        result["ema_spread_pct"] = abs(ef - es) / es * 100 if es > 0 else 0.0

    elif strategy_name == "Breakout_Consolidation":
        # Range tightness: consolidation range / ATR — smaller = tighter base = better breakout
        atr_val = float(_atr(window, 14).iloc[-1])
        h10 = float(df["High"].iloc[max(0, idx - 12): idx].max())
        l10 = float(df["Low"].iloc[max(0, idx - 12): idx].min())
        range_size = h10 - l10
        result["range_atr_ratio"] = range_size / atr_val if atr_val > 0 else None

    elif strategy_name == "BB_Mean_Reversion":
        # How far below the lower band the price dipped (% of band width)
        # Look back up to 4 bars to find the oversold touch
        bb_lower, _, bb_upper = _bb(close, 20, 2.0)
        depth = 0.0
        for k in range(-4, 0):
            c_bar = float(close.iloc[k]) if abs(k) <= len(close) else float(close.iloc[-1])
            bl    = float(bb_lower.iloc[k]) if abs(k) <= len(bb_lower) else float(bb_lower.iloc[-1])
            bu    = float(bb_upper.iloc[k]) if abs(k) <= len(bb_upper) else float(bb_upper.iloc[-1])
            bw    = bu - bl
            if c_bar < bl and bw > 0:
                depth = max(depth, (bl - c_bar) / bw)
        result["bb_depth_pct"] = depth

    elif strategy_name == "Fib_Pullback_Support":
        # Which Fib level was hit — extract from the trade reason string recorded at entry
        reason = buy_trade.get("reason", "")
        for lvl in [0.382, 0.50, 0.618]:
            if f"{lvl:.1%}" in reason or f"{lvl*100:.1f}%" in reason:
                result["fib_level"] = lvl
                break

    return result


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_trades(
    trades: List[dict],
    df: pd.DataFrame,
    strategy_name: str = "",
) -> List[TradeSnapshot]:
    """
    Match buy/sell pairs from the backtest trade log and snapshot entry-bar
    indicators for each completed trade.

    trades: the raw trade list from PerplexityBacktestResult.trades
    df:     the full OHLCV dataframe used in the backtest
    """
    date_to_iloc: Dict[str, int] = {str(d)[:10]: i for i, d in enumerate(df.index)}

    # Pair BUY → SELL in order
    pairs = []
    pending_buy = None
    for t in trades:
        if t["side"] == "BUY":
            pending_buy = t
        elif "SELL" in t["side"] and pending_buy is not None:
            pairs.append((pending_buy, t))
            pending_buy = None

    snapshots: List[TradeSnapshot] = []

    for i, (buy, sell) in enumerate(pairs):
        entry_date = buy["date"]
        exit_date = sell["date"]
        entry_price = buy["price"]
        exit_price = sell["price"]
        pnl = sell.get("pnl", 0.0) or 0.0
        entry_value = buy.get("value", entry_price)
        pnl_pct = pnl / entry_value * 100 if entry_value > 0 else 0.0
        outcome = "win" if pnl > 0 else "loss"

        entry_iloc = date_to_iloc.get(entry_date)
        exit_iloc = date_to_iloc.get(exit_date)
        if entry_iloc is None or entry_iloc < 200:
            continue

        hold_bars = (exit_iloc - entry_iloc) if exit_iloc and exit_iloc > entry_iloc else 0

        inds = _snapshot_indicators(df, entry_iloc)
        strat_inds = _strategy_indicators(df, entry_iloc, strategy_name, buy)

        # Time features
        try:
            ts = pd.Timestamp(entry_date)
            month = ts.month
            dow = ts.dayofweek
            quarter = ts.quarter
        except Exception:
            month, dow, quarter = 0, 0, 0

        # Sequence context
        prior_won: Optional[bool] = None
        bars_since: Optional[int] = None
        if i > 0:
            prev_buy, prev_sell = pairs[i - 1]
            prior_pnl = prev_sell.get("pnl", 0.0) or 0.0
            prior_won = prior_pnl > 0
            prev_exit_iloc = date_to_iloc.get(prev_sell["date"])
            if prev_exit_iloc is not None and entry_iloc is not None:
                bars_since = entry_iloc - prev_exit_iloc

        snapshots.append(TradeSnapshot(
            date=entry_date,
            outcome=outcome,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            entry_price=entry_price,
            exit_price=exit_price,
            hold_bars=hold_bars,
            rsi=round(inds["rsi"], 2),
            atr_pct=round(inds["atr_pct"], 3),
            ema_dist_pct=round(inds["ema_dist_pct"], 3),
            sma200_dist_pct=round(inds["sma200_dist_pct"], 3),
            bb_pct=round(inds["bb_pct"], 3),
            volume_ratio=round(inds["volume_ratio"], 3),
            body_pct=round(inds["body_pct"], 3),
            upper_wick_pct=round(inds["upper_wick_pct"], 3),
            lower_wick_pct=round(inds["lower_wick_pct"], 3),
            month=month,
            day_of_week=dow,
            quarter=quarter,
            prior_trade_won=prior_won,
            bars_since_last_trade=bars_since,
            ema_spread_pct=round(strat_inds["ema_spread_pct"], 3) if strat_inds["ema_spread_pct"] is not None else None,
            range_atr_ratio=round(strat_inds["range_atr_ratio"], 3) if strat_inds["range_atr_ratio"] is not None else None,
            bb_depth_pct=round(strat_inds["bb_depth_pct"], 3) if strat_inds["bb_depth_pct"] is not None else None,
            fib_level=strat_inds["fib_level"],
        ))

    return snapshots


# ── Pattern finder ────────────────────────────────────────────────────────────

@dataclass
class PatternInsight:
    indicator: str
    description: str
    win_mean: float
    loss_mean: float
    separation: float     # abs(win_mean - loss_mean) / pooled_std — effect size
    recommendation: str   # human-readable filter suggestion
    direction: str        # "higher_is_better" | "lower_is_better" | "range"
    suggested_min: Optional[float] = None
    suggested_max: Optional[float] = None


def find_patterns(snapshots: List[TradeSnapshot]) -> List[PatternInsight]:
    """
    Compare winning vs losing trade distributions on each numeric indicator.
    Returns insights sorted by separation strength (biggest discriminators first).
    """
    wins  = [s for s in snapshots if s.outcome == "win"]
    losses = [s for s in snapshots if s.outcome == "loss"]

    if len(wins) < 3 or len(losses) < 3:
        return []

    INDICATORS = {
        "rsi":              ("RSI at entry", "RSI"),
        "atr_pct":          ("ATR % of price (volatility)", "ATR%"),
        "ema_dist_pct":     ("EMA20 distance % (negative=below EMA)", "EMA dist%"),
        "sma200_dist_pct":  ("SMA200 distance % (trend strength)", "SMA200 dist%"),
        "bb_pct":           ("BB position (0=lower band, 1=upper band)", "BB%"),
        "volume_ratio":     ("Volume vs 20d average", "Vol ratio"),
        "body_pct":         ("Candle body size (× ATR)", "Body%"),
        "lower_wick_pct":   ("Lower wick % of range (rejection)", "Lower wick%"),
        "upper_wick_pct":   ("Upper wick % of range", "Upper wick%"),
        "hold_bars":        ("Days held", "Hold bars"),
        # Strategy-specific
        "ema_spread_pct":   ("EMA20/50 spread % at crossover (MA_Crossover)", "EMA spread%"),
        "range_atr_ratio":  ("Consolidation range / ATR (Breakout)", "Range/ATR"),
        "bb_depth_pct":     ("Depth below BB lower band (BB_MeanRev)", "BB depth%"),
    }

    def _vals(group, key):
        return [v for s in group if (v := getattr(s, key, None)) is not None]

    def _pooled_std(a, b):
        if len(a) < 2 and len(b) < 2:
            return 1.0
        var_a = statistics.variance(a) if len(a) >= 2 else 0.0
        var_b = statistics.variance(b) if len(b) >= 2 else 0.0
        return ((var_a + var_b) / 2) ** 0.5 or 1.0

    insights = []
    for key, (description, label) in INDICATORS.items():
        w_vals = _vals(wins, key)
        l_vals = _vals(losses, key)

        if not w_vals or not l_vals:
            continue

        w_mean = statistics.mean(w_vals)
        l_mean = statistics.mean(l_vals)
        pooled = _pooled_std(w_vals, l_vals)
        separation = abs(w_mean - l_mean) / pooled

        if separation < 0.15:
            continue  # not meaningfully different

        # Derive direction and suggested filter
        direction = "higher_is_better" if w_mean > l_mean else "lower_is_better"

        # Suggested filter = the DECISION BOUNDARY between the two distributions,
        # not a tail of the winners. A bare win-percentile (the old approach) could
        # land *past* the loser cluster — e.g. ema_dist_pct winners avg -1.06 but
        # their 20th pct reaches -3.1, below the -2.8 loss avg, so "≥ -3.1" admits
        # every loser and filters nothing. Instead anchor on the midpoint of the
        # means, then clamp so the cut never sits on the wrong side of the losers
        # (a useful "higher→win" filter must exclude the typical loser).
        midpoint = (w_mean + l_mean) / 2.0

        if direction == "higher_is_better":
            # cut between the means, but no looser than the loss mean
            thresh = max(midpoint, l_mean)
            suggestion = f"Require {label} ≥ {thresh:.2f} (wins avg {w_mean:.2f} vs losses {l_mean:.2f})"
            smin, smax = thresh, None
        else:
            # cut between the means, but no looser than the loss mean
            thresh = min(midpoint, l_mean)
            suggestion = f"Require {label} ≤ {thresh:.2f} (wins avg {w_mean:.2f} vs losses {l_mean:.2f})"
            smin, smax = None, thresh

        insights.append(PatternInsight(
            indicator=key,
            description=description,
            win_mean=round(w_mean, 3),
            loss_mean=round(l_mean, 3),
            separation=round(separation, 3),
            recommendation=suggestion,
            direction=direction,
            suggested_min=round(smin, 3) if smin is not None else None,
            suggested_max=round(smax, 3) if smax is not None else None,
        ))

    return sorted(insights, key=lambda x: x.separation, reverse=True)


# ── Season / time breakdown ───────────────────────────────────────────────────

def time_breakdown(snapshots: List[TradeSnapshot]) -> Dict[str, Any]:
    """Win rate by month, quarter, and day of week."""
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri"]

    def _wr_by(key, labels, offset=0):
        out = {}
        for i, name in enumerate(labels):
            idx = i + offset
            group = [s for s in snapshots if getattr(s, key) == idx]
            if not group:
                continue
            wins = sum(1 for s in group if s.outcome == "win")
            pnl_vals = [s.pnl_pct for s in group]
            out[name] = {
                "trades": len(group),
                "win_rate": round(wins / len(group) * 100, 1),
                "avg_pnl_pct": round(statistics.mean(pnl_vals), 2) if pnl_vals else 0.0,
            }
        return out

    return {
        "by_month":    _wr_by("month",       MONTH_NAMES, offset=1),
        "by_quarter":  _wr_by("quarter",     ["Q1","Q2","Q3","Q4"], offset=1),
        "by_day":      _wr_by("day_of_week", DOW_NAMES, offset=0),
    }
