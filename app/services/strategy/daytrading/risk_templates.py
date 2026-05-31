"""
risk_templates.py — Central risk, sizing, and regime constants for day trading.

All pure functions / constants — no side effects, no imports from strategy files.
Consumed by:
  - strategy generate_signals() to build ExitPlan
  - ExitManager / PositionManager to look up scale levels and trail modes
  - execution layer for position sizing and daily-loss gates

Symbol bucket classification
-----------------------------
US_ETF        : SPY, QQQ, IWM, DIA, XLK, XLF, …
US_LARGE_CAP  : AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, …
US_MID_SMALL  : everything else on US markets
NSE_LARGE_CAP : NIFTY-50 constituents — RELIANCE, BHARTIARTL, INFY, …
NSE_MID_CAP   : everything else on NSE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Symbol bucket type ────────────────────────────────────────────────────────

SymbolBucket = Literal[
    "US_ETF",
    "US_LARGE_CAP",
    "US_MID_SMALL",
    "NSE_LARGE_CAP",
    "NSE_MID_CAP",
]

TrailType = Literal["ema9_5m", "ema9_15m", "supertrend_5m", "prior_bar_low_5m", "atr_fixed", "none"]

# ── Known symbol sets (extend as needed) ─────────────────────────────────────

_US_ETFS: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLRE", "XLU", "XLB", "XLP",
    "SQQQ", "TQQQ", "SPXS", "SPXL", "UVXY", "VXX",
})

_US_LARGE_CAPS: frozenset[str] = frozenset({
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "BRK.B", "JPM", "V", "MA", "UNH", "JNJ", "XOM", "CVX", "PG",
    "HD", "MRK", "ABBV", "LLY", "AVGO", "COST", "PEP", "KO",
    "AMD", "INTC", "QCOM", "MU", "NFLX", "CRM", "ORCL", "ADBE",
    "BA", "CAT", "GE", "MMM", "HON", "RTX",
})

_NSE_LARGE_CAPS: frozenset[str] = frozenset({
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HCLTECH", "WIPRO",
    "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE", "TITAN", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "ADANIENT",
    "ADANIPORTS", "JSWSTEEL", "TATASTEEL", "SUNPHARMA", "DRREDDY",
    "CIPLA", "DIVISLAB", "TECHM", "HEROMOTOCO", "EICHERMOT",
    "BAJAJFINSV", "BAJAJ-AUTO", "TATAMOTORS", "M&M", "HINDALCO",
})


def get_symbol_bucket(symbol: str, market: str = "US") -> SymbolBucket:
    """Classify a symbol into its risk bucket.

    Parameters
    ----------
    symbol : ticker without suffix (e.g. "RELIANCE", not "RELIANCE.NS")
    market : "US" or "NSE"
    """
    sym = symbol.upper().split(".")[0]  # strip .NS / .BO if present
    if market == "NSE" or sym in _NSE_LARGE_CAPS:
        if sym in _NSE_LARGE_CAPS:
            return "NSE_LARGE_CAP"
        return "NSE_MID_CAP"
    if sym in _US_ETFS:
        return "US_ETF"
    if sym in _US_LARGE_CAPS:
        return "US_LARGE_CAP"
    return "US_MID_SMALL"


# ── Risk per trade (fraction of account equity) ───────────────────────────────

RISK_PER_TRADE_PCT: dict[SymbolBucket, float] = {
    "US_ETF":        0.0040,   # 0.40 %
    "US_LARGE_CAP":  0.0050,   # 0.50 %
    "US_MID_SMALL":  0.0060,   # 0.60 %
    "NSE_LARGE_CAP": 0.0050,   # 0.50 %
    "NSE_MID_CAP":   0.0060,   # 0.60 %
}


def position_size(
    account_value: float,
    bucket: SymbolBucket,
    entry: float,
    stop: float,
) -> int:
    """Return integer share count that risks exactly RISK_PER_TRADE_PCT of account."""
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or account_value <= 0:
        return 0
    risk_dollars = account_value * RISK_PER_TRADE_PCT[bucket]
    return max(1, int(risk_dollars / risk_per_share))


# ── Daily loss limits ─────────────────────────────────────────────────────────

DAILY_CAPS = {
    "hard_stop_pct":          -0.020,   # -2.0 %  → HALT all trading
    "soft_limit_pct":         -0.012,   # -1.2 %  → half size
    "hard_stop_r":            -4.0,     # -4 R    → HALT (R-based equivalent)
    "soft_limit_r":           -2.5,     # -2.5 R  → half size
    "consecutive_loss_pause":  3,       # 3 consecutive stops → 45-min pause
    "max_trades_per_day":      8,       # raised from 6
}


def session_risk_adjustment(
    current_pnl_r: float,
    consec_losses: int,
    current_pnl_pct: float = 0.0,
) -> dict:
    """Return the current session risk state.

    Returns
    -------
    dict with keys:
      status      : "NORMAL" | "REDUCE_SIZE" | "PAUSED" | "HALT"
      size_mult   : 0.0 – 1.0 multiplier to apply to position size
      reason      : human-readable explanation
      pause_minutes : int (only meaningful when status == "PAUSED")
    """
    # Hard stops (either R or %)
    if (current_pnl_r <= DAILY_CAPS["hard_stop_r"] or
            current_pnl_pct <= DAILY_CAPS["hard_stop_pct"]):
        return {
            "status": "HALT",
            "size_mult": 0.0,
            "reason": (
                f"Daily loss limit reached: {current_pnl_r:.1f}R / "
                f"{current_pnl_pct*100:.2f}%"
            ),
            "pause_minutes": 0,
        }

    # Consecutive loss pause
    if consec_losses >= DAILY_CAPS["consecutive_loss_pause"]:
        return {
            "status": "PAUSED",
            "size_mult": 0.0,
            "reason": f"{consec_losses} consecutive losses → 45-min mandatory pause",
            "pause_minutes": 45,
        }

    # Soft limits (either R or %)
    if (current_pnl_r <= DAILY_CAPS["soft_limit_r"] or
            current_pnl_pct <= DAILY_CAPS["soft_limit_pct"]):
        return {
            "status": "REDUCE_SIZE",
            "size_mult": 0.50,
            "reason": (
                f"Soft limit: {current_pnl_r:.1f}R / "
                f"{current_pnl_pct*100:.2f}% → half size"
            ),
            "pause_minutes": 0,
        }

    return {
        "status": "NORMAL",
        "size_mult": 1.0,
        "reason": "Within daily limits",
        "pause_minutes": 0,
    }


# ── Correlation / concurrency checks ─────────────────────────────────────────

MAX_CONCURRENT: dict[str, int] = {
    "per_symbol":  1,
    "US":          4,
    "NSE":         3,
    "US_sector":   2,
}


def can_enter(
    symbol: str,
    market: str,
    sector: str,
    open_positions: list[dict],
) -> dict:
    """Gate check before every new entry signal.

    Parameters
    ----------
    symbol         : ticker
    market         : "US" or "NSE"
    sector         : GICS sector string (empty string if unknown)
    open_positions : list of dicts, each with keys "symbol", "market", "sector"

    Returns
    -------
    dict with keys:
      allowed : bool
      reason  : str
    """
    # Rule 1: one position per symbol at a time
    if any(p["symbol"] == symbol for p in open_positions):
        return {
            "allowed": False,
            "reason": f"Symbol {symbol} already has an open position",
        }

    # Rule 2: market-level concurrency cap
    market_count = sum(1 for p in open_positions if p.get("market") == market)
    market_cap = MAX_CONCURRENT.get(market, 999)
    if market_count >= market_cap:
        return {
            "allowed": False,
            "reason": f"Max concurrent {market} positions ({market_cap}) reached",
        }

    # Rule 3: US sector cap (skip if sector unknown)
    if market == "US" and sector:
        sector_count = sum(
            1 for p in open_positions
            if p.get("market") == "US" and p.get("sector") == sector
        )
        if sector_count >= MAX_CONCURRENT["US_sector"]:
            return {
                "allowed": False,
                "reason": f"Sector '{sector}' already at cap ({MAX_CONCURRENT['US_sector']})",
            }

    return {"allowed": True, "reason": ""}


# ── CHOPPY / HIGH_VOL regime overrides ───────────────────────────────────────

CHOPPY_CONFIG = {
    "allowed_strategies": {"EMAMomentum", "VWAPMeanReversion"},
    "size_multiplier":    0.50,
    "min_rr_override":    2.0,
    "max_hold_bars_factor": 0.60,
}

HIGH_VOL_CONFIG = {
    "allowed_strategies": {"ORBBreakout", "SupertrendTrend"},
    "size_multiplier":    0.50,
    "min_rr_override":    2.0,
}


def regime_size_multiplier(
    regime: str,
    strategy_name: str,
) -> float:
    """Return the size multiplier implied by the regime for this strategy.

    Returns 0.0 if the strategy should not run in this regime.
    """
    if regime == "CHOPPY":
        if strategy_name not in CHOPPY_CONFIG["allowed_strategies"]:
            return 0.0
        return CHOPPY_CONFIG["size_multiplier"]
    if regime in ("HIGH_VOL", "NEWS_RISK"):
        if regime == "NEWS_RISK":
            return 0.0
        if strategy_name not in HIGH_VOL_CONFIG["allowed_strategies"]:
            return 0.0
        return HIGH_VOL_CONFIG["size_multiplier"]
    return 1.0


# ── ExitPlan dataclass ────────────────────────────────────────────────────────

@dataclass
class ScaleLevel:
    """One scale-out tier."""
    trigger_r: float        # take profit when position reaches this R multiple
    pct_to_close: float     # fraction of remaining position to close (0.0–1.0)
    trigger_price: float = 0.0   # absolute price (filled in by strategy, optional aid)


@dataclass
class ExitPlan:
    """
    Complete exit specification attached to every DayTradeSignal.

    The PositionManager / ExitManager consume this instead of hard-coded
    per-strategy magic numbers.  All fields carry explicit defaults so
    existing strategies that don't populate every field continue to work.

    Stop placement
    --------------
    initial_stop_price   : hard stop level (already computed by strategy)
    stop_type            : how the initial stop was anchored:
                           "atr"          – entry ± N×ATR
                           "bar_low_atr"  – bar_low − N×ATR (EMA bounce)
                           "st_line_atr"  – Supertrend line − N×ATR buffer
                           "nr_bar_low"   – NR7 bar low − N×ATR
                           "pct_gap"      – % beyond gap extreme

    Scale-out tiers
    ---------------
    scale_levels         : list of ScaleLevel in ascending trigger_r order.
                           ExitManager works through them left-to-right.

    Trailing stop
    -------------
    trail_type           : "ema9_5m" | "ema9_15m" | "supertrend_5m"
                           | "prior_bar_low_5m" | "atr_fixed" | "none"
    trail_trigger_r      : activate trailing once position reaches this R.
                           0.0 = activate immediately after first scale-out.
    trail_atr_mult       : only used when trail_type == "atr_fixed"

    Time exits
    ----------
    max_hold_bars        : position closed after this many 5m bars regardless
    hard_exit_time_et    : "HH:MM" ET — absolute wall-clock force-flat (US)
    hard_exit_time_ist   : "HH:MM" IST — absolute wall-clock force-flat (NSE)

    Break-even
    ----------
    breakeven_r          : move stop to entry when position reaches this R.
                           Convention: fire BEFORE the first scale-out (i.e.
                           set this lower than scale_levels[0].trigger_r).
    """
    # Stop
    initial_stop_price: float = 0.0
    stop_type: str = "atr"

    # Scale-outs (empty → single-target, full-exit at first_target)
    scale_levels: list[ScaleLevel] = field(default_factory=list)

    # Trailing
    trail_type: TrailType = "none"
    trail_trigger_r: float = 0.0
    trail_atr_mult: float = 1.5     # used only for "atr_fixed"

    # Time
    max_hold_bars: int = 48
    hard_exit_time_et: str = "15:45"
    hard_exit_time_ist: str = "14:45"

    # Break-even
    breakeven_r: float = 1.0       # move stop to entry at this R

    # Metadata
    bucket: str = ""               # SymbolBucket string for audit / UI
    strategy: str = ""

    def to_dict(self) -> dict:
        return {
            "stop_type":           self.stop_type,
            "initial_stop_price":  round(self.initial_stop_price, 4),
            "scale_levels": [
                {
                    "trigger_r":     s.trigger_r,
                    "pct_to_close":  s.pct_to_close,
                    "trigger_price": round(s.trigger_price, 4),
                }
                for s in self.scale_levels
            ],
            "trail_type":          self.trail_type,
            "trail_trigger_r":     self.trail_trigger_r,
            "trail_atr_mult":      self.trail_atr_mult,
            "max_hold_bars":       self.max_hold_bars,
            "hard_exit_time_et":   self.hard_exit_time_et,
            "hard_exit_time_ist":  self.hard_exit_time_ist,
            "breakeven_r":         self.breakeven_r,
            "bucket":              self.bucket,
        }


# ── Per-strategy, per-bucket ExitPlan factories ───────────────────────────────
# Each function returns a fully-populated ExitPlan.
# Strategies call the appropriate factory at signal generation time.
# The "bucket" argument comes from get_symbol_bucket(symbol, market).

def orb_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    orb_range: float,
) -> ExitPlan:
    """ORB Breakout — 3-tranche scale with EMA9 runner."""
    risk = abs(entry - stop)
    if risk <= 0:
        return ExitPlan(strategy="ORBBreakout", bucket=bucket)

    if bucket in ("US_ETF", "US_LARGE_CAP"):
        s1_price = entry + orb_range * 1.2
        s2_price = entry + orb_range * 2.0
        scales = [
            ScaleLevel(trigger_r=(s1_price - entry) / risk, pct_to_close=0.40, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=(s2_price - entry) / risk, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="ema9_5m",
            trail_trigger_r=(s1_price - entry) / risk,
            max_hold_bars=36,
            hard_exit_time_et="13:30",
            hard_exit_time_ist="15:45",  # ORB not primary for NSE; US time controls
            breakeven_r=(s1_price - entry) / risk,
            bucket=bucket,
            strategy="ORBBreakout",
        )

    if bucket == "US_MID_SMALL":
        s1_price = entry + orb_range * 1.5
        s2_price = entry + orb_range * 2.5
        scales = [
            ScaleLevel(trigger_r=(s1_price - entry) / risk, pct_to_close=0.40, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=(s2_price - entry) / risk, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="prior_bar_low_5m",
            trail_trigger_r=(s1_price - entry) / risk,
            max_hold_bars=36,
            hard_exit_time_et="13:00",
            hard_exit_time_ist="15:45",
            breakeven_r=(s1_price - entry) / risk,
            bucket=bucket,
            strategy="ORBBreakout",
        )

    # NSE_LARGE_CAP / NSE_MID_CAP
    s1_price = entry + orb_range * 1.2
    s2_price = entry + orb_range * 1.8
    scales = [
        ScaleLevel(trigger_r=(s1_price - entry) / risk, pct_to_close=0.50, trigger_price=round(s1_price, 4)),
        ScaleLevel(trigger_r=(s2_price - entry) / risk, pct_to_close=0.50, trigger_price=round(s2_price, 4)),
    ]
    return ExitPlan(
        stop_type="atr",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="none",          # no runner in NSE ORB
        trail_trigger_r=0.0,
        max_hold_bars=24,
        hard_exit_time_et="15:45",
        hard_exit_time_ist="11:45",
        breakeven_r=(s1_price - entry) / risk,
        bucket=bucket,
        strategy="ORBBreakout",
    )


def vwap_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    vwap: float,
    atr: float,
    direction: str = "BUY",
) -> ExitPlan:
    """VWAP Mean Reversion — scale at VWAP, trail past it."""
    risk = abs(entry - stop)
    if risk <= 0 or atr <= 0:
        return ExitPlan(strategy="VWAPMeanReversion", bucket=bucket)

    sign = 1 if direction == "BUY" else -1
    # Scale1 = at VWAP (reversion complete)
    s1_price = vwap
    # Scale2 = VWAP ± overshoot
    overshoot_mult = 0.5 if bucket in ("US_ETF",) else (0.4 if "NSE" in bucket else 0.8)
    s2_price = vwap + sign * overshoot_mult * atr

    s1_r = abs(s1_price - entry) / risk
    s2_r = abs(s2_price - entry) / risk

    if bucket in ("US_ETF",):
        scales = [
            ScaleLevel(trigger_r=s1_r, pct_to_close=0.60, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=s2_r, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="ema9_5m",
            trail_trigger_r=s1_r,
            max_hold_bars=12,
            hard_exit_time_et="15:30",
            hard_exit_time_ist="14:45",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="VWAPMeanReversion",
        )

    if bucket == "US_LARGE_CAP":
        scales = [
            ScaleLevel(trigger_r=s1_r, pct_to_close=0.50, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=s2_r, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="ema9_5m",
            trail_trigger_r=s1_r,
            max_hold_bars=16,
            hard_exit_time_et="15:30",
            hard_exit_time_ist="14:45",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="VWAPMeanReversion",
        )

    # NSE — no runner; tighter stop; exit both tranches at fixed levels
    scales = [
        ScaleLevel(trigger_r=s1_r, pct_to_close=0.70, trigger_price=round(s1_price, 4)),
        ScaleLevel(trigger_r=s2_r, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
    ]
    return ExitPlan(
        stop_type="atr",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="none",
        trail_trigger_r=0.0,
        max_hold_bars=10,
        hard_exit_time_et="15:30",
        hard_exit_time_ist="11:30",   # morning session cap
        breakeven_r=s1_r,
        bucket=bucket,
        strategy="VWAPMeanReversion",
    )


def ema_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    atr: float,
    direction: str = "BUY",
) -> ExitPlan:
    """EMA Momentum — tight wick-anchored stop; 3-tranche."""
    risk = abs(entry - stop)
    if risk <= 0 or atr <= 0:
        return ExitPlan(strategy="EMAMomentum", bucket=bucket)

    sign = 1 if direction == "BUY" else -1

    if "NSE" in bucket:
        # NSE: 5m timeframe, EMA21 bounce, faster targets
        s1_price = entry + sign * 1.2 * atr
        s2_price = entry + sign * 2.0 * atr
        s1_r = abs(s1_price - entry) / risk
        s2_r = abs(s2_price - entry) / risk
        scales = [
            ScaleLevel(trigger_r=s1_r, pct_to_close=0.50, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=s2_r, pct_to_close=0.50, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="bar_low_atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="none",
            trail_trigger_r=0.0,
            max_hold_bars=18,
            hard_exit_time_et="15:45",
            hard_exit_time_ist="11:45",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="EMAMomentum",
        )

    # US — 15m timeframe, EMA9 bounce
    s1_price = entry + sign * 1.5 * atr
    s2_price = entry + sign * 2.5 * atr
    s1_r = abs(s1_price - entry) / risk
    s2_r = abs(s2_price - entry) / risk
    scales = [
        ScaleLevel(trigger_r=s1_r, pct_to_close=0.40, trigger_price=round(s1_price, 4)),
        ScaleLevel(trigger_r=s2_r, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
    ]
    return ExitPlan(
        stop_type="bar_low_atr",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="ema9_15m",
        trail_trigger_r=s1_r,
        max_hold_bars=12,
        hard_exit_time_et="14:30",
        hard_exit_time_ist="11:45",
        breakeven_r=s1_r,
        bucket=bucket,
        strategy="EMAMomentum",
    )


def gap_fade_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    gap_close: float,     # prior session close (100% gap fill level)
    gap_pct: float,
    direction: str = "SELL",
) -> ExitPlan:
    """Opening Gap Fade — 60% fill + runner for US; 50% + 80% fixed for NSE."""
    risk = abs(entry - stop)
    if risk <= 0:
        return ExitPlan(strategy="OpeningGapFade", bucket=bucket)

    sign = -1 if direction == "SELL" else 1  # fade short → price moves toward close
    fill_60 = entry + sign * abs(entry - gap_close) * 0.60
    fill_80 = entry + sign * abs(entry - gap_close) * 0.80
    r_60 = abs(fill_60 - entry) / risk
    r_80 = abs(fill_80 - entry) / risk

    if "NSE" in bucket:
        fill_50 = entry + sign * abs(entry - gap_close) * 0.50
        r_50 = abs(fill_50 - entry) / risk
        scales = [
            ScaleLevel(trigger_r=r_50, pct_to_close=0.70, trigger_price=round(fill_50, 4)),
            ScaleLevel(trigger_r=r_80, pct_to_close=0.30, trigger_price=round(fill_80, 4)),
        ]
        return ExitPlan(
            stop_type="pct_gap",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="none",
            trail_trigger_r=0.0,
            max_hold_bars=6,
            hard_exit_time_et="15:45",
            hard_exit_time_ist="11:00",
            breakeven_r=min(r_50 * 0.5, 0.5),
            bucket=bucket,
            strategy="OpeningGapFade",
        )

    # US — 60% fill scale-out, then tight 0.3×ATR trail on runner
    scales = [
        ScaleLevel(trigger_r=r_60, pct_to_close=0.60, trigger_price=round(fill_60, 4)),
    ]
    return ExitPlan(
        stop_type="pct_gap",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="atr_fixed",         # position manager uses 0.3×ATR trail
        trail_trigger_r=r_60,
        trail_atr_mult=0.30,
        max_hold_bars=8,
        hard_exit_time_et="11:30",
        hard_exit_time_ist="15:45",
        breakeven_r=min(r_60 * 0.5, 0.5),
        bucket=bucket,
        strategy="OpeningGapFade",
    )


def supertrend_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    direction: str = "BUY",
) -> ExitPlan:
    """Supertrend Pullback — 30% at 2R then trail ST line to flip."""
    risk = abs(entry - stop)
    if risk <= 0:
        return ExitPlan(strategy="SupertrendTrend", bucket=bucket)

    s1_r = 2.0 if "NSE" not in bucket else 1.5
    sign = 1 if direction == "BUY" else -1
    s1_price = entry + sign * s1_r * risk

    scales = [
        ScaleLevel(trigger_r=s1_r, pct_to_close=0.30, trigger_price=round(s1_price, 4)),
    ]
    # Runner (70%) stays open, trailed by Supertrend line until ST flips.
    # ExitManager detects ST flip via "supertrend_5m" trail_type.
    if "NSE" in bucket:
        return ExitPlan(
            stop_type="st_line_atr",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="supertrend_5m",
            trail_trigger_r=s1_r,
            max_hold_bars=40,
            hard_exit_time_et="15:45",
            hard_exit_time_ist="14:45",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="SupertrendTrend",
        )
    return ExitPlan(
        stop_type="st_line_atr",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="supertrend_5m",
        trail_trigger_r=s1_r,
        max_hold_bars=60,
        hard_exit_time_et="15:45",
        hard_exit_time_ist="14:45",
        breakeven_r=s1_r,
        bucket=bucket,
        strategy="SupertrendTrend",
    )


def nr_squeeze_exit_plan(
    bucket: SymbolBucket,
    entry: float,
    stop: float,
    direction: str = "BUY",
) -> ExitPlan:
    """NR/Bollinger Squeeze Breakout — 2 scale levels + EMA9 runner."""
    risk = abs(entry - stop)
    if risk <= 0:
        return ExitPlan(strategy="NRSqueezeBreakout", bucket=bucket)

    sign = 1 if direction == "BUY" else -1

    if "NSE" in bucket:
        s1_r, s2_r = 1.5, 2.5
        s1_price = entry + sign * s1_r * risk
        s2_price = entry + sign * s2_r * risk
        scales = [
            ScaleLevel(trigger_r=s1_r, pct_to_close=0.50, trigger_price=round(s1_price, 4)),
            ScaleLevel(trigger_r=s2_r, pct_to_close=0.50, trigger_price=round(s2_price, 4)),
        ]
        return ExitPlan(
            stop_type="nr_bar_low",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="none",
            trail_trigger_r=0.0,
            max_hold_bars=24,
            hard_exit_time_et="15:45",
            hard_exit_time_ist="14:00",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="NRSqueezeBreakout",
        )

    if bucket == "US_ETF":
        # ETF: squeeze only, no NR requirement → 2R + EMA9 runner
        s1_r = 2.0
        s1_price = entry + sign * s1_r * risk
        scales = [
            ScaleLevel(trigger_r=s1_r, pct_to_close=0.35, trigger_price=round(s1_price, 4)),
        ]
        return ExitPlan(
            stop_type="nr_bar_low",
            initial_stop_price=round(stop, 4),
            scale_levels=scales,
            trail_type="ema9_5m",
            trail_trigger_r=s1_r,
            max_hold_bars=48,
            hard_exit_time_et="14:30",
            hard_exit_time_ist="14:45",
            breakeven_r=s1_r,
            bucket=bucket,
            strategy="NRSqueezeBreakout",
        )

    # US single names: 2 scales + structure trail
    s1_r, s2_r = 2.0, 3.5
    s1_price = entry + sign * s1_r * risk
    s2_price = entry + sign * s2_r * risk
    scales = [
        ScaleLevel(trigger_r=s1_r, pct_to_close=0.30, trigger_price=round(s1_price, 4)),
        ScaleLevel(trigger_r=s2_r, pct_to_close=0.30, trigger_price=round(s2_price, 4)),
    ]
    return ExitPlan(
        stop_type="nr_bar_low",
        initial_stop_price=round(stop, 4),
        scale_levels=scales,
        trail_type="prior_bar_low_5m",
        trail_trigger_r=s1_r,
        max_hold_bars=48,
        hard_exit_time_et="14:30",
        hard_exit_time_ist="14:45",
        breakeven_r=s1_r,
        bucket=bucket,
        strategy="NRSqueezeBreakout",
    )
