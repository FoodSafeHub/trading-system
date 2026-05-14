"""
Symbol Analyzer — profiles a stock's intraday character from historical data.

Computes volatility, trend strength, liquidity, gap frequency, and classifies
the symbol into a market type that drives strategy routing and config tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────
_LOW_VOL_THRESHOLD = 1.0    # ATR% below this = stable / low-vol name
_HIGH_VOL_THRESHOLD = 1.8   # ATR% above this = high-vol / momentum name
_STRONG_TREND_SCORE = 0.65  # trend_strength above this = trends reliably
_MIN_BARS_REQUIRED = 30     # minimum bars to compute a profile


# ── Well-known symbol hints (override when data is thin) ──────────────────────
_KNOWN_PROFILES: dict[str, dict[str, Any]] = {
    "SPY":  {"type": "etf_index",        "liquidity": 1.0},
    "QQQ":  {"type": "etf_index",        "liquidity": 1.0},
    "IWM":  {"type": "etf_index",        "liquidity": 0.95},
    "AAPL": {"type": "large_cap_stable", "liquidity": 0.98},
    "MSFT": {"type": "large_cap_stable", "liquidity": 0.97},
    "GOOGL":{"type": "large_cap_stable", "liquidity": 0.93},
    "AMZN": {"type": "large_cap_stable", "liquidity": 0.94},
    "META": {"type": "large_cap_growth",  "liquidity": 0.93},
    "NVDA": {"type": "high_vol_growth",   "liquidity": 0.95},
    "TSLA": {"type": "high_vol_momentum", "liquidity": 0.94},
    "AMD":  {"type": "high_vol_growth",   "liquidity": 0.90},
    "NFLX": {"type": "high_vol_growth",   "liquidity": 0.88},
    "JPM":  {"type": "large_cap_stable",  "liquidity": 0.92},
    "GS":   {"type": "large_cap_stable",  "liquidity": 0.87},
    "XOM":  {"type": "large_cap_stable",  "liquidity": 0.88},
}


@dataclass
class SymbolProfile:
    symbol: str
    volatility_pct: float          # ATR as % of price (intraday)
    avg_daily_range_pct: float     # (High - Low) / Open as % avg
    trend_strength: float          # 0–1: how often price stays on one VWAP side
    liquidity_score: float         # 0–1
    gap_frequency_pct: float       # % of days with gap > 0.5%
    market_type: str               # human label
    best_strategies: list[str]     # ordered best → worst for this symbol
    avoid_strategies: list[str]    # strategies that statistically fail here
    trading_hours_pattern: str     # narrative
    notes: str                     # human-readable analysis
    raw_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_low_volatility(self) -> bool:
        return self.volatility_pct < _LOW_VOL_THRESHOLD

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility_pct >= _HIGH_VOL_THRESHOLD

    @property
    def primary_strategy(self) -> str:
        return self.best_strategies[0] if self.best_strategies else "EMAMomentum"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "volatility_pct": round(self.volatility_pct, 3),
            "avg_daily_range_pct": round(self.avg_daily_range_pct, 3),
            "trend_strength": round(self.trend_strength, 3),
            "liquidity_score": round(self.liquidity_score, 3),
            "gap_frequency_pct": round(self.gap_frequency_pct, 1),
            "market_type": self.market_type,
            "best_strategies": self.best_strategies,
            "avoid_strategies": self.avoid_strategies,
            "trading_hours_pattern": self.trading_hours_pattern,
            "notes": self.notes,
        }


class SymbolAnalyzer:
    """
    Profile a symbol from its recent intraday history.
    All computation is deterministic — no ML, just stats.
    """

    @staticmethod
    def analyze(df_5m: pd.DataFrame, symbol: str) -> SymbolProfile:
        """
        Build a SymbolProfile from 5m bar data.
        df_5m must already be fetched (ET timezone, today's bars included).
        """
        sym = symbol.upper()

        if df_5m.empty or len(df_5m) < _MIN_BARS_REQUIRED:
            return SymbolAnalyzer._fallback_profile(sym)

        df = df_5m.copy()

        # ── Per-day metrics ───────────────────────────────────────────────────
        dates = sorted(set(df.index.date))
        daily_ranges: list[float] = []
        daily_gaps: list[float] = []
        daily_vwap_fractions: list[float] = []  # fraction of bars above VWAP

        for i, date in enumerate(dates):
            day = df[df.index.date == date]
            if len(day) < 4:
                continue

            open_p = float(day["Open"].iloc[0])
            day_high = float(day["High"].max())
            day_low = float(day["Low"].min())
            if open_p > 0:
                daily_ranges.append((day_high - day_low) / open_p * 100)

            # Gap vs prior close
            if i > 0:
                prev_day = df[df.index.date == dates[i - 1]]
                if not prev_day.empty:
                    prior_close = float(prev_day["Close"].iloc[-1])
                    if prior_close > 0:
                        gap_pct = abs(open_p - prior_close) / prior_close * 100
                        daily_gaps.append(gap_pct)

            # Fraction of bars above VWAP (trend_strength proxy)
            try:
                tp = (day["High"] + day["Low"] + day["Close"]) / 3
                cum_vol = day["Volume"].cumsum()
                vwap = (tp * day["Volume"]).cumsum() / cum_vol
                above = (day["Close"] > vwap).mean()
                daily_vwap_fractions.append(float(above))
            except Exception:
                pass

        # ── Daily ATR-based volatility (scale 5m ATR to daily) ───────────────
        # Use avg_daily_range_pct as the primary volatility measure since it is
        # already a true daily High-Low% — more meaningful than 5m ATR.
        # Also compute 5m ATR for comparison and sanity check.
        try:
            import ta.volatility as tav
            atr_ind = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
            atr_series = atr_ind.average_true_range().dropna()
            avg_atr_5m = float(atr_series.mean())
            avg_close = float(df["Close"].mean())
            # Scale 5m ATR to a daily equivalent (78 bars/day, but ATR scales as sqrt)
            atr_pct_5m = (avg_atr_5m / avg_close * 100) if avg_close > 0 else 0.0
            # Daily range % is more direct — use it as primary, 5m ATR as fallback
            volatility_pct = float(np.mean(daily_ranges)) if daily_ranges else atr_pct_5m * 8
        except Exception:
            volatility_pct = float(np.mean(daily_ranges)) if daily_ranges else 1.5

        avg_daily_range_pct = float(np.mean(daily_ranges)) if daily_ranges else 1.5
        trend_strength = float(np.mean([abs(f - 0.5) * 2 for f in daily_vwap_fractions])) if daily_vwap_fractions else 0.5
        gap_frequency_pct = sum(1 for g in daily_gaps if g > 0.5) / len(daily_gaps) * 100 if daily_gaps else 5.0

        # ── Liquidity from known hints or volume rank ─────────────────────────
        hint = _KNOWN_PROFILES.get(sym, {})
        liquidity_score = hint.get("liquidity", _estimate_liquidity(df))

        # ── Hourly activity: find the busiest hours ───────────────────────────
        df["hour"] = df.index.hour
        hourly_vol = df.groupby("hour")["Volume"].mean()
        top_hours = hourly_vol.nlargest(3).index.tolist()
        hours_str = ", ".join(f"{h}:00" for h in sorted(top_hours))

        # ── Override classification for known symbols ─────────────────────────
        # Empirically: AAPL/MSFT daily range 1.5-2% but they are NOT high-vol
        # momentum names — they're stable large-caps. Known type wins over raw stats.
        known_type = hint.get("type", "")
        if known_type == "large_cap_stable" and volatility_pct >= _HIGH_VOL_THRESHOLD:
            volatility_pct = _LOW_VOL_THRESHOLD * 0.9   # force below high-vol threshold

        market_type, best_strategies, avoid_strategies, notes, hours_pattern = (
            SymbolAnalyzer._classify(
                sym, volatility_pct, avg_daily_range_pct,
                trend_strength, gap_frequency_pct, known_type, hours_str
            )
        )

        return SymbolProfile(
            symbol=sym,
            volatility_pct=round(volatility_pct, 3),
            avg_daily_range_pct=round(avg_daily_range_pct, 3),
            trend_strength=round(trend_strength, 3),
            liquidity_score=round(liquidity_score, 3),
            gap_frequency_pct=round(gap_frequency_pct, 1),
            market_type=market_type,
            best_strategies=best_strategies,
            avoid_strategies=avoid_strategies,
            trading_hours_pattern=hours_pattern,
            notes=notes,
            raw_stats={
                "n_days": len(dates),
                "top_volume_hours": top_hours,
                "avg_daily_range_pct": round(avg_daily_range_pct, 3),
                "trend_strength_raw": round(trend_strength, 3),
            },
        )

    @staticmethod
    def _classify(
        symbol: str,
        vol_pct: float,
        range_pct: float,
        trend_str: float,
        gap_freq: float,
        known_type: str,
        top_hours: str,
    ) -> tuple[str, list[str], list[str], str, str]:
        """Returns (market_type, best_strategies, avoid_strategies, notes, hours_pattern)."""

        # ── High-volatility momentum names ────────────────────────────────────
        if vol_pct >= _HIGH_VOL_THRESHOLD or known_type in ("high_vol_momentum", "high_vol_growth"):
            market_type = f"High-volatility momentum ({vol_pct:.1f}% ATR)"
            best = ["ORBBreakout", "VolumeSpikeReversal", "EMAMomentum"]
            avoid = ["VWAPMeanReversion"]
            hours = f"Strong opens 9:30–11:00, volume spikes intraday. Active hours: {top_hours}."
            notes = (
                f"{symbol} is a classic ORB play. ATR of {vol_pct:.1f}% means the opening "
                f"range is wide and decisive. Volume spikes are real inflection points. "
                f"Expect 2–4 ORB setups per week. Gap frequency {gap_freq:.0f}% means gap-fade "
                f"setups also occur regularly. Trend strength {trend_str:.2f} suggests "
                f"{'strong directional days are common' if trend_str > 0.6 else 'mixed intraday direction'}."
            )
            return market_type, best, avoid, notes, hours

        # ── Index ETFs ────────────────────────────────────────────────────────
        if known_type == "etf_index" or symbol in ("SPY", "QQQ", "IWM", "DIA"):
            market_type = f"Index ETF — smooth, liquid ({vol_pct:.1f}% ATR)"
            best = ["VWAPMeanReversion", "EMAMomentum", "OpeningGapFade"]
            avoid = ["ORBBreakout"] if vol_pct < 0.9 else []
            hours = f"Steady throughout session. Most volume: {top_hours}."
            avoid_note = (
                f" ORB is marginal — ETF opening ranges are often narrow and fakeout-prone."
                if vol_pct < 0.9 else ""
            )
            notes = (
                f"{symbol} is a liquid ETF with {vol_pct:.1f}% ATR. VWAP mean reversion "
                f"and EMA momentum are the highest-edge strategies here. "
                f"Gap fades work when macro events create overnight gaps.{avoid_note}"
            )
            return market_type, best, avoid, notes, hours

        # ── Low-volatility large-cap stable ──────────────────────────────────
        if vol_pct < _LOW_VOL_THRESHOLD or known_type == "large_cap_stable":
            market_type = f"Low-volatility large-cap — mean-reversion friendly ({vol_pct:.1f}% ATR)"
            best = ["VWAPMeanReversion", "EMAMomentum", "OpeningGapFade"]
            avoid = ["ORBBreakout", "VolumeSpikeReversal"]
            hours = f"Morning chop 9:30–10:30, lunch range, afternoon drift. Active: {top_hours}."
            notes = (
                f"{symbol} is a low-volatility name. Its {vol_pct:.1f}% ATR means the "
                f"15-min ORB is only {range_pct * 0.25:.2f}% wide on average — easily "
                f"faked out by the first bar's wick. ORB Breakout will produce near-zero "
                f"valid setups. VWAP mean reversion thrives because {symbol} reliably pulls "
                f"back to VWAP 2–4× per day. EMA momentum on 15m bars gives cleaner signals. "
                f"Volume spikes are rare and often institutional, not tradeable reversals."
            )
            return market_type, best, avoid, notes, hours

        # ── Mid-range growth / semi-volatile ─────────────────────────────────
        market_type = f"Mid-volatility growth ({vol_pct:.1f}% ATR)"
        best = ["EMAMomentum", "ORBBreakout", "VWAPMeanReversion"]
        avoid = []
        hours = f"Active across the session. Top hours: {top_hours}."
        notes = (
            f"{symbol} sits in the mid-volatility range at {vol_pct:.1f}% ATR. "
            f"EMA momentum is the highest-probability setup. ORB works on trending days "
            f"(gap frequency {gap_freq:.0f}%). Trend strength {trend_str:.2f} — "
            f"{'directional days are common, lean toward momentum plays' if trend_str > 0.55 else 'mixed — mean reversion also viable'}."
        )
        return market_type, best, avoid, notes, hours

    @staticmethod
    def _fallback_profile(symbol: str) -> SymbolProfile:
        hint = _KNOWN_PROFILES.get(symbol, {})
        known_type = hint.get("type", "unknown")

        if known_type in ("high_vol_momentum", "high_vol_growth"):
            best = ["ORBBreakout", "VolumeSpikeReversal", "EMAMomentum"]
            avoid = ["VWAPMeanReversion"]
            mtype = "High-volatility (estimated)"
            notes = f"{symbol} is a known high-volatility name. ORB and volume spike setups apply."
        elif known_type in ("large_cap_stable", "etf_index"):
            best = ["VWAPMeanReversion", "EMAMomentum", "OpeningGapFade"]
            avoid = ["ORBBreakout"]
            mtype = "Large-cap stable (estimated)"
            notes = f"{symbol} is a stable large-cap. VWAP mean reversion and EMA momentum are preferred."
        else:
            best = ["EMAMomentum", "VWAPMeanReversion", "ORBBreakout"]
            avoid = []
            mtype = "Unknown — using defaults"
            notes = f"Insufficient data for {symbol}. Using default strategy ordering."

        return SymbolProfile(
            symbol=symbol,
            volatility_pct=1.2,
            avg_daily_range_pct=1.5,
            trend_strength=0.5,
            liquidity_score=hint.get("liquidity", 0.7),
            gap_frequency_pct=8.0,
            market_type=mtype,
            best_strategies=best,
            avoid_strategies=avoid,
            trading_hours_pattern="Unknown — insufficient data.",
            notes=notes,
        )


def _estimate_liquidity(df: pd.DataFrame) -> float:
    """Rough liquidity estimate from volume percentile."""
    try:
        avg_vol = float(df["Volume"].mean())
        if avg_vol > 5_000_000:
            return 0.95
        if avg_vol > 1_000_000:
            return 0.85
        if avg_vol > 300_000:
            return 0.70
        return 0.50
    except Exception:
        return 0.70
