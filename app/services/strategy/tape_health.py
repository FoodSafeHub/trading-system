"""Tape-health gate — veto BUYs into a falling knife.

Root cause this fixes (audited 2026-07-02 over 19 live scheduler fills): every
strategy's "trend intact" filter is slow (SMA200, EMA50-rising-over-5-bars,
SPY regime), so a stock that put in a recent high and then crashed 10-20% in
1-2 weeks still passes every entry gate — the dip-buy strategies (fib pullback,
EMA50 pullback, BB mean reversion) then buy mid-waterfall. The 10 fills that
violated the checks below averaged -4.4%% (post-fill lows -5.7%); the 9 clean
fills averaged +3.7%. This gate measures the SYMBOL'S OWN short-horizon tape —
the dimension no strategy filter covers.

Checks (BUY is vetoed when ANY trips):
    velocity    — 5-session return worse than -max_5d_drop_pct
    streak      — >= max_red_streak consecutive red closes
    ema20_gap   — price more than max_below_ema20_pct below its EMA20
    drawdown    — price more than max_off_20d_high_pct below its 20-day high

Panic-style mean-reversion strategies (RSI2) are exempt from velocity/streak —
buying a short sharp panic IS their edge — but still subject to the deeper
structural checks (ema20_gap, drawdown), which catch a broken trend rather
than a healthy dip.

Fail-open: any data problem returns ok=True. A missing quote must never block
trading (the strategies already ran on real data to produce the signal).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Strategy-name fragments exempt from the velocity/streak checks (panic buyers).
_PANIC_STRATEGY_FRAGMENTS = ("rsi2", "vix_spike")


@dataclass
class TapeHealth:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "healthy"


def _is_panic_strategy(strategy_name: str | None) -> bool:
    s = (strategy_name or "").lower()
    return any(f in s for f in _PANIC_STRATEGY_FRAGMENTS)


def check_tape_health(
    symbol: str,
    strategy_name: str | None = None,
    *,
    max_5d_drop_pct: float = 6.0,
    max_red_streak: int = 4,
    max_below_ema20_pct: float = 5.0,
    max_off_20d_high_pct: float = 12.0,
    ohlcv=None,
) -> TapeHealth:
    """Evaluate the symbol's own short-horizon tape before a BUY.

    Returns TapeHealth(ok=False, reasons=[...]) when the tape is in freefall.
    `ohlcv` may be passed to skip the fetch (tests / callers with data in hand).
    """
    try:
        if ohlcv is None:
            from app.services.market_data.provider import get_ohlcv
            ohlcv = get_ohlcv(symbol, period="3mo")
        closes = ohlcv["Close"].dropna()
        if len(closes) < 25:
            return TapeHealth(ok=True, metrics={"insufficient_bars": len(closes)})

        import pandas as pd  # noqa: F401  (closes is already a Series)

        px = float(closes.iloc[-1])
        ret_5d = (px / float(closes.iloc[-6]) - 1.0) * 100.0
        highs = ohlcv["High"].dropna() if "High" in ohlcv else closes
        hi20 = float(highs.iloc[-21:].max())
        off_hi20 = (px / hi20 - 1.0) * 100.0 if hi20 > 0 else 0.0
        ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        vs_ema20 = (px / ema20 - 1.0) * 100.0 if ema20 > 0 else 0.0

        streak = 0
        for i in range(len(closes) - 1, 0, -1):
            if float(closes.iloc[i]) < float(closes.iloc[i - 1]):
                streak += 1
            else:
                break

        metrics = {
            "ret_5d_pct": round(ret_5d, 2),
            "off_20d_high_pct": round(off_hi20, 2),
            "vs_ema20_pct": round(vs_ema20, 2),
            "red_streak": streak,
        }

        panic = _is_panic_strategy(strategy_name)
        reasons: list[str] = []
        if not panic:
            if ret_5d < -max_5d_drop_pct:
                reasons.append(f"5d return {ret_5d:+.1f}% (limit -{max_5d_drop_pct:.0f}%)")
            if streak >= max_red_streak:
                reasons.append(f"{streak} consecutive red closes (limit {max_red_streak})")
        if vs_ema20 < -max_below_ema20_pct:
            reasons.append(f"{vs_ema20:+.1f}% below EMA20 (limit -{max_below_ema20_pct:.0f}%)")
        if off_hi20 < -max_off_20d_high_pct:
            reasons.append(
                f"{off_hi20:+.1f}% off 20d high (limit -{max_off_20d_high_pct:.0f}%)"
            )

        return TapeHealth(ok=not reasons, reasons=reasons, metrics=metrics)
    except Exception as exc:
        logger.warning("[tape_health] %s check failed (fail-open): %s", symbol, exc)
        return TapeHealth(ok=True, metrics={"error": str(exc)})
