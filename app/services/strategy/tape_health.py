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
        # Rule 1 — VELOCITY (primary). Calibrated 2026-07-02 on 359 backtest
        # round-trips across 15 symbols: 5d<-6% blocks avg +1.0%/54%wr vs
        # allowed +2.6%/77%wr — the single best separator. Panic strategies
        # (RSI2/VIX-spike) exempt: buying the fast dip is their edge.
        if not panic and ret_5d < -max_5d_drop_pct:
            reasons.append(f"5d return {ret_5d:+.1f}% (limit -{max_5d_drop_pct:.0f}%)")
        # Rule 2 — STRUCTURAL BREAK: far below EMA20 AND deep off the 20d high
        # TOGETHER. Each alone blocked too many winners (standalone off-20d-high
        # blocks averaged +2.2%, and deep-drawdown entries actually bounced);
        # combined they mark a broken trend, not a healthy dip.
        if vs_ema20 < -max_below_ema20_pct and off_hi20 < -max_off_20d_high_pct:
            reasons.append(
                f"structural break: {vs_ema20:+.1f}% below EMA20 and "
                f"{off_hi20:+.1f}% off 20d high"
            )
        # NOTE: consecutive-red-closes was REMOVED as a standalone blocker
        # (study: streak>=4 blocks averaged +1.8%/71%wr — low-volatility names
        # like AAPL/KO drift down 4 sessions then bounce; it filtered winners).
        # The streak stays in metrics for display. max_red_streak is kept in
        # the signature for config compatibility but no longer gates alone.
        _ = max_red_streak

        return TapeHealth(ok=not reasons, reasons=reasons, metrics=metrics)
    except Exception as exc:
        logger.warning("[tape_health] %s check failed (fail-open): %s", symbol, exc)
        return TapeHealth(ok=True, metrics={"error": str(exc)})


def verdict_line(th: TapeHealth) -> str:
    """One-line human verdict for notifications / scan reasons / trade rows."""
    m = th.metrics or {}
    if "ret_5d_pct" not in m:
        return "Tape gate: skipped (insufficient data)"
    if th.ok:
        return (f"Tape gate: PASSED — 5d {m['ret_5d_pct']:+.1f}%, "
                f"{m['red_streak']} red, EMA20 {m['vs_ema20_pct']:+.1f}%, "
                f"20d-high {m['off_20d_high_pct']:+.1f}%")
    return f"Tape gate: BLOCKED — {th.summary}"


def annotate_backtest_trades(
    symbol: str,
    trades: list[dict],
    ohlcv,
    strategy_name: str | None = None,
) -> dict:
    """Stamp a point-in-time tape-gate verdict on every BUY in a backtest trade
    list, and summarize how gated vs clean entries performed.

    trades: list of dicts with at least {date, side, value} (the shape every
    backtest endpoint already returns). Mutated in place — each BUY gains
    `tape_gate` ("pass"|"block") and `tape_gate_reason`.

    Returns a summary dict:
        {"pass":  {"round_trips": n, "win_rate_pct": x, "avg_return_pct": y},
         "block": {...same...},
         "blocked_buys": n_blocked}
    so the UI can show, per strategy, what the knife veto would have done to
    the historical trade set. Best-effort: returns {} on any failure and never
    raises into the endpoint.
    """
    try:
        import pandas as pd
        df = ohlcv.copy()
        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            pass  # already naive

        buckets: dict[str, list[float]] = {"pass": [], "block": []}
        n_blocked = 0
        open_verdict: str | None = None

        for t in trades:
            side = (t.get("side") or "").upper()
            if side in ("SELL_SIGNAL", "SHORT", "COVER"):
                continue  # markers / short legs — the knife veto is a LONG-entry gate
            if side == "BUY":
                hist = df.loc[: str(t.get("date"))[:10]]
                th = check_tape_health(symbol, strategy_name, ohlcv=hist)
                verdict = "pass" if th.ok else "block"
                t["tape_gate"] = verdict
                t["tape_gate_reason"] = th.summary if not th.ok else ""
                open_verdict = verdict
                if verdict == "block":
                    n_blocked += 1
            elif "SELL" in side and open_verdict is not None:
                # Pair with the most recent BUY (engines emit alternating
                # BUY/SELL round-trips).
                buy = next(
                    (x for x in reversed(trades[: trades.index(t)])
                     if (x.get("side") or "").upper() == "BUY"),
                    None,
                )
                if buy and (buy.get("value") or 0) > 0:
                    pct = ((t.get("value") or 0) - buy["value"]) / buy["value"] * 100
                    buckets[open_verdict].append(pct)
                open_verdict = None

        def _stats(rets: list[float]) -> dict:
            if not rets:
                return {"round_trips": 0, "win_rate_pct": None, "avg_return_pct": None}
            wins = sum(1 for r in rets if r > 0)
            return {
                "round_trips": len(rets),
                "win_rate_pct": round(wins / len(rets) * 100, 1),
                "avg_return_pct": round(sum(rets) / len(rets), 2),
            }

        return {
            "pass": _stats(buckets["pass"]),
            "block": _stats(buckets["block"]),
            "blocked_buys": n_blocked,
        }
    except Exception as exc:
        logger.warning("[tape_health] trade annotation failed for %s: %s", symbol, exc)
        return {}
