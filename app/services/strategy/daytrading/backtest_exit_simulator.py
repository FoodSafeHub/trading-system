"""
Bar-by-bar exit simulator for the day-trading backtester.

Honors the same `ExitPlan` schema (`risk_templates.ExitPlan`) the live
`autotrader.ExitManager` consumes, so the backtest evaluates the same trading
model the live engine runs. Without this, the backtester collapsed every
ExitPlan into a single-bracket trade (entry → stop OR target OR time-exit) and
ignored breakeven, scale-outs, and trailing — giving a materially different
P&L distribution than live.

Why not call the live ExitManager directly?
    The live ExitManager is bound to wall-clock helpers (now_et, stale-bar
    detection) and processes one position at a time on streaming bars; that
    machinery has no meaning in a bar-by-bar backtest. This module re-uses the
    exact same ExitPlan semantics but walks future bars deterministically.

Intra-bar fill model
    Conservative stop-before-target ordering: if a single bar's range touches
    both the current stop and a scale/target/trail level, the stop fires first.
    This matches the legacy backtester behavior and is the safer assumption.

What this DOES NOT model (intentionally)
    Live-only behaviors that have no bar-by-bar equivalent are not simulated:
    momentum-fade exits, stale-bar safety exits, NEWS_RISK regime flatten,
    Supertrend-flip runners. Those are explicitly live-execution overlays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd

from app.services.strategy.daytrading.market_open import market_session
from app.services.strategy.daytrading.risk_templates import ExitPlan, ScaleLevel


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class ExitLeg:
    """One realised fill that closed (part of) the position."""
    qty: float                  # shares closed in this leg
    price: float                # fill price (level price, never bar close beyond it)
    time: pd.Timestamp
    reason: str                 # e.g. "scale_1", "stop", "trail", "eod"


@dataclass
class SimulatedExit:
    """Outcome of walking future bars under an ExitPlan."""
    legs: list[ExitLeg] = field(default_factory=list)
    final_stop: float = 0.0      # stop level when the last leg closed
    hold_bars: int = 0
    breakeven_hit: bool = False
    trail_activated: bool = False
    primary_outcome: str = "OPEN"   # rolled-up exit reason for compatibility

    @property
    def vwap_exit_price(self) -> float:
        """Qty-weighted average exit price across all legs."""
        total_qty = sum(l.qty for l in self.legs)
        if total_qty <= 0:
            return 0.0
        return sum(l.qty * l.price for l in self.legs) / total_qty

    @property
    def closed_qty(self) -> float:
        return sum(l.qty for l in self.legs)


# ── Public entry point ──────────────────────────────────────────────────────


def simulate_exit(
    direction: str,
    entry_price: float,
    initial_stop: float,
    initial_target: float,
    qty: float,
    signal_time: pd.Timestamp,
    future_bars: pd.DataFrame,
    exit_plan: ExitPlan | None,
    symbol: str = "",
    max_hold_bars_default: int = 60,
) -> SimulatedExit:
    """
    Walk `future_bars` forward and close `qty` according to `exit_plan`.

    If `exit_plan` is None, falls back to the legacy single-bracket behavior
    (stop OR target OR time-exit OR EOD-close) — full back-compat for strategies
    that do not yet emit an ExitPlan.

    Parameters
    ----------
    direction : "BUY" (long) or "SELL"/"SELL_SHORT" (short)
    entry_price : fill price assumed for risk math; bar fills come from the
        executing layer (FillSimulator) — pass the raw signal entry here.
    initial_stop, initial_target : as the signal generated them.
    qty : total shares opened.
    signal_time : timestamp of entry bar; used only to assert future_bars are
        post-entry (caller already slices to bars > signal_time).
    future_bars : OHLCV DataFrame indexed by timestamp, ascending.
    exit_plan : the strategy's ExitPlan (may be None).
    symbol : used to pick session (IST/ET) for hard_exit_time enforcement.
    max_hold_bars_default : used when exit_plan is None.

    Returns
    -------
    SimulatedExit with one or more legs whose qty sums to `qty` (or less, if
    the bar feed runs out — caller treats the remainder as an EOD close).
    """
    side = "LONG" if direction == "BUY" else "SHORT"

    # No future bars at all → nothing to simulate.
    if future_bars is None or future_bars.empty or qty <= 0:
        return SimulatedExit(primary_outcome="NO_FUTURE_BARS")

    # Resolve session for hard-time exit (ET for US, IST for India).
    sess = market_session(symbol) if symbol else None
    hard_exit_clock: time | None = _resolve_hard_exit_time(exit_plan, symbol)

    # Effective max-hold (ExitPlan wins, else strategy default).
    max_hold = exit_plan.max_hold_bars if exit_plan else max_hold_bars_default

    # Mutable state for the walk.
    current_stop = initial_stop
    remaining = qty
    scale_idx = 0
    scale_levels: list[ScaleLevel] = list(exit_plan.scale_levels) if exit_plan else []
    breakeven_r = exit_plan.breakeven_r if exit_plan else None
    trail_type = exit_plan.trail_type if exit_plan else "none"
    trail_trigger_r = exit_plan.trail_trigger_r if exit_plan else 0.0
    trail_atr_mult = exit_plan.trail_atr_mult if exit_plan else 1.5

    out = SimulatedExit(final_stop=current_stop)
    risk_unit = abs(entry_price - initial_stop)

    # Running excursions for the trail (high since entry for LONG, low for SHORT).
    excursion_high = float(future_bars["High"].iloc[0])
    excursion_low = float(future_bars["Low"].iloc[0])

    for bar_idx, (ts, bar) in enumerate(future_bars.iterrows()):
        out.hold_bars = bar_idx + 1
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])

        # Update excursion bounds (used by trail).
        if bar_high > excursion_high:
            excursion_high = bar_high
        if bar_low < excursion_low:
            excursion_low = bar_low

        # ── 1. Hard-time exit (session EOD) ─────────────────────────────────
        if hard_exit_clock is not None:
            bar_time = ts.time() if hasattr(ts, "time") else None
            if bar_time is not None and bar_time >= hard_exit_clock:
                out.legs.append(ExitLeg(
                    qty=remaining, price=bar_close, time=ts,
                    reason=f"hard_exit_{hard_exit_clock.strftime('%H:%M')}",
                ))
                remaining = 0
                out.primary_outcome = _rollup_outcome(out.legs, "EOD_EXIT")
                out.final_stop = current_stop
                return out

        # ── 2. Stop hit ─────────────────────────────────────────────────────
        # Conservative ordering: stop checked before targets/scales within the
        # same bar. This matches the legacy backtester behavior.
        if side == "LONG" and bar_low <= current_stop:
            out.legs.append(ExitLeg(
                qty=remaining, price=current_stop, time=ts,
                reason=("trail_stop" if out.trail_activated else
                        ("breakeven_stop" if out.breakeven_hit else "stop")),
            ))
            remaining = 0
            out.primary_outcome = _rollup_outcome(out.legs, "STOPPED")
            out.final_stop = current_stop
            return out

        if side == "SHORT" and bar_high >= current_stop:
            out.legs.append(ExitLeg(
                qty=remaining, price=current_stop, time=ts,
                reason=("trail_stop" if out.trail_activated else
                        ("breakeven_stop" if out.breakeven_hit else "stop")),
            ))
            remaining = 0
            out.primary_outcome = _rollup_outcome(out.legs, "STOPPED")
            out.final_stop = current_stop
            return out

        # Compute R reached this bar (using the favorable extreme).
        if risk_unit > 0:
            if side == "LONG":
                r_reached = (excursion_high - entry_price) / risk_unit
            else:
                r_reached = (entry_price - excursion_low) / risk_unit
        else:
            r_reached = 0.0

        # ── 3. Break-even stop move ─────────────────────────────────────────
        if (
            exit_plan is not None
            and not out.breakeven_hit
            and breakeven_r is not None
            and r_reached >= breakeven_r
        ):
            # Move stop to entry (tightest allowed direction only).
            if side == "LONG" and entry_price > current_stop:
                current_stop = entry_price
            elif side == "SHORT" and entry_price < current_stop:
                current_stop = entry_price
            out.breakeven_hit = True
            out.final_stop = current_stop

        # ── 4. Scale-out tiers ──────────────────────────────────────────────
        # Walk tiers in ascending trigger_r order; close the configured
        # fraction of REMAINING qty at the trigger_price (or bar_close if the
        # tier has no fixed price). Multiple tiers can fire in one bar.
        while (
            scale_levels
            and scale_idx < len(scale_levels)
            and remaining > 0
        ):
            lvl = scale_levels[scale_idx]
            level_price = lvl.trigger_price if lvl.trigger_price > 0 else None
            tier_hit = False
            fill_price = bar_close

            if level_price is not None:
                if side == "LONG" and bar_high >= level_price:
                    tier_hit = True
                    fill_price = level_price
                elif side == "SHORT" and bar_low <= level_price:
                    tier_hit = True
                    fill_price = level_price
            else:
                # Trigger by R if no explicit price.
                if r_reached >= lvl.trigger_r and risk_unit > 0:
                    tier_hit = True
                    fill_price = bar_close

            if not tier_hit:
                break

            close_qty = _round_qty(remaining * lvl.pct_to_close)
            # If rounding yielded 0 but tier was hit, take at least 1 share
            # so the tier visibly fires (matches live behavior).
            if close_qty <= 0 and remaining >= 1:
                close_qty = min(1.0, remaining)
            if close_qty > remaining:
                close_qty = remaining

            if close_qty > 0:
                out.legs.append(ExitLeg(
                    qty=close_qty, price=fill_price, time=ts,
                    reason=f"scale_{scale_idx + 1}",
                ))
                remaining -= close_qty

            scale_idx += 1
            # If this is the trail trigger or all scales are done, activation
            # may flip below — fall through to the trail block.

        if remaining <= 0:
            out.primary_outcome = _rollup_outcome(out.legs, "TARGET")
            out.final_stop = current_stop
            return out

        # ── 5. Trail activation + ratchet ───────────────────────────────────
        if (
            exit_plan is not None
            and trail_type != "none"
            and not out.trail_activated
            and r_reached >= trail_trigger_r
        ):
            out.trail_activated = True

        if out.trail_activated and trail_type != "none":
            new_stop = _compute_trail_stop(
                trail_type=trail_type,
                side=side,
                future_bars=future_bars,
                up_to_idx=bar_idx,
                entry_price=entry_price,
                trail_atr_mult=trail_atr_mult,
            )
            if new_stop is not None:
                # Only ratchet in our favor.
                if side == "LONG" and new_stop > current_stop:
                    current_stop = new_stop
                    out.final_stop = current_stop
                elif side == "SHORT" and new_stop < current_stop:
                    current_stop = new_stop
                    out.final_stop = current_stop

        # ── 6. Legacy single target (only when no scale-out tiers) ─────────
        if not scale_levels and initial_target > 0:
            if side == "LONG" and bar_high >= initial_target:
                out.legs.append(ExitLeg(
                    qty=remaining, price=initial_target, time=ts,
                    reason="target",
                ))
                remaining = 0
                out.primary_outcome = _rollup_outcome(out.legs, "TARGET")
                out.final_stop = current_stop
                return out
            if side == "SHORT" and bar_low <= initial_target:
                out.legs.append(ExitLeg(
                    qty=remaining, price=initial_target, time=ts,
                    reason="target",
                ))
                remaining = 0
                out.primary_outcome = _rollup_outcome(out.legs, "TARGET")
                out.final_stop = current_stop
                return out

        # ── 7. Max-hold time exit ───────────────────────────────────────────
        if out.hold_bars >= max_hold:
            out.legs.append(ExitLeg(
                qty=remaining, price=bar_close, time=ts,
                reason="max_hold_bars",
            ))
            remaining = 0
            out.primary_outcome = _rollup_outcome(out.legs, "TIME_EXIT")
            out.final_stop = current_stop
            return out

    # Ran out of bars before anything closed the remainder → EOD close.
    if remaining > 0:
        last_ts = future_bars.index[-1]
        last_close = float(future_bars["Close"].iloc[-1])
        out.legs.append(ExitLeg(
            qty=remaining, price=last_close, time=last_ts,
            reason="eod",
        ))
        out.primary_outcome = _rollup_outcome(out.legs, "EOD_EXIT")
    out.final_stop = current_stop
    return out


# ── Internals ───────────────────────────────────────────────────────────────


def _resolve_hard_exit_time(exit_plan: ExitPlan | None, symbol: str) -> time | None:
    """Pick the right hard-exit clock for this symbol's market (ET vs IST)."""
    if exit_plan is None or not symbol:
        return None
    try:
        # is_india_symbol lives in markets.py and is the existing classifier.
        from app.services.markets import is_india_symbol
        is_india = is_india_symbol(symbol)
    except Exception:
        is_india = False
    raw = exit_plan.hard_exit_time_ist if is_india else exit_plan.hard_exit_time_et
    if not raw or ":" not in raw:
        return None
    try:
        h, m = raw.split(":")
        return time(int(h), int(m))
    except Exception:
        return None


def _compute_trail_stop(
    trail_type: str,
    side: str,
    future_bars: pd.DataFrame,
    up_to_idx: int,
    entry_price: float,
    trail_atr_mult: float,
) -> float | None:
    """
    Compute the new candidate trailing stop after bar `up_to_idx` closed.

    Supported trail types (mirrors risk_templates.TrailType):
        ema9_5m         — EMA9 on bars up to up_to_idx (5m series)
        ema9_15m        — EMA9 on the same series (caller passes 15m if needed)
        supertrend_5m   — uses prior bar low (long) / high (short) as a proxy;
                          the full Supertrend flip exit is a live-only overlay
        prior_bar_low_5m — prior bar's low (long) / high (short)
        atr_fixed       — last_close ± trail_atr_mult × ATR(14)

    Returns None if the inputs aren't ready (not enough bars, NaN).
    """
    if up_to_idx < 0 or up_to_idx >= len(future_bars):
        return None

    window = future_bars.iloc[: up_to_idx + 1]
    if window.empty:
        return None

    last_close = float(window["Close"].iloc[-1])

    if trail_type == "prior_bar_low_5m" or trail_type == "supertrend_5m":
        if len(window) < 2:
            return None
        prev_bar = window.iloc[-2]
        if side == "LONG":
            return float(prev_bar["Low"])
        return float(prev_bar["High"])

    if trail_type in ("ema9_5m", "ema9_15m"):
        # Simple EMA9 on the close series; OK for backtest fidelity.
        if len(window) < 9:
            return None
        ema9 = window["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
        if pd.isna(ema9):
            return None
        return float(ema9)

    if trail_type == "atr_fixed":
        if len(window) < 14:
            return None
        high = window["High"].astype(float)
        low = window["Low"].astype(float)
        close = window["Close"].astype(float)
        tr = (high - low).combine(
            (high - close.shift(1)).abs(), max
        ).combine((low - close.shift(1)).abs(), max)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        if side == "LONG":
            return last_close - trail_atr_mult * float(atr)
        return last_close + trail_atr_mult * float(atr)

    return None


def _round_qty(q: float) -> float:
    """Integer share count (matches live behavior; fractional shares not modelled)."""
    return float(int(q))


def _rollup_outcome(legs: list[ExitLeg], default: str) -> str:
    """
    Map the final leg's reason to the chunky outcome bucket the existing
    trade-record dict uses ("STOPPED" / "TARGET" / "TIME_EXIT" / "EOD_EXIT").
    Preserves dashboard/metrics consumers that key off this string.
    """
    if not legs:
        return default
    last = legs[-1].reason
    if last.startswith("stop") or last == "breakeven_stop":
        return "STOPPED"
    if last == "trail_stop":
        return "TRAIL_EXIT"
    if last == "target" or last.startswith("scale_"):
        return "TARGET"
    if last == "max_hold_bars":
        return "TIME_EXIT"
    if last == "eod" or last.startswith("hard_exit_"):
        return "EOD_EXIT"
    return default


# ── Aggregation helper for callers ──────────────────────────────────────────


def aggregate_pnl(
    direction: str,
    entry_price: float,
    sim: SimulatedExit,
) -> dict[str, float]:
    """
    Roll up SimulatedExit legs into the dict shape the backtester's trade
    record already uses: exit_price (qty-weighted), realized $ P&L (gross, on
    leg fills only), and total qty closed.

    Commission/slippage is applied by the caller (FillSimulator) so this
    helper deliberately returns the pre-cost numbers.
    """
    closed_qty = sim.closed_qty
    if closed_qty <= 0:
        return {"exit_price": entry_price, "gross_pnl": 0.0, "qty": 0.0}

    exit_vwap = sim.vwap_exit_price
    sign = 1.0 if direction == "BUY" else -1.0
    gross_pnl = sum(
        (leg.price - entry_price) * leg.qty * sign for leg in sim.legs
    )
    return {
        "exit_price": exit_vwap,
        "gross_pnl": gross_pnl,
        "qty": closed_qty,
    }
