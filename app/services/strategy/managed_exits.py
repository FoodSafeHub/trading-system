from __future__ import annotations

"""
Stateful trade-management overlay ("managed exits").

Designed for CEEI (long-entry ignition engine) but entry-agnostic: given an
OHLCV frame and an entry bar, it walks the trade forward bar-by-bar applying a
configurable stack of exit rules and records exactly which rule caused every
stop move, partial fill, and final exit.

Rules (all optional, coexisting):
  - initial stop: ATR-multiple below entry and/or structure stop under the most
    recent swing low (when both are enabled the HIGHER stop — more conservative
    for a long — wins)
  - breakeven move after a configurable R multiple
  - partial take-profit at a configurable R multiple / fraction
  - trailing stop: ATR, structure (recent swing low), chandelier
    (highest-high - mult*ATR), or "tightest" = max of the enabled trails
  - optional full profit target at an R multiple
  - time-stop fallback
  - optional mirrored SELL-signal exit (config A / legacy behaviour)

Anti-lookahead contract: the stop tested against bar i was fixed using data
through bar i-1. Stop raises earned on bar i (breakeven, trail ratchets) apply
from bar i+1. When a bar's range spans both the stop and a target, the STOP is
assumed to fill first (conservative).

This module is pure computation + logging — it does not place orders. It is
additive: the stateless signal-level overlay in ``exits.py`` is untouched.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ManagedExitConfig:
    # ── Initial stop ──
    initial_atr_mult: Optional[float] = 1.5   # entry - mult*ATR(14); None disables
    structure_initial: bool = False           # swing-low initial stop
    swing_lookback: int = 10                  # bars for "most recent swing low"
    structure_buffer_atr: float = 0.25        # stop sits this far below the swing low
    # ── Breakeven ──
    breakeven_at_r: Optional[float] = 1.0     # move stop to entry after +NR; None disables
    # ── Partial take-profit ──
    partial_at_r: Optional[float] = 1.5       # take partial at +NR; None disables
    partial_fraction: float = 0.5             # fraction of position closed at the partial
    # ── Trailing ──
    trail: str = "none"                       # "none"|"atr"|"structure"|"chandelier"|"tightest"
    trail_atr_mult: float = 2.5               # ATR trail: highest close - mult*ATR
    chandelier_mult: float = 3.0              # chandelier: highest high - mult*ATR
    # ── Full exits ──
    profit_target_r: Optional[float] = None   # optional full target at +NR
    time_stop_bars: int = 40                  # fallback time stop
    use_sell_signal: bool = False             # mirrored SELL-signal exit (legacy config A)
    atr_period: int = 14


@dataclass
class ManagedTrade:
    symbol: str
    entry_date: str
    entry: float
    initial_stop: float
    r_unit: float                 # entry - initial_stop (1R in price terms)
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""         # rule that closed the remainder
    partial_date: Optional[str] = None
    partial_price: Optional[float] = None
    partial_r: Optional[float] = None
    final_stop: float = 0.0
    stop_moves: int = 0
    hold_bars: int = 0
    realized_pct: float = 0.0     # blended over partial + final legs
    realized_r: float = 0.0       # blended R-multiple
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    mfe_captured_pct: Optional[float] = None  # realized_pct / mfe_pct
    events: List[str] = field(default_factory=list)


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def manage_trade(
    symbol: str,
    df: pd.DataFrame,
    entry_i: int,
    config: ManagedExitConfig,
    sell_signal: pd.Series | None = None,
    log_context: str = "",
) -> Optional[ManagedTrade]:
    """Simulate one long trade entered at the OPEN of bar ``entry_i``.
    Returns None when the entry bar is out of range or no stop can be formed."""
    n = len(df)
    if entry_i >= n:
        return None
    c = config
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    atr = _atr_series(df, c.atr_period).to_numpy(dtype=float)

    entry = opens[entry_i]
    atr0 = atr[entry_i - 1] if entry_i > 0 and not np.isnan(atr[entry_i - 1]) else entry * 0.02

    # ── Initial stop: most conservative (highest) of the enabled candidates ──
    candidates = []
    if c.initial_atr_mult is not None:
        candidates.append(("initial_atr", entry - c.initial_atr_mult * atr0))
    if c.structure_initial and entry_i >= c.swing_lookback:
        swing_low = float(np.min(lows[entry_i - c.swing_lookback:entry_i]))
        candidates.append(("initial_structure", swing_low - c.structure_buffer_atr * atr0))
    if not candidates:
        candidates.append(("initial_atr_default", entry - 1.5 * atr0))
    stop_source, stop = max(candidates, key=lambda kv: kv[1])
    if stop >= entry:  # degenerate (e.g. swing low above entry) — fall back to ATR
        stop_source, stop = "initial_atr", entry - (c.initial_atr_mult or 1.5) * atr0

    r_unit = entry - stop
    if r_unit <= 0:
        return None

    trade = ManagedTrade(
        symbol=symbol, entry_date=str(df.index[entry_i])[:10],
        entry=round(entry, 4), initial_stop=round(stop, 4), r_unit=round(r_unit, 4),
    )
    ctx = f"[{log_context or symbol}] "

    partial_done = c.partial_at_r is None
    breakeven_done = c.breakeven_at_r is None
    remaining = 1.0
    highest_close = entry
    highest_high = entry
    mfe = 0.0
    mae = 0.0

    def _log(msg: str) -> None:
        trade.events.append(msg)
        logger.info("ManagedExit %s%s", ctx, msg)

    def _blend(final_px: float) -> None:
        legs = []
        if trade.partial_price is not None:
            legs.append((c.partial_fraction, trade.partial_price))
        legs.append((remaining, final_px))
        total = sum(f for f, _ in legs)
        avg_exit = sum(f * px for f, px in legs) / total
        trade.realized_pct = round((avg_exit - entry) / entry * 100, 3)
        trade.realized_r = round((avg_exit - entry) / r_unit, 3)

    def _close(i: int, px: float, reason: str) -> ManagedTrade:
        trade.exit_date = str(df.index[i])[:10]
        trade.exit_price = round(px, 4)
        trade.exit_reason = reason
        trade.final_stop = round(stop, 4)
        trade.hold_bars = i - entry_i
        trade.mfe_pct = round(mfe * 100, 3)
        trade.mae_pct = round(mae * 100, 3)
        _blend(px)
        if trade.mfe_pct > 0:
            trade.mfe_captured_pct = round(trade.realized_pct / trade.mfe_pct * 100, 1)
        _log(f"EXIT {reason} @ {px:.2f} ({trade.realized_r:+.2f}R, hold {trade.hold_bars} bars)")
        return trade

    for i in range(entry_i, n):
        o, h, lo = opens[i], highs[i], lows[i]
        bars_held = i - entry_i

        # ── Time stop (at this bar's open) ──
        if bars_held >= c.time_stop_bars:
            return _close(i, o, "time_stop")

        # ── Mirrored SELL-signal exit (fills next open; legacy behaviour) ──
        if (c.use_sell_signal and sell_signal is not None and i > entry_i
                and sell_signal.iloc[i - 1] == "SELL"):
            return _close(i, o, "sell_signal")

        # ── Stop test (gap-aware; stop was fixed from data through bar i-1) ──
        if o <= stop:
            return _close(i, o, f"stop_gap:{stop_source}")
        if lo <= stop:
            return _close(i, stop, f"stop:{stop_source}")

        # ── Excursions ──
        mfe = max(mfe, (h - entry) / entry)
        mae = min(mae, (lo - entry) / entry)

        # ── Full profit target ──
        if c.profit_target_r is not None:
            tgt = entry + c.profit_target_r * r_unit
            if h >= tgt:
                return _close(i, max(tgt, o), "profit_target")

        # ── Partial take-profit ──
        if not partial_done:
            level = entry + c.partial_at_r * r_unit
            if h >= level:
                px = max(level, o)   # gap above the level fills at the open
                trade.partial_date = str(df.index[i])[:10]
                trade.partial_price = round(px, 4)
                trade.partial_r = round((px - entry) / r_unit, 3)
                remaining = 1.0 - c.partial_fraction
                partial_done = True
                _log(f"PARTIAL {c.partial_fraction:.0%} @ {px:.2f} "
                     f"(+{trade.partial_r:.2f}R, level {c.partial_at_r}R)")

        # ── Stop raises earned on this bar (apply from the NEXT bar) ──
        if not breakeven_done and h >= entry + c.breakeven_at_r * r_unit:
            if entry > stop:
                stop, stop_source = entry, "breakeven"
                trade.stop_moves += 1
                _log(f"BREAKEVEN stop -> {entry:.2f} (hit +{c.breakeven_at_r}R)")
            breakeven_done = True

        highest_close = max(highest_close, closes[i])
        highest_high = max(highest_high, h)
        atr_i = atr[i] if not np.isnan(atr[i]) else atr0

        trail_candidates: list[tuple[str, float]] = []
        if c.trail in ("atr", "tightest"):
            trail_candidates.append(("atr_trail", highest_close - c.trail_atr_mult * atr_i))
        if c.trail in ("structure", "tightest") and i + 1 >= c.swing_lookback:
            swing = float(np.min(lows[i + 1 - c.swing_lookback:i + 1]))
            trail_candidates.append(("structure_trail", swing - c.structure_buffer_atr * atr_i))
        if c.trail == "chandelier":
            trail_candidates.append(("chandelier_trail", highest_high - c.chandelier_mult * atr_i))
        if trail_candidates:
            # "tightest" = the most conservative (highest) stop for a long
            src, level = max(trail_candidates, key=lambda kv: kv[1])
            if level > stop:
                stop, stop_source = level, src
                trade.stop_moves += 1
                _log(f"TRAIL[{src}] stop -> {level:.2f}")

    return _close(n - 1, closes[-1], "end_of_data")


# ── Preset configurations used by the exit research ───────────────────────────

EXIT_PRESETS: dict[str, ManagedExitConfig] = {
    # A: legacy behaviour — mirrored SELL exit + 20-bar time stop, nothing else.
    # The 100-ATR "stop" is a never-hit placeholder so behaviour matches the
    # prior research sim (which had no stop); ignore A's R-multiples.
    "A_mirrored_sell": ManagedExitConfig(
        initial_atr_mult=100.0, breakeven_at_r=None, partial_at_r=None,
        trail="none", time_stop_bars=20, use_sell_signal=True,
    ),
    # B: ATR trail + partial + breakeven
    "B_atr_trail": ManagedExitConfig(
        initial_atr_mult=1.5, breakeven_at_r=1.0,
        partial_at_r=1.5, partial_fraction=0.5,
        trail="atr", trail_atr_mult=2.5, time_stop_bars=40,
    ),
    # C: structure stop + partial + breakeven
    "C_structure": ManagedExitConfig(
        initial_atr_mult=1.5, structure_initial=True,
        breakeven_at_r=1.0, partial_at_r=1.5, partial_fraction=0.5,
        trail="structure", time_stop_bars=40,
    ),
    # D: chandelier variant
    "D_chandelier": ManagedExitConfig(
        initial_atr_mult=1.5, breakeven_at_r=1.0,
        partial_at_r=2.0, partial_fraction=0.5,
        trail="chandelier", chandelier_mult=3.0, time_stop_bars=40,
    ),
    # E: tightest-of-ATR/structure trail (the "whichever is more conservative" stack)
    "E_tightest": ManagedExitConfig(
        initial_atr_mult=1.5, structure_initial=True,
        breakeven_at_r=1.0, partial_at_r=1.5, partial_fraction=0.5,
        trail="tightest", trail_atr_mult=2.5, time_stop_bars=40,
    ),
    # ── Wide variants ──────────────────────────────────────────────────────────
    # The first exit study showed 1.5-ATR stops sit INSIDE the entries' noise
    # band (median MAE -3.8%, p75 -6.9% vs ~3% stop) and delete the edge, which
    # accrues over 10-20 bars. These place the initial stop beyond the measured
    # p75 MAE (~3 ATR) and trail loosely so the trend has room to develop.
    "B2_atr_wide": ManagedExitConfig(
        initial_atr_mult=3.0, breakeven_at_r=1.0,
        partial_at_r=1.0, partial_fraction=0.5,     # 1R ≈ +6% ≈ mean MFE
        trail="atr", trail_atr_mult=3.0, time_stop_bars=40,
    ),
    "C2_structure_wide": ManagedExitConfig(
        initial_atr_mult=3.0, structure_initial=True,
        breakeven_at_r=1.0, partial_at_r=1.0, partial_fraction=0.5,
        trail="structure", time_stop_bars=40,
    ),
    # F: let-it-run — no partial, wide chandelier, later breakeven
    "F_runner": ManagedExitConfig(
        initial_atr_mult=3.0, breakeven_at_r=1.5, partial_at_r=None,
        trail="chandelier", chandelier_mult=4.0, time_stop_bars=60,
    ),
}
