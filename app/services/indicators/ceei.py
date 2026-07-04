from __future__ import annotations

"""
Compression Expansion Efficiency Indicator (CEEI).

Swing-trading ignition detector: finds the transition from low-volatility
compression (stored energy) into directional expansion with strong
participation (volume) and low noise (efficiency). Not a trailing-stop /
trend-line indicator — it scores the *regime transition* itself.

Three components, each normalized to 0-100:

  1. Compression — how abnormally tight volatility is right now, from the
     rolling percentile rank of ATR, high-low range, and Bollinger bandwidth
     (inverted: tighter = higher score).
  2. Expansion  — how forcefully the current bar breaks out, from true-range
     thrust vs ATR(20), close location within the bar, close vs the recent
     breakout level, and relative volume. Direction (+1 up / -1 down) is
     tracked separately so the score stays 0-100.
  3. Efficiency — Kaufman-style directional efficiency ratio over a short
     window: |net move| / sum(|bar moves|). High = directional, low = chop.

Composite: CEEI = w_c*compression + w_e*expansion + w_f*efficiency
(default 35/40/25, configurable, renormalized to sum to 1).

States:
  setup_state   — instrument is coiling: compression currently (or within the
                  setup window) above the setup threshold.
  trigger_state — the coil is firing: recent setup + expansion crossing its
                  threshold with efficiency confirmation.

Signals:
  BUY  — recent setup, expansion crosses above its threshold pointing UP,
         efficiency confirms, close breaks above the prior N-bar high, and
         the CEEI score is above the buy threshold.
  SELL — mirror conditions to the downside.
  HOLD — otherwise.

All intermediate series are returned per bar for plotting/backtests.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CEEIParams:
    # ── Compression ──
    vol_lookback: int = 30          # percentile window for ATR/range/BB width (20-40)
    atr_period: int = 14
    bb_period: int = 20
    # ── Expansion ──
    exp_atr_period: int = 20        # average TR for thrust ratio
    breakout_lookback: int = 15     # prior N-bar high/low breakout level (10-20)
    rel_volume_period: int = 20
    thrust_floor: float = 0.8       # TR/ATR mapped 0->1 between floor..ceiling
    thrust_ceiling: float = 2.5
    relvol_floor: float = 0.8
    relvol_ceiling: float = 2.0
    # ── Efficiency ──
    efficiency_period: int = 6      # directional efficiency window (5-8)
    # ── Composite weights (renormalized to sum 1) ──
    w_compression: float = 0.35
    w_expansion: float = 0.40
    w_efficiency: float = 0.25
    # ── Signal thresholds ──
    # Defaults calibrated on 5y AAPL/MSFT/NVDA/SPY/QQQ daily score distributions:
    # setup 70 ≈ p80 of compression, expansion 45 ≈ p85-88 of expansion,
    # trigger 48 ≈ p80 of the composite. See scripts/backtest_ceei.py.
    setup_threshold: float = 70.0   # compression score that marks a coil
    setup_window: int = 10          # bars a setup stays "recent"
    expansion_threshold: float = 45.0  # expansion cross level for the trigger
    efficiency_min: float = 55.0    # efficiency confirmation floor
    buy_threshold: float = 48.0     # CEEI score cross for BUY
    sell_threshold: float = 48.0    # CEEI score cross for SELL (mirror side)


@dataclass
class CEEIResult:
    compression_score: pd.Series   # 0-100, high = abnormally tight
    expansion_score: pd.Series     # 0-100, high = forceful breakout bar
    expansion_direction: pd.Series # +1 up / -1 down / 0 neutral
    efficiency_score: pd.Series    # 0-100, high = directional (low noise)
    efficiency_direction: pd.Series  # sign of the net move over the ER window
    ceei_score: pd.Series          # weighted composite, 0-100
    setup_state: pd.Series         # bool — coiling (recent compression)
    trigger_state: pd.Series       # bool — coil firing (setup + expansion cross)
    breakout_level: pd.Series      # prior N-bar high (long-side trigger level)
    breakdown_level: pd.Series     # prior N-bar low (short-side trigger level)
    signal: pd.Series              # "BUY" / "SELL" / "HOLD" per bar
    gates: pd.DataFrame = None     # per-bar booleans for each signal subcondition
    params: CEEIParams = field(default_factory=CEEIParams)

    @property
    def latest_score(self) -> Optional[float]:
        valid = self.ceei_score.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None


def _pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0-100) of the latest value within its window."""
    return series.rolling(window, min_periods=max(5, window // 3)).rank(pct=True) * 100.0


def _unit(series: pd.Series, floor: float, ceiling: float) -> pd.Series:
    """Linear map floor..ceiling -> 0..1, clipped."""
    return ((series - floor) / (ceiling - floor)).clip(0.0, 1.0)


def compute_ceei(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
    params: CEEIParams | None = None,
    log_context: str = "",
) -> CEEIResult:
    """Full CEEI computation. Volume is optional — without it the relative-volume
    expansion input drops out (remaining inputs are reweighted).

    `log_context` (e.g. the symbol) is prefixed to signal log lines.
    """
    p = params or CEEIParams()
    idx = close.index

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(p.atr_period).mean()

    # ── 1. Compression: inverted percentile of three tightness measures ──────
    bar_range = high - low
    bb_width = close.rolling(p.bb_period).std() / close.rolling(p.bb_period).mean().replace(0, np.nan)
    compression = 100.0 - pd.concat([
        _pct_rank(atr, p.vol_lookback),
        _pct_rank(bar_range, p.vol_lookback),
        _pct_rank(bb_width, p.vol_lookback),
    ], axis=1).mean(axis=1)

    # ── 2. Expansion: thrust + close location + breakout distance + volume ───
    exp_atr = tr.rolling(p.exp_atr_period).mean()
    thrust = _unit(tr / exp_atr.replace(0, np.nan), p.thrust_floor, p.thrust_ceiling)

    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng          # -1 (close@low) .. +1 (close@high)
    clv = clv.fillna(0.0)

    # Prior N-bar extremes EXCLUDING the current bar (true breakout reference)
    breakout_level = high.shift(1).rolling(p.breakout_lookback).max()
    breakdown_level = low.shift(1).rolling(p.breakout_lookback).min()
    # Distance beyond the level in ATRs, mapped to 0..1 per side
    brk_up = _unit((close - breakout_level) / atr.replace(0, np.nan), 0.0, 1.0)
    brk_dn = _unit((breakdown_level - close) / atr.replace(0, np.nan), 0.0, 1.0)

    has_volume = (
        volume is not None
        and volume.notna().any()
        and float(volume.fillna(0).abs().sum()) > 0
    )
    if has_volume:
        vol_sma = volume.rolling(p.rel_volume_period).mean()
        rel_vol = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)
        relvol_u = _unit(rel_vol, p.relvol_floor, p.relvol_ceiling)
    else:
        relvol_u = None

    # Per-side close-location: for longs a close near the high scores, shorts mirror
    clv_up = ((clv + 1.0) / 2.0).clip(0.0, 1.0)
    clv_dn = ((1.0 - clv) / 2.0).clip(0.0, 1.0)

    def _side_score(clv_side: pd.Series, brk_side: pd.Series) -> pd.Series:
        parts = [thrust, clv_side, brk_side]
        weights = [0.30, 0.25, 0.25]
        if relvol_u is not None:
            parts.append(relvol_u)
            weights.append(0.20)
        w = np.array(weights) / sum(weights)
        return sum(part * wi for part, wi in zip(parts, w)) * 100.0

    exp_up = _side_score(clv_up, brk_up)
    exp_dn = _side_score(clv_dn, brk_dn)
    expansion = pd.concat([exp_up, exp_dn], axis=1).max(axis=1)
    expansion_direction = pd.Series(
        np.where(exp_up >= exp_dn, 1, -1), index=idx
    ).where(expansion.notna(), 0).astype(int)

    # ── 3. Efficiency: Kaufman ER over a short window ─────────────────────────
    n = p.efficiency_period
    net_move = close - close.shift(n)
    path = close.diff().abs().rolling(n).sum()
    er = (net_move.abs() / path.replace(0, np.nan)).clip(0.0, 1.0)
    efficiency = (er * 100.0).where(path.notna())
    efficiency_direction = pd.Series(
        np.sign(net_move).fillna(0), index=idx
    ).astype(int)

    # ── Composite ─────────────────────────────────────────────────────────────
    w_sum = p.w_compression + p.w_expansion + p.w_efficiency
    ceei = (
        compression * (p.w_compression / w_sum)
        + expansion * (p.w_expansion / w_sum)
        + efficiency * (p.w_efficiency / w_sum)
    )

    # ── States ────────────────────────────────────────────────────────────────
    setup_now = compression >= p.setup_threshold
    # A setup stays "recent" for setup_window bars after it was seen
    setup_recent = setup_now.rolling(p.setup_window, min_periods=1).max().astype(bool)
    setup_state = setup_now.fillna(False)

    exp_prev = expansion.shift(1)
    expansion_cross = (exp_prev <= p.expansion_threshold) & (expansion > p.expansion_threshold)
    eff_ok = efficiency >= p.efficiency_min
    trigger_state = (setup_recent & expansion_cross & eff_ok).fillna(False)

    # ── Signals ───────────────────────────────────────────────────────────────
    # The expansion cross is the one-shot trigger event; the composite acts as a
    # level confirmation (requiring a same-bar composite cross too would make the
    # signal degenerate — the composite often crosses a bar before expansion does).
    buy = (
        setup_recent
        & expansion_cross
        & (expansion_direction > 0)
        & eff_ok & (efficiency_direction > 0)
        & (close > breakout_level)
        & (ceei > p.buy_threshold)
    )
    sell = (
        setup_recent
        & expansion_cross
        & (expansion_direction < 0)
        & eff_ok & (efficiency_direction < 0)
        & (close < breakdown_level)
        & (ceei > p.sell_threshold)
    )
    signal = pd.Series("HOLD", index=idx)
    signal[buy.fillna(False)] = "BUY"
    signal[sell.fillna(False)] = "SELL"

    # Per-bar subcondition booleans, kept for audit logging and research
    gates = pd.DataFrame({
        "setup_recent": setup_recent.fillna(False),
        "expansion_cross": expansion_cross.fillna(False),
        "expansion_up": (expansion_direction > 0),
        "efficiency_ok": eff_ok.fillna(False),
        "efficiency_up": (efficiency_direction > 0),
        "close_above_breakout": (close > breakout_level).fillna(False),
        "close_below_breakdown": (close < breakdown_level).fillna(False),
        "ceei_above_buy": (ceei > p.buy_threshold).fillna(False),
        "ceei_above_sell": (ceei > p.sell_threshold).fillna(False),
    }, index=idx)

    # ── Audit logging for the latest bar ─────────────────────────────────────
    if len(idx) and signal.iloc[-1] != "HOLD":
        i = -1
        gate_str = " ".join(f"{k}={bool(v)}" for k, v in gates.iloc[i].items())
        logger.info(
            "CEEI %s fired%s: compression=%.1f (setup within %d bars), "
            "expansion=%.1f crossed %.1f (dir=%+d), efficiency=%.1f (min %.1f), "
            "close=%.2f vs %s=%.2f, CEEI=%.1f > %.1f | gates: %s",
            signal.iloc[i],
            f" [{log_context}]" if log_context else "",
            float(compression.iloc[i]), p.setup_window,
            float(expansion.iloc[i]), p.expansion_threshold, int(expansion_direction.iloc[i]),
            float(efficiency.iloc[i]), p.efficiency_min,
            float(close.iloc[i]),
            "breakout_level" if signal.iloc[i] == "BUY" else "breakdown_level",
            float(breakout_level.iloc[i]) if signal.iloc[i] == "BUY" else float(breakdown_level.iloc[i]),
            float(ceei.iloc[i]),
            p.buy_threshold if signal.iloc[i] == "BUY" else p.sell_threshold,
            gate_str,
        )

    return CEEIResult(
        compression_score=compression,
        expansion_score=expansion,
        expansion_direction=expansion_direction,
        efficiency_score=efficiency,
        efficiency_direction=efficiency_direction,
        ceei_score=ceei,
        setup_state=setup_state,
        trigger_state=trigger_state,
        breakout_level=breakout_level,
        breakdown_level=breakdown_level,
        signal=signal,
        gates=gates,
        params=p,
    )
