from __future__ import annotations

"""
CEEI meta-layer research framework.

Couples the CEEI ignition indicator to any registered base strategy and
measures the effect. CEEI is NOT the strategy here — it is a context layer
whose value is tested per coupling mode:

  A_base           — the strategy untouched (control)
  B_setup_filter   — entry only if CEEI setup_state was active within N bars
  C_trigger_filter — entry only if CEEI trigger_state is active on the signal bar
  D_score_filter   — entry only if CEEI score >= threshold
  E_ranking        — competing same-day entries across symbols: keep top-K by score
  F_exit_assist    — base exits kept, but a 2-ATR trail arms when CEEI deteriorates
  G_veto           — entries blocked when CEEI shows hostile conditions
                     (active downside ignition / efficient downtrend)

All modes share ONE CEEI configuration (no per-strategy tuning) and one
transparent simulator (entry next-bar open after BUY; exit next-bar open after
base SELL, exit-assist trail, or a time-stop fallback) so differences are
attributable to the coupling alone.

Modular by design: `discover_strategies` supports include/exclude whitelists,
signal series are cached per (strategy, symbol) so re-runs and new modes are
cheap, and each coupling mode is a standalone mask/simulator option.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from app.services.indicators.ceei import CEEIParams, CEEIResult, compute_ceei
from app.services.strategy.rules import _RULE_REGISTRY

logger = logging.getLogger(__name__)

# Meta-layer / non-swing entries excluded from auto-discovery
DEFAULT_EXCLUDE = {"amat", "ceei"}

# Near-identical duplicate registrations in the rule registry (alias -> canonical).
# With dedupe=True discovery keeps only the canonical name so aggregate stats
# don't double-count the same behaviour.
KNOWN_DUPLICATES = {
    "rsi2_reversion": "rsi2_mean_reversion",
    "panic_reversal": "vix_spike_reversal",
}

MODES = ["A_base", "B_setup_filter", "C_trigger_filter", "D_score_filter",
         "E_ranking", "F_exit_assist", "G_veto"]


@dataclass
class CouplingConfig:
    setup_window: int = 10          # B: bars a CEEI setup stays "recent"
    score_threshold: float = 48.0   # D: minimum CEEI score at entry
    ranking_top: int = 1            # E: same-day entries kept per strategy
    assist_score: float = 40.0      # F: CEEI score below this arms the tighter trail
    assist_atr_mult: float = 2.0    # F: trail distance once armed
    veto_expansion: float = 60.0    # G: downside expansion score considered hostile
    veto_efficiency: float = 60.0   # G: downside efficiency score considered hostile
    time_stop_bars: int = 40        # simulator fallback exit
    warmup_bars: int = 260          # bars before the first evaluated signal


def discover_strategies(include: Optional[Iterable[str]] = None,
                        exclude: Optional[Iterable[str]] = None,
                        dedupe: bool = False) -> List[str]:
    """Strategy types from the live registry, minus meta/excluded ones.
    dedupe=True drops known duplicate registrations (keeps the canonical name)."""
    excl = DEFAULT_EXCLUDE | set(exclude or ())
    if dedupe:
        excl |= set(KNOWN_DUPLICATES)
    names = [s for s in _RULE_REGISTRY if s not in excl]
    if include:
        include = set(include)
        names = [s for s in names if s in include]
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Base-strategy signal series (expensive → cached)
# ──────────────────────────────────────────────────────────────────────────────

def signal_series(strategy: str, symbol: str, df: pd.DataFrame,
                  warmup: int = 260) -> pd.Series:
    """Per-bar BUY/SELL/HOLD from a registry rule, walking an expanding window
    exactly like the backtest engine (no lookahead)."""
    fn = _RULE_REGISTRY[strategy]
    out = np.array(["HOLD"] * len(df), dtype=object)
    for i in range(warmup, len(df)):
        sub = df.iloc[:i + 1]
        try:
            out[i] = fn(symbol, sub["Close"], {}, ohlcv=sub).direction
        except Exception:               # a single bad bar must not kill the run
            out[i] = "HOLD"
    return pd.Series(out, index=df.index)


def cached_signal_series(strategy: str, symbol: str, df: pd.DataFrame,
                         cache_dir: Path, warmup: int = 260) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / f"{strategy}__{symbol}__{len(df)}.csv"
    if key.exists():
        cached = pd.read_csv(key, index_col=0)
        if len(cached) == len(df):
            return pd.Series(cached["signal"].to_numpy(), index=df.index)
    sig = signal_series(strategy, symbol, df, warmup)
    pd.DataFrame({"signal": sig.to_numpy()}, index=df.index.astype(str)).to_csv(key)
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# Coupling masks
# ──────────────────────────────────────────────────────────────────────────────

def entry_mask(mode: str, ceei: CEEIResult, cfg: CouplingConfig) -> pd.Series:
    """Per-bar boolean: is a base-strategy BUY allowed on this bar?"""
    if mode in ("A_base", "E_ranking", "F_exit_assist"):
        return pd.Series(True, index=ceei.signal.index)
    if mode == "B_setup_filter":
        return ceei.setup_state.rolling(cfg.setup_window, min_periods=1).max().astype(bool)
    if mode == "C_trigger_filter":
        return ceei.trigger_state.astype(bool)
    if mode == "D_score_filter":
        return (ceei.ceei_score >= cfg.score_threshold).fillna(False)
    if mode == "G_veto":
        hostile = (
            ((ceei.expansion_score > cfg.veto_expansion) & (ceei.expansion_direction < 0))
            | ((ceei.efficiency_score > cfg.veto_efficiency) & (ceei.efficiency_direction < 0))
        )
        return ~hostile.fillna(False)
    raise ValueError(f"unknown mode {mode!r}")


def ranking_selection(candidates: Dict[str, List[int]],
                      dates: Dict[str, pd.Index],
                      scores: Dict[str, pd.Series],
                      top: int = 1) -> Dict[str, set]:
    """E_ranking: among same-day candidate entries across symbols, keep the
    top-K by CEEI score. Returns {symbol: allowed signal-bar indices}."""
    by_date: Dict[str, list] = {}
    for sym, idxs in candidates.items():
        for i in idxs:
            d = str(dates[sym][i])[:10]
            s = scores[sym].iloc[i]
            by_date.setdefault(d, []).append((float(s) if not pd.isna(s) else -1.0, sym, i))
    allowed: Dict[str, set] = {sym: set() for sym in candidates}
    for d, lst in by_date.items():
        for _, sym, i in sorted(lst, reverse=True)[:top]:
            allowed[sym].add(i)
    return allowed


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

def simulate(symbol: str, df: pd.DataFrame, base_sig: pd.Series,
             mask: pd.Series, ceei: CEEIResult, cfg: CouplingConfig,
             exit_assist: bool = False,
             allowed_bars: Optional[set] = None) -> List[dict]:
    """Long-only walk: entry next open after an allowed BUY; exit next open
    after a base SELL, on the exit-assist trail, or at the time stop."""
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy(dtype=float)
    sig = base_sig.to_numpy()
    mask_v = mask.to_numpy()
    score = ceei.ceei_score.to_numpy(dtype=float)
    n = len(df)

    trades: List[dict] = []
    t = 0
    while t < n - 1:
        if sig[t] != "BUY" or not mask_v[t] or (allowed_bars is not None and t not in allowed_bars):
            t += 1
            continue
        e = t + 1
        entry = opens[e]
        exit_i = exit_px = None
        reason = ""
        assist_armed = False
        trail = -np.inf
        highest_close = entry
        for u in range(e, n):
            if u - e >= cfg.time_stop_bars:
                exit_i, exit_px, reason = u, opens[u], "time_stop"
                break
            if u > e and sig[u - 1] == "SELL":
                exit_i, exit_px, reason = u, opens[u], "base_sell"
                break
            if exit_assist and assist_armed:
                if opens[u] <= trail:
                    exit_i, exit_px, reason = u, opens[u], "assist_trail_gap"
                    break
                if lows[u] <= trail:
                    exit_i, exit_px, reason = u, trail, "assist_trail"
                    break
            highest_close = max(highest_close, closes[u])
            if exit_assist:
                a = atr[u] if not np.isnan(atr[u]) else entry * 0.02
                if not assist_armed and not np.isnan(score[u]) and score[u] < cfg.assist_score:
                    assist_armed = True
                if assist_armed:
                    trail = max(trail, highest_close - cfg.assist_atr_mult * a)
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, closes[-1], "end_of_data"
        trades.append({
            "symbol": symbol,
            "signal_date": str(df.index[t])[:10],
            "entry_date": str(df.index[e])[:10],
            "exit_date": str(df.index[exit_i])[:10],
            "ret_pct": round((exit_px - entry) / entry * 100, 3),
            "hold_bars": exit_i - e,
            "exit_reason": reason,
            "ceei_score_at_entry": round(float(score[t]), 1) if not np.isnan(score[t]) else np.nan,
        })
        t = exit_i + 1
    return trades


# ──────────────────────────────────────────────────────────────────────────────
# Metrics (self-contained so the module has no scripts/ dependency)
# ──────────────────────────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame, years: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    eq = (1 + r / 100).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax() * 100).max()
    tpy = len(r) / years if years > 0 else np.nan
    sharpe = (r.mean() / r.std() * np.sqrt(tpy)) if len(r) > 2 and r.std() > 0 else np.nan
    return {
        "trades": len(r),
        "win_rate": round(len(wins) / len(r) * 100, 1),
        "expectancy": round(r.mean(), 3),
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if losses.sum() != 0 else np.inf,
        "sharpe": round(sharpe, 2) if not pd.isna(sharpe) else np.nan,
        "avg_win": round(wins.mean(), 2) if len(wins) else np.nan,
        "avg_loss": round(losses.mean(), 2) if len(losses) else np.nan,
        "max_dd_pct": round(dd, 2),
        "avg_hold_bars": round(trades["hold_bars"].mean(), 1),
    }


def compute_ceei_for(df: pd.DataFrame, params: CEEIParams | None = None) -> CEEIResult:
    return compute_ceei(df["High"], df["Low"], df["Close"], df.get("Volume"),
                        params=params or CEEIParams())
