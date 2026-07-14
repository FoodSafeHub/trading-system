from __future__ import annotations

"""
Per-symbol calibrated-parameter store for the 5 scanner strategies.

Unlike the Perplexity engine (which has dedicated filter_* config keys and a
SymbolFilterProfile in symbol_profiles.py), the scanner strategies in rules.py
expose their entry gates *as ordinary params* (rsi_min/rsi_max, wick_ratio_min,
price_ema_proximity_pct, atr_skip_threshold, vol_ratio_min, …). So calibration
here means tightening those existing params per-symbol — NOT adding new rule
logic. The rule code is never touched; only the params dict it receives changes.

Profiles persist to a JSON file so they survive restarts. `_make_generic_configs`
in scanner_service.py merges the saved overrides at config-build time, which means
the scheduler, the live scanner, and the Backtest page all read the same
calibrated params with no extra wiring.
"""

import json
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "scanner_profiles.json")


@dataclass
class ScannerParamProfile:
    symbol: str
    strategy_type: str          # rsi2_mean_reversion | ema_macd_crossover | ...
    calibrated_at: str          # ISO date
    n_trades: int               # round-trips used to calibrate
    n_wins: int
    win_rate_pct: float
    # The param overrides to merge onto the factory defaults at config-build time.
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    # Evidence + audit
    survival_rate_pct: float = 0.0
    improved_by: List[str] = field(default_factory=list)
    notes: str = ""


# ── Storage ───────────────────────────────────────────────────────────────────

def _load_all() -> Dict[str, dict]:
    if not os.path.exists(_PROFILE_PATH):
        return {}
    try:
        with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: Dict[str, dict]) -> None:
    with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _key(strategy_type: str, symbol: str) -> str:
    return f"{strategy_type}:{symbol.upper()}"


def save_profile(profile: ScannerParamProfile) -> None:
    data = _load_all()
    data[_key(profile.strategy_type, profile.symbol)] = asdict(profile)
    _save_all(data)


def load_profile(strategy_type: str, symbol: str) -> Optional[ScannerParamProfile]:
    raw = _load_all().get(_key(strategy_type, symbol.upper()))
    if raw is None:
        return None
    try:
        return ScannerParamProfile(**raw)
    except Exception:
        return None


def list_profiles(strategy_type: Optional[str] = None) -> List[ScannerParamProfile]:
    out: List[ScannerParamProfile] = []
    for key, raw in _load_all().items():
        if strategy_type and not key.startswith(f"{strategy_type}:"):
            continue
        try:
            out.append(ScannerParamProfile(**raw))
        except Exception:
            pass
    return sorted(out, key=lambda p: (p.strategy_type, p.symbol))


def delete_profile(strategy_type: str, symbol: str) -> bool:
    data = _load_all()
    k = _key(strategy_type, symbol)
    if k in data:
        del data[k]
        _save_all(data)
        return True
    return False


# Param keys that are LIVE trail scaffolding, not calibration data. A profile
# whose only overrides are these (and which found no calibration trades) is a
# trail-enabler (e.g. the live COP/TOST/ARLO/AVPT Chandelier configs), not a
# genuine grid-search calibration — migration must skip these so it never copies
# live trail scaffolding under the new strategy names.
_TRAIL_SCAFFOLD_KEYS = {"trail_enabled", "trail_trigger_pct", "atr_trail_mult", "atr_trail_period", "stop_loss_pct"}


def _is_genuine_calibration(raw: dict) -> bool:
    """True when a saved profile represents real calibration data.

    Genuine = it produced calibration round-trips (n_trades > 0) OR it carries at
    least one entry-param override beyond the trail-scaffolding keys. A 0-trade,
    trail-only profile returns False (it is live trail config, not calibration).
    """
    overrides = raw.get("param_overrides") or {}
    entry_overrides = [k for k in overrides if k not in _TRAIL_SCAFFOLD_KEYS]
    return int(raw.get("n_trades") or 0) > 0 or bool(entry_overrides)


def migrate_aliased_profiles(dry_run: bool = True, overwrite: bool = False) -> List[dict]:
    """Materialize NEW-name profiles from their predecessors' GENUINE calibration.

    NON-DESTRUCTIVE: old profiles are never deleted or modified — this only adds
    a copy under ``<new_type>:SYMBOL`` (taking the PRIMARY predecessor that has a
    genuine calibration profile). Rollback = delete the new-name copy; the old one
    still drives the legacy strategy and the alias fallback. ``dry_run=True``
    (default) just reports what WOULD be copied. ``overwrite=False`` never
    clobbers an existing new-name profile.

    Trail-only / 0-trade profiles (live Chandelier scaffolding, e.g. COP/TOST/
    ARLO/AVPT) are SKIPPED and reported as such — they are live config, not
    calibration, and migrating them would copy inert trail params under the new
    names. They remain untouched in place.

    Strictly optional: ``get_param_overrides`` already inherits old calibration
    for new names via the alias map, so the new strategies work WITHOUT migrating.
    """
    from app.services.strategy.strategy_aliases import STRATEGY_ALIASES

    data = _load_all()
    report: List[dict] = []
    changed = False
    for new_type, olds in STRATEGY_ALIASES.items():
        if not olds:
            continue
        syms = {key.partition(":")[2] for key in data
                if key.partition(":")[0] in olds and key.partition(":")[2]}
        for sym in sorted(syms):
            new_key = _key(new_type, sym)
            if new_key in data and not overwrite:
                report.append({"new_type": new_type, "symbol": sym,
                               "source": None, "action": "skip (exists)"})
                continue
            # Only consider predecessors that are GENUINE calibration profiles.
            src = next((o for o in olds
                        if _key(o, sym) in data and _is_genuine_calibration(data[_key(o, sym)])),
                       None)
            if src is None:
                # A predecessor exists but it is trail-only/empty -> skip (and
                # surface it for transparency). Live trail config stays in place.
                trail_src = next((o for o in olds if _key(o, sym) in data), None)
                if trail_src is not None:
                    report.append({"new_type": new_type, "symbol": sym,
                                   "source": trail_src, "action": "skip (trail-only/no calibration)"})
                continue
            report.append({"new_type": new_type, "symbol": sym,
                           "source": src, "action": "copy"})
            if not dry_run:
                src_raw = dict(data[_key(src, sym)])
                src_raw["strategy_type"] = new_type
                src_raw["notes"] = f"migrated from {src} (alias); " + str(src_raw.get("notes", ""))
                data[new_key] = src_raw
                changed = True
    if not dry_run and changed:
        _save_all(data)
    return report


def get_param_overrides(strategy_type: str, symbol: str) -> Dict[str, Any]:
    """Runtime lookup used by _make_generic_configs. Empty dict = no calibration.

    The self key is tried FIRST, so every existing (live) strategy type returns
    exactly what it always has. Only when nothing is saved under the queried
    type do we fall back to predecessor types via the alias map — that path is
    dormant in Phase 0 (no new types are queried yet) and lets Phase 1's renamed
    strategies inherit calibration saved under their old names.
    """
    from app.services.strategy.strategy_aliases import resolve_alias

    for st in resolve_alias(strategy_type):
        p = load_profile(st, symbol)
        if p and p.param_overrides:
            return dict(p.param_overrides)
    return {}


# ── Calibration: grid-search the params each rule ACTUALLY gates on ────────────
#
# Earlier this module nudged params from a winner-vs-loser indicator split. That
# was unreliable for two reasons a trader spotted: (1) the trade analyzer captures
# generic Perplexity-style indicators (EMA20 distance, RSI-14), NOT what the
# scanner rules gate on (EMA50 proximity, the reclaim wick, RSI(2), ATR%) — so it
# analyzed the wrong things; and (2) high-win-rate strategies (pullback_ema50 at
# 85–100%) have almost no losers, so "separate winners from losers" is statistically
# empty. Result: it always said "no separation" and before/after were identical.
#
# The robust fix is model-free: grid-search each strategy's REAL entry params, re-run
# the backtest for every combo, and keep the combo that maximises expectancy (with a
# trade-survival floor). The backtest itself is the oracle — no indicator proxy needed.
# Grids widen AND tighten: a 90%-win strategy improves by catching more good entries,
# a choppy one improves by tightening. The caller's save gate validates the winner
# against the factory baseline (win-rate OR expectancy OR return must materially beat it).

# Per-strategy grids over the params each rule reads in rules.py. Kept small
# (≤ a few dozen combos) so a 10y calibration stays interactive.
_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    # RSI(2) mean reversion. Entry depth + volatility skip were already tuned;
    # added the two levers a mean-reversion trader actually adjusts after that:
    # WHERE to take the bounce (rsi_exit too low caps winners, too high gives
    # them back) and HOW LONG to wait (a revert that hasn't happened in N bars
    # is a failed thesis — cut it).
    "rsi2_mean_reversion": {
        "rsi_entry_threshold": [5, 8, 10, 12, 15],    # lower = deeper dip required
        "atr_skip_threshold":  [3.0, 4.0, 5.0, 7.0],  # max ATR% regime
        "rsi_exit_threshold":  [60, 65, 70, 75, 80],  # bounce-target RSI
        "max_hold_bars":       [5, 8, 10, 15],         # time-stop the dead thesis
    },
    # EMA/MACD crossover. RSI floor + volume were tuned; added the RSI CEILING
    # (buying a crossover at RSI 70 is chasing, not entering) and the time-stop.
    "ema_macd_crossover": {
        "rsi_min":       [40, 45, 50, 55],
        "rsi_max":       [62, 65, 70, 75],
        "vol_ratio_min": [1.0, 1.1, 1.3, 1.5],
        "max_hold_bars": [10, 15, 20, 30],
    },
    # BB squeeze breakout. RSI floor + volume were tuned; added the two that
    # DEFINE the setup quality: how many bars of contracting bandwidth count as
    # a real coil, and how wide the bands (breakout strictness).
    "bb_squeeze_breakout": {
        "rsi_entry_min": [45, 50, 55, 60],
        "vol_ratio_min": [1.1, 1.3, 1.5, 2.0],
        "squeeze_bars":  [4, 5, 6, 7],
        "bb_std":        [1.8, 2.0, 2.2, 2.5],
    },
    # Pullback to EMA50. Proximity + RSI floor + reclaim wick were tuned; added
    # the RSI CEILING (a "pullback" at RSI 55 is barely a pullback) and the
    # profit-take extension distance.
    "pullback_ema50": {
        "price_ema_proximity_pct": [0.5, 1.0, 1.5, 2.0],
        "rsi_min":                 [30, 35, 40],
        "rsi_max":                 [50, 55, 60],
        "wick_ratio_min":          [0.3, 0.4, 0.5, 0.6],
        "exit_extension_pct":      [2.0, 3.0, 4.0, 5.0],
        # Downside trend-fail exit (0 = off). Per-symbol: high-beta names may
        # want 5%, sleepy dividend names 3%.
        "trend_fail_below_ema_pct": [0.0, 3.0, 4.0, 5.0],
    },
    # VIX-spike reversal. ATR spike + RSI ceiling + wick were tuned; added how
    # DEEP into the lower band the panic must reach (bb_pos_max) and how much
    # prior capitulation is required to call it a washout (prior_decline_pct).
    "vix_spike_reversal": {
        "atr_spike_threshold": [2.5, 3.0, 3.5, 4.0],
        "rsi_entry_max":       [25, 30, 35],
        "wick_ratio_min":      [0.4, 0.5, 0.6],
        "bb_pos_max":          [0.10, 0.15, 0.20, 0.25],
        "prior_decline_pct":   [1.0, 2.0, 3.0, 4.0],
    },
    # ── Unified strategies (Phase 1/2) — grids over the params each NEW rule in
    #    rules.py actually reads. Kept parallel to the legacy grids above; old
    #    names are unchanged. Calibrating a new name saves under "<new>:SYMBOL".
    "rsi2_reversion": {
        "rsi_entry_threshold": [5, 8, 10, 12, 15],
        "atr_skip_threshold":  [3.0, 4.0, 5.0, 7.0],
        "rsi_exit_threshold":  [55, 60, 65, 70, 75],
        "hard_stop_pct":       [2.0, 2.5, 3.0, 4.0],
    },
    "trend_pullback": {
        "atr_proximity_mult": [0.8, 1.0, 1.2, 1.6],
        "rsi_min":            [30, 35, 40],
        "rsi_max":            [50, 55, 60],
        "wick_ratio_min":     [0.3, 0.4, 0.5, 0.6],
        "exit_extension_pct": [2.0, 3.0, 4.0, 5.0],
    },
    "squeeze_breakout": {
        "rsi_entry_min":      [45, 50, 55, 60],
        "vol_ratio_min":      [1.1, 1.3, 1.5, 2.0],
        "squeeze_percentile": [0.30, 0.40, 0.50],
        "break_atr_frac":     [0.05, 0.10, 0.20],
        "bb_std":             [1.8, 2.0, 2.2, 2.5],
    },
    "momentum_breakout": {
        "donchian":      [15, 20, 30, 40],
        "donchian_exit": [5, 10, 15],
        "vol_ratio_min": [1.0, 1.2, 1.5],
        "stop_atr_mult": [2.0, 2.5, 3.0],
    },
    "panic_reversal": {
        "atr_spike_threshold": [2.5, 3.0, 3.5, 4.0],
        "rsi_entry_max":       [25, 30, 35],
        "wick_ratio_min":      [0.4, 0.5, 0.6],
        "bb_pos_max":          [0.10, 0.15, 0.20, 0.25],
        "prior_decline_pct":   [1.0, 2.0, 3.0, 4.0],
    },
    "trend_follow": {
        "st_multiplier": [2.0, 2.5, 3.0, 3.5],
        "adx_min":       [15.0, 20.0, 25.0, 30.0],
        "st_period":     [7, 10, 14],
    },
}


def grid_for(strategy_type: str) -> Dict[str, list]:
    return _PARAM_GRIDS.get(strategy_type, {})


def coordinate_descent_search(
    strategy_type: str,
    base_params: Dict[str, Any],
    evaluate,
    rounds: int = 2,
):
    """
    Tune a strategy's grid params one at a time (coordinate descent) — the way a
    trader optimises by hand: hold everything fixed, sweep one param, keep the
    best value, move to the next; repeat for a couple of passes so params that
    interact can re-settle. Far cheaper than a full grid (≈ sum of grid lengths
    per round vs the product) while still landing on a strong combo.

    evaluate(params) -> (score, metrics_dict). score is what we maximise
    (expectancy); the caller decides the final save gate. Returns
    (best_overrides, best_score, best_metrics, n_evals).
    """
    grid = _PARAM_GRIDS.get(strategy_type, {})
    if not grid:
        return {}, None, None, 0

    def _ck(p: Dict[str, Any]) -> tuple:
        # Cache key over SCALAR params only. Non-scalar params (e.g. a unified
        # strategy's exit_policy dict) are constant across a run and would be
        # unhashable; the grid only varies scalars, so they fully identify a trial.
        return tuple(sorted((k, v) for k, v in p.items()
                            if isinstance(v, (int, float, str, bool)) or v is None))

    cur = dict(base_params)
    best_score, best_metrics = evaluate(cur)
    n_evals = 1
    cache: Dict[tuple, Any] = {_ck(cur): (best_score, best_metrics)}

    for _ in range(max(1, rounds)):
        improved_this_round = False
        for key, values in grid.items():
            for v in values:
                if cur.get(key) == v:
                    continue
                trial = dict(cur); trial[key] = v
                # Respect RSI window ordering when both bounds are tunable/known.
                lo = trial.get("rsi_min"); hi = trial.get("rsi_max")
                if lo is not None and hi is not None and lo >= hi - 4:
                    continue
                ck = _ck(trial)
                if ck in cache:
                    score, metrics = cache[ck]
                else:
                    score, metrics = evaluate(trial)
                    cache[ck] = (score, metrics); n_evals += 1
                if score is not None and (best_score is None or score > best_score):
                    best_score, best_metrics = score, metrics
                    cur = trial
                    improved_this_round = True
        if not improved_this_round:
            break

    overrides = {k: v for k, v in cur.items() if base_params.get(k) != v}
    return overrides, best_score, best_metrics, n_evals
