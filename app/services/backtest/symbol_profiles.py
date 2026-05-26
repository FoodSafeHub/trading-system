from __future__ import annotations

"""
Per-symbol filter profile store for EMA_Mean_Reversion (and extensible to other
strategies).

Profiles are persisted to a JSON file so they survive server restarts.
The strategy reads the active profile for a symbol at BUY-signal time.
"""

import json
import os
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "symbol_profiles.json")


@dataclass
class SymbolFilterProfile:
    symbol: str
    strategy: str
    calibrated_at: str          # ISO date string
    n_trades: int               # trades used to calibrate
    n_wins: int
    win_rate_pct: float

    # Filter thresholds (0.0 = disabled)
    vol_min: float = 0.0          # all strategies: per-symbol volume ratio minimum

    # EMA_Mean_Reversion
    ema_dist_min: float = 0.0     # min % distance from EMA20
    bb_pos_min: float = 0.0       # min BB position (0=lower band, 1=upper band)

    # MA_Crossover_RSI
    ema_spread_min: float = 0.0   # min fast/slow EMA spread % (wider = stronger momentum)

    # BB_Mean_Reversion — use actual top discriminators (RSI, bb_pct, ema_dist, atr)
    atr_pct_max: float = 0.0      # max ATR% of price (avoid volatility spikes)
    bb_depth_min: float = 0.0     # min depth below BB lower band as % of band width
    rsi_min: float = 0.0          # min RSI at entry (winners have higher RSI = recovery already starting)
    bb_pct_min: float = 0.0       # min BB% position at entry (0=lower, 1=upper)
    ema_dist_pct_min: float = 0.0 # min EMA20 dist% (positive = above EMA, less negative = less extended)

    # Breakout_Consolidation — reuses atr_pct_max field as range_atr_max (range/ATR ratio cap)

    # Fib_Pullback_Support
    lower_wick_min: float = 0.0   # min lower wick % of bar range (quality of rejection candle)
    body_max: float = 0.0         # max candle body as % of ATR (small body = indecision/rejection → wins)

    # RSI_Swing_Reversal — dedicated field (rsi_min above is used by BB_Mean_Reversion)
    swing_rsi_min: float = 0.0    # min RSI(14) at entry for the swing-reversal pullback

    # Evidence means (filled with available indicators)
    win_ema_dist_mean: float = 0.0
    loss_ema_dist_mean: float = 0.0
    win_vol_mean: float = 0.0
    loss_vol_mean: float = 0.0
    win_bb_pos_mean: float = 0.0
    loss_bb_pos_mean: float = 0.0

    # Walk-forward verification
    wfe_before: Optional[float] = None
    wfe_after: Optional[float] = None
    oos_cagr_before: Optional[float] = None
    oos_cagr_after: Optional[float] = None
    verified: bool = False      # True if WFE improved after applying filters


# ── Storage ───────────────────────────────────────────────────────────────────

def _load_all() -> Dict[str, dict]:
    if not os.path.exists(_PROFILE_PATH):
        return {}
    try:
        with open(_PROFILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: Dict[str, dict]) -> None:
    with open(_PROFILE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _profile_key(strategy: str, symbol: str) -> str:
    return f"{strategy}:{symbol.upper()}"


def save_profile(profile: SymbolFilterProfile) -> None:
    data = _load_all()
    data[_profile_key(profile.strategy, profile.symbol)] = asdict(profile)
    _save_all(data)


def load_profile(strategy: str, symbol: str) -> Optional[SymbolFilterProfile]:
    data = _load_all()
    raw = data.get(_profile_key(strategy, symbol.upper()))
    if raw is None:
        return None
    return SymbolFilterProfile(**raw)


def list_profiles(strategy: Optional[str] = None) -> List[SymbolFilterProfile]:
    data = _load_all()
    profiles = []
    for key, raw in data.items():
        if strategy and not key.startswith(f"{strategy}:"):
            continue
        try:
            profiles.append(SymbolFilterProfile(**raw))
        except Exception:
            pass
    return sorted(profiles, key=lambda p: p.symbol)


def delete_profile(strategy: str, symbol: str) -> bool:
    data = _load_all()
    key = _profile_key(strategy, symbol)
    if key in data:
        del data[key]
        _save_all(data)
        return True
    return False


# ── Auto-calibration ──────────────────────────────────────────────────────────

def calibrate_from_snapshots(
    strategy: str,
    symbol: str,
    snapshots: list,
) -> SymbolFilterProfile:
    """
    Derive optimal filter thresholds from trade snapshots for a specific symbol/strategy.

    Self-validating logic per filter:
    1. Check separation between win/loss distributions (effect size ≥ threshold)
    2. Scan candidate thresholds and pick the one that maximises win rate on
       the filtered subset — only keep it if filtered win rate > baseline win rate
       AND at least 40% of trades survive the filter (no over-pruning)
    3. If no threshold improves win rate, set filter to OFF (0.0)

    Strategy-specific indicator sets are used automatically.
    """
    wins   = [s for s in snapshots if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "win"]
    losses = [s for s in snapshots if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "loss"]

    def _get(s, key):
        return s.get(key) if isinstance(s, dict) else getattr(s, key)

    def _vals(group, key):
        return [_get(s, key) for s in group if _get(s, key) is not None]

    def _mean(vals):
        return statistics.mean(vals) if vals else 0.0

    def _separation(w_vals, l_vals):
        if len(w_vals) < 2 and len(l_vals) < 2:
            return 0.0
        wm, lm = _mean(w_vals), _mean(l_vals)
        var_w = statistics.variance(w_vals) if len(w_vals) >= 2 else 0.0
        var_l = statistics.variance(l_vals) if len(l_vals) >= 2 else 0.0
        pooled = ((var_w + var_l) / 2) ** 0.5 or 1.0
        return abs(wm - lm) / pooled

    baseline_wr = len(wins) / len(snapshots) if snapshots else 0.0
    min_survive = 0.40  # filter must keep ≥ 40% of all trades
    min_wr_gain = 0.03  # filtered win rate must beat baseline by ≥ 3pp

    def _best_min_threshold(all_vals_w, all_vals_l, indicator_key):
        """
        Scan percentiles of winning trade values as candidate thresholds.
        Pick the one that maximises win rate on surviving trades, subject to:
        - survival rate ≥ min_survive
        - win rate improvement ≥ min_wr_gain
        Returns best threshold or 0.0 if none qualifies.
        """
        all_snaps = snapshots
        all_ind = _vals(all_snaps, indicator_key)
        if not all_ind:
            return 0.0

        # Candidate thresholds: 10th–60th percentile of win values
        w_sorted = sorted(all_vals_w)
        candidates = set()
        for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            idx = max(0, int(len(w_sorted) * p))
            candidates.add(round(w_sorted[idx], 2))

        best_thresh = 0.0
        best_wr = baseline_wr

        for thresh in sorted(candidates):
            # Apply filter: keep only trades where indicator >= thresh
            surviving = [s for s in all_snaps
                         if (_get(s, indicator_key) or 0) >= thresh]
            if len(surviving) < 5:  # need at least 5 trades to draw a conclusion
                continue
            survival_rate = len(surviving) / len(all_snaps)
            if survival_rate < min_survive:
                continue
            s_wins = sum(1 for s in surviving
                         if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "win")
            filtered_wr = s_wins / len(surviving)
            if filtered_wr > best_wr + min_wr_gain:
                best_wr = filtered_wr
                best_thresh = thresh

        return best_thresh

    def _best_max_threshold(all_vals_w, all_vals_l, indicator_key):
        """Same but for indicators where lower = better (e.g. upper_wick_pct)."""
        all_snaps = snapshots
        w_sorted = sorted(all_vals_w)
        candidates = set()
        for p in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            idx = min(len(w_sorted) - 1, int(len(w_sorted) * p))
            candidates.add(round(w_sorted[idx], 2))

        best_thresh = 0.0
        best_wr = baseline_wr

        for thresh in sorted(candidates, reverse=True):
            surviving = [s for s in all_snaps
                         if (_get(s, indicator_key) or 999) <= thresh]
            if len(surviving) < 5:
                continue
            if len(surviving) / len(all_snaps) < min_survive:
                continue
            s_wins = sum(1 for s in surviving
                         if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "win")
            filtered_wr = s_wins / len(surviving)
            if filtered_wr > best_wr + min_wr_gain:
                best_wr = filtered_wr
                best_thresh = thresh

        return best_thresh

    def _calibrate_indicator(key: str, higher_is_better: bool = True) -> float:
        w_vals = _vals(wins, key)
        l_vals = _vals(losses, key)
        sep = _separation(w_vals, l_vals)
        if sep < 0.20:
            return 0.0
        if higher_is_better:
            if _mean(w_vals) <= _mean(l_vals):
                return 0.0
            return _best_min_threshold(w_vals, l_vals, key)
        else:
            if _mean(w_vals) >= _mean(l_vals):
                return 0.0
            return _best_max_threshold(w_vals, l_vals, key)

    # ── Common: volume ratio (all strategies) ─────────────────
    w_vol = _vals(wins,   "volume_ratio")
    l_vol = _vals(losses, "volume_ratio")
    vol_min = _calibrate_indicator("volume_ratio", higher_is_better=True)

    # ── Strategy-specific indicator calibration ────────────────
    ema_dist_min     = 0.0
    bb_pos_min       = 0.0
    ema_spread_min   = 0.0
    atr_pct_max      = 0.0
    bb_depth_min     = 0.0
    lower_wick_min   = 0.0
    rsi_min          = 0.0
    bb_pct_min       = 0.0
    ema_dist_pct_min = 0.0
    body_max         = 0.0
    swing_rsi_min    = 0.0

    if strategy == "EMA_Mean_Reversion":
        # Higher EMA distance and higher BB position both correlate with stronger pullback quality
        ema_dist_min = _calibrate_indicator("ema_dist_pct", higher_is_better=True)
        bb_pos_min   = _calibrate_indicator("bb_pct",       higher_is_better=True)

    elif strategy == "MA_Crossover_RSI":
        # Wider EMA spread at crossover = more separation = stronger momentum signal
        # Now captured correctly via ema_spread_pct in the trade analyzer
        ema_spread_min = _calibrate_indicator("ema_spread_pct", higher_is_better=True)

    elif strategy == "Breakout_Consolidation":
        # Tighter base (lower range/ATR ratio) = better breakout quality
        # range_atr_ratio: lower is better — a ratio of 1.5 is much tighter than 2.8
        atr_pct_max = _calibrate_indicator("range_atr_ratio", higher_is_better=False)

    elif strategy == "BB_Mean_Reversion":
        # RSI is the top discriminator (d~1.4): winners enter with RSI already recovering (higher)
        # bb_pct (BB position): winners are higher in the band = price already bouncing
        # ema_dist_pct: winners are closer to / above EMA20 = not in a downtrend
        # atr_pct: calmer regime = better mean reversion environment
        atr_pct_max      = _calibrate_indicator("atr_pct",      higher_is_better=False)
        bb_depth_min     = _calibrate_indicator("bb_depth_pct", higher_is_better=True)
        rsi_min          = _calibrate_indicator("rsi",          higher_is_better=True)
        bb_pct_min       = _calibrate_indicator("bb_pct",       higher_is_better=True)
        ema_dist_pct_min = _calibrate_indicator("ema_dist_pct", higher_is_better=True)

        # Combined survival check: if all filters together cut more than 60% of trades,
        # progressively disable the weakest ones (ema_dist first, then bb_pct, then rsi)
        # until survival >= 40% — keeps meaningful filters without over-pruning.
        def _combined_survival(thresholds_dict):
            surviving = snapshots
            for key, (thresh, direction) in thresholds_dict.items():
                if thresh == 0.0:
                    continue
                if direction == "min":
                    surviving = [s for s in surviving if (_get(s, key) or 0) >= thresh]
                else:
                    surviving = [s for s in surviving if (_get(s, key) or 999) <= thresh]
            return len(surviving) / len(snapshots) if snapshots else 1.0

        filters_bb = {
            "rsi":          (rsi_min,          "min"),
            "bb_pct":       (bb_pct_min,        "min"),
            "ema_dist_pct": (ema_dist_pct_min,  "min"),
            "atr_pct":      (atr_pct_max,       "max"),
        }
        # Relax filters one at a time until combined survival >= 40%.
        # Sort by individual impact (lowest individual survival = most restrictive = relax last).
        def _individual_survival(key, thresh, direction):
            if thresh == 0.0:
                return 1.0
            if direction == "min":
                return sum(1 for s in snapshots if (_get(s, key) or 0) >= thresh) / len(snapshots)
            return sum(1 for s in snapshots if (_get(s, key) or 999) <= thresh) / len(snapshots)

        # Compute individual survival for each active filter; relax those with highest
        # individual survival first (least impactful filters come off first)
        active_filters = [(k, t, d) for k, (t, d) in filters_bb.items() if t != 0.0]
        active_filters.sort(key=lambda x: _individual_survival(x[0], x[1], x[2]), reverse=True)

        for relax_key, _, _ in active_filters:
            if _combined_survival(filters_bb) >= 0.40:
                break
            filters_bb[relax_key] = (0.0, filters_bb[relax_key][1])

        rsi_min          = filters_bb["rsi"][0]
        bb_pct_min       = filters_bb["bb_pct"][0]
        ema_dist_pct_min = filters_bb["ema_dist_pct"][0]
        atr_pct_max      = filters_bb["atr_pct"][0]

    elif strategy == "Fib_Pullback_Support":
        # Larger lower wick = stronger rejection candle at the level.
        # Smaller candle body (body_pct, the d~0.7 discriminator) = indecision/absorption → wins.
        # (No atr filter here — the rule already gates on an ATR spike, so it'd double-count.)
        lower_wick_min = _calibrate_indicator("lower_wick_pct", higher_is_better=True)
        body_max       = _calibrate_indicator("body_pct",       higher_is_better=False)

    elif strategy == "RSI_Swing_Reversal":
        # Deeper RSI dip on the entry bar = a more genuine oversold pullback in the uptrend.
        swing_rsi_min = _calibrate_indicator("rsi", higher_is_better=True)

    # ── Evidence means ─────────────────────────────────────────
    w_ema = _vals(wins,   "ema_dist_pct")
    l_ema = _vals(losses, "ema_dist_pct")
    w_bb  = _vals(wins,   "bb_pct")
    l_bb  = _vals(losses, "bb_pct")

    n_wins   = len(wins)
    n_trades = len(snapshots)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    return SymbolFilterProfile(
        symbol=symbol.upper(),
        strategy=strategy,
        calibrated_at=datetime.utcnow().strftime("%Y-%m-%d"),
        n_trades=n_trades,
        n_wins=n_wins,
        win_rate_pct=round(win_rate, 1),
        vol_min=vol_min,
        ema_dist_min=ema_dist_min,
        bb_pos_min=bb_pos_min,
        ema_spread_min=ema_spread_min,
        atr_pct_max=atr_pct_max,
        bb_depth_min=bb_depth_min,
        lower_wick_min=lower_wick_min,
        rsi_min=rsi_min,
        bb_pct_min=bb_pct_min,
        ema_dist_pct_min=ema_dist_pct_min,
        body_max=body_max,
        swing_rsi_min=swing_rsi_min,
        win_ema_dist_mean=round(_mean(w_ema), 3),
        loss_ema_dist_mean=round(_mean(l_ema), 3),
        win_vol_mean=round(_mean(w_vol), 3),
        loss_vol_mean=round(_mean(l_vol), 3),
        win_bb_pos_mean=round(_mean(w_bb), 3),
        loss_bb_pos_mean=round(_mean(l_bb), 3),
    )


# ── Runtime lookup (called by strategy at signal time) ────────────────────────

def get_filters_for_symbol(strategy: str, symbol: str) -> dict:
    """
    Returns calibrated filter thresholds for this strategy+symbol.
    Falls back to empty dict (no filtering) if no profile exists.
    """
    profile = load_profile(strategy, symbol)
    if profile is None:
        return {}
    if strategy == "EMA_Mean_Reversion":
        return {"ema_dist_min": profile.ema_dist_min, "vol_min": profile.vol_min, "bb_pos_min": profile.bb_pos_min}
    if strategy == "MA_Crossover_RSI":
        return {"vol_min": profile.vol_min, "ema_spread_min": profile.ema_spread_min}
    if strategy == "Breakout_Consolidation":
        return {"vol_min": profile.vol_min, "atr_pct_max": profile.atr_pct_max}
    if strategy == "BB_Mean_Reversion":
        return {
            "vol_min":          profile.vol_min,
            "atr_pct_max":      profile.atr_pct_max,
            "bb_depth_min":     profile.bb_depth_min,
            "rsi_min":          profile.rsi_min,
            "bb_pct_min":       profile.bb_pct_min,
            "ema_dist_pct_min": profile.ema_dist_pct_min,
        }
    if strategy == "Fib_Pullback_Support":
        return {"lower_wick_min": profile.lower_wick_min, "vol_min": profile.vol_min,
                "body_max": profile.body_max}
    if strategy == "RSI_Swing_Reversal":
        return {"vol_min": profile.vol_min, "swing_rsi_min": profile.swing_rsi_min}
    return {}
