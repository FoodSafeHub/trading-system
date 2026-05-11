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
    ema_dist_min: float
    vol_min: float
    bb_pos_min: float

    # Metrics that justified these thresholds
    win_ema_dist_mean: float
    loss_ema_dist_mean: float
    win_vol_mean: float
    loss_vol_mean: float
    win_bb_pos_mean: float
    loss_bb_pos_mean: float

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
    snapshots: list,  # List[TradeSnapshot] or list of dicts from API
) -> SymbolFilterProfile:
    """
    Derive optimal filter thresholds from trade snapshots for a specific symbol.

    Threshold logic: use the 25th percentile of winning trade values as the
    minimum filter — this preserves ~75% of wins while excluding losses that
    cluster below that level.

    Returns a profile with thresholds set. Call save_profile() to persist.
    """
    wins  = [s for s in snapshots if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "win"]
    losses = [s for s in snapshots if (s.get("outcome") if isinstance(s, dict) else s.outcome) == "loss"]

    def _get(s, key):
        return s.get(key) if isinstance(s, dict) else getattr(s, key)

    def _vals(group, key):
        return [_get(s, key) for s in group if _get(s, key) is not None]

    def _pct(vals, p):
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = max(0, int(len(s) * p))
        return s[idx]

    def _mean(vals):
        return statistics.mean(vals) if vals else 0.0

    def _separation(w_vals, l_vals):
        if len(w_vals) < 2 or len(l_vals) < 2:
            return 0.0
        wm = _mean(w_vals)
        lm = _mean(l_vals)
        var_w = statistics.variance(w_vals)
        var_l = statistics.variance(l_vals)
        pooled = ((var_w + var_l) / 2) ** 0.5 or 1.0
        return abs(wm - lm) / pooled

    # EMA distance filter
    w_ema = _vals(wins,   "ema_dist_pct")
    l_ema = _vals(losses, "ema_dist_pct")
    ema_sep = _separation(w_ema, l_ema)
    # Only add filter if wins clearly higher than losses AND separation meaningful
    if ema_sep >= 0.25 and _mean(w_ema) > _mean(l_ema):
        ema_dist_min = round(_pct(w_ema, 0.25), 2)  # 25th pct of wins
    else:
        ema_dist_min = 0.0

    # Volume ratio filter
    w_vol = _vals(wins,   "volume_ratio")
    l_vol = _vals(losses, "volume_ratio")
    vol_sep = _separation(w_vol, l_vol)
    if vol_sep >= 0.20 and _mean(w_vol) > _mean(l_vol):
        vol_min = round(_pct(w_vol, 0.25), 2)
    else:
        vol_min = 0.0

    # BB position filter
    w_bb = _vals(wins,   "bb_pct")
    l_bb = _vals(losses, "bb_pct")
    bb_sep = _separation(w_bb, l_bb)
    if bb_sep >= 0.20 and _mean(w_bb) > _mean(l_bb):
        bb_pos_min = round(_pct(w_bb, 0.25), 2)
    else:
        bb_pos_min = 0.0

    n_wins = len(wins)
    n_trades = len(snapshots)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    return SymbolFilterProfile(
        symbol=symbol.upper(),
        strategy=strategy,
        calibrated_at=datetime.utcnow().strftime("%Y-%m-%d"),
        n_trades=n_trades,
        n_wins=n_wins,
        win_rate_pct=round(win_rate, 1),
        ema_dist_min=ema_dist_min,
        vol_min=vol_min,
        bb_pos_min=bb_pos_min,
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
    Returns the filter thresholds to apply for this strategy+symbol combination.
    Falls back to zeros (no filtering) if no profile exists.
    """
    profile = load_profile(strategy, symbol)
    if profile is None:
        return {"ema_dist_min": 0.0, "vol_min": 0.0, "bb_pos_min": 0.0}
    return {
        "ema_dist_min": profile.ema_dist_min,
        "vol_min":      profile.vol_min,
        "bb_pos_min":   profile.bb_pos_min,
    }
