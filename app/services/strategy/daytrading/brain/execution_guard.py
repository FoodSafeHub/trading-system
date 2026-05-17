"""
Execution Guard — final validation before a signal becomes a trade candidate.

This is the last gate. Even if the brain says "trade", the guard checks:
  1. R:R meets minimum threshold
  2. Signal confidence meets minimum threshold
  3. Not too close to market close (no new entries after 3:15 PM ET)
  4. Volume is sufficient (proxy: not a micro-cap or illiquid bar)
  5. Stop distance is not absurdly wide (slippage proxy)
  6. Signal is not stale (entry bar is recent, not hours old)

Each rejection includes an explicit reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from app.services.strategy.daytrading.market_open import ET
from app.services.strategy.daytrading.models import DayTradeSignal

_NO_NEW_ENTRY_AFTER = time(15, 15)
_MIN_BARS_STALE = 3   # signal older than 3 bars (15 min on 5m) is stale


@dataclass
class GuardDecision:
    accepted: bool
    reason: str   # single most important reason (accept or reject)
    checks: dict[str, bool]   # all individual check results


class ExecutionGuard:
    DEFAULT_CONFIG = {
        "min_rr": 1.5,
        "min_confidence": 0.50,
        "max_stop_pct": 3.0,    # stop > 3% from entry = too wide
        "min_volume": 50_000,   # minimum bar volume (shares/contracts)
        "stale_bars": 3,        # reject if signal bar is older than this many 5m bars
    }

    def __init__(self, config: dict | None = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def validate(
        self,
        signal: DayTradeSignal | dict,
        current_bar_time: datetime | None = None,
    ) -> GuardDecision:
        """
        Validate a signal before allowing execution.
        signal can be a DayTradeSignal dataclass or a plain dict.
        """
        if isinstance(signal, dict):
            entry = float(signal.get("entry_price", 0))
            stop = float(signal.get("stop_price", 0))
            target = float(signal.get("target_price", 0))
            confidence = float(signal.get("confidence", 0))
            sig_time_str = signal.get("signal_time", "")
            indicators = signal.get("indicators", {})
        else:
            entry = signal.entry_price
            stop = signal.stop_price
            target = signal.target_price
            confidence = signal.confidence
            sig_time_str = signal.signal_time
            indicators = signal.indicators

        checks: dict[str, bool] = {}
        rejection_reason = ""

        # ── 1. R:R check ──────────────────────────────────────────────────────
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0.0
        checks["rr_ok"] = rr >= self.config["min_rr"]
        if not checks["rr_ok"]:
            rejection_reason = f"R:R {rr:.2f} below minimum {self.config['min_rr']}."

        # ── 2. Confidence check ────────────────────────────────────────────────
        checks["confidence_ok"] = confidence >= self.config["min_confidence"]
        if not checks["confidence_ok"] and not rejection_reason:
            rejection_reason = f"Confidence {confidence:.0%} below minimum {self.config['min_confidence']:.0%}."

        # ── 3. Market close proximity ──────────────────────────────────────────
        # In backtest mode use the signal's own timestamp; live mode uses now().
        if current_bar_time is not None:
            try:
                ref_time = pd.Timestamp(current_bar_time)
                if ref_time.tzinfo is None:
                    ref_time = ref_time.tz_localize(ET)
                check_time = ref_time.time()
            except Exception:
                check_time = datetime.now(ET).time()
        else:
            check_time = datetime.now(ET).time()
        checks["not_near_close"] = check_time < _NO_NEW_ENTRY_AFTER
        if not checks["not_near_close"] and not rejection_reason:
            rejection_reason = f"Too close to market close — no new entries after {_NO_NEW_ENTRY_AFTER}."

        # ── 4. Stop distance (slippage proxy) ────────────────────────────────
        stop_pct = risk / entry * 100 if entry > 0 else 999
        checks["stop_not_too_wide"] = stop_pct <= self.config["max_stop_pct"]
        if not checks["stop_not_too_wide"] and not rejection_reason:
            rejection_reason = f"Stop distance {stop_pct:.2f}% exceeds maximum {self.config['max_stop_pct']}%."

        # ── 5. Volume proxy ────────────────────────────────────────────────────
        # Use indicator vol_ratio if present, else skip this check
        vol_ratio = indicators.get("vol_ratio") if isinstance(indicators, dict) else None
        if vol_ratio is not None:
            checks["volume_ok"] = float(vol_ratio) >= 0.5  # at least half avg volume
            if not checks["volume_ok"] and not rejection_reason:
                rejection_reason = f"Volume ratio {vol_ratio:.2f} too low — illiquid bar."
        else:
            checks["volume_ok"] = True  # can't check, assume ok

        # ── 6. Staleness check ────────────────────────────────────────────────
        checks["not_stale"] = True
        if sig_time_str and current_bar_time is not None:
            try:
                sig_dt = pd.Timestamp(sig_time_str)
                if sig_dt.tzinfo is None:
                    sig_dt = sig_dt.tz_localize(ET)
                cur_dt = pd.Timestamp(current_bar_time)
                if cur_dt.tzinfo is None:
                    cur_dt = cur_dt.tz_localize(ET)
                age_minutes = (cur_dt - sig_dt).total_seconds() / 60
                max_age = self.config["stale_bars"] * 5  # bars × 5 min
                checks["not_stale"] = age_minutes <= max_age
                if not checks["not_stale"] and not rejection_reason:
                    rejection_reason = f"Signal is {age_minutes:.0f} min old — stale (max {max_age} min)."
            except Exception:
                pass

        all_passed = all(checks.values())

        if all_passed:
            return GuardDecision(
                accepted=True,
                reason=f"All checks passed. R:R {rr:.1f}, conf {confidence:.0%}, stop {stop_pct:.2f}%.",
                checks=checks,
            )
        else:
            return GuardDecision(
                accepted=False,
                reason=rejection_reason or "One or more execution checks failed.",
                checks=checks,
            )
