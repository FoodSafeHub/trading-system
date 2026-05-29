from __future__ import annotations

"""
Exit-policy overlay layer — Phase 0 scaffolding.

This generalizes the single Chandelier trailing overlay that used to live in
``rules.py`` into a pluggable layer driven by a per-strategy ``exit_policy``
dict. It is the seam through which Phase 1 strategies will declare trail-first
exits (ATR / Chandelier / time-stop / trend-failure / regime).

PRECEDENCE in :func:`apply_exit_overlay` (and the Phase-0 behaviour contract):

  1. ``position is None``                  -> return the signal unchanged
                                              (entries are NEVER altered).
  2. ``params['exit_policy']`` present     -> run the NEW policy path.
  3. elif ``params['trail_enabled']``      -> run the LEGACY Chandelier overlay,
                                              byte-for-byte as before.
  4. else                                  -> return the signal unchanged.

Because no Phase 0 config sets ``exit_policy``, only branches 1/3/4 execute in
production and branch 3 is a verbatim copy of the old function — so existing
trailing symbols and passthrough behaviour are identical. The policy path is
exercised only by tests until Phase 1.

Partial exits are DEFINED in the schema but NOT honoured yet (the engine is
all-or-nothing); a policy carrying ``partial`` logs once and is treated as a
full exit at the first leg.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.strategy.models import StrategySignal

logger = logging.getLogger(__name__)

_PARTIAL_WARNED = False


# ── Exit policy schema ─────────────────────────────────────────────────────────

@dataclass
class ExitPolicy:
    trail: str = "none"                 # "none" | "atr" | "chandelier"
    atr_mult: float = 3.0
    atr_period: int = 22
    trigger_pct: float = 3.0            # start trailing once unrealized >= this %
    time_stop_bars: Optional[int] = None
    trend_fail: str = "none"            # "none" | "ema:N" | "ema_cross:f,s" | "supertrend"
    regime_exit: str = "none"           # "none" | "deep_bear"
    partial: List[Dict[str, float]] = field(default_factory=list)  # NOT honoured yet

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> Optional["ExitPolicy"]:
        raw = params.get("exit_policy")
        if not raw:
            return None
        if isinstance(raw, ExitPolicy):
            return raw
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


# ── ATR helper (kept local to avoid a rules.py <-> exits.py import cycle) ──────

def _atr_raw(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR via EWM — identical formula to rules._atr_raw."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── Public entry point ─────────────────────────────────────────────────────────

def apply_exit_overlay(
    signal: StrategySignal,
    prices: pd.Series,
    ohlcv: Optional[pd.DataFrame],
    params: Dict[str, Any],
    position,  # Optional[PositionState] — typed loosely to avoid import cycle
) -> StrategySignal:
    """Dispatch to the policy path, the legacy Chandelier path, or passthrough.

    See module docstring for the precedence contract.
    """
    if position is None:
        return signal
    if params.get("exit_policy"):
        policy = ExitPolicy.from_params(params)
        if policy is None:
            return signal
        return _apply_policy(signal, prices, ohlcv, policy, position)
    if params.get("trail_enabled"):
        return _legacy_chandelier(signal, prices, ohlcv, params, position)
    return signal


# ── Legacy Chandelier overlay (verbatim move from rules._apply_chandelier_overlay) ─

def _legacy_chandelier(
    signal: StrategySignal,
    prices: pd.Series,
    ohlcv: Optional[pd.DataFrame],
    params: Dict[str, Any],
    position,
) -> StrategySignal:
    """ATR-Chandelier trailing-stop overlay on the exit decision.

    Once an open position is up >= trail_trigger_pct, stop honouring the rule's
    fixed-band SELL and ride the trend until the close breaks below the
    Chandelier line (highest_close_since_entry - atr_mult*ATR). Below the
    trigger, the rule's SELL is honoured as-is. Entries are never affected.
    """
    if not params.get("trail_enabled") or position is None:
        return signal

    trigger_pct = float(params.get("trail_trigger_pct", 3.0))
    atr_mult    = float(params.get("atr_trail_mult", 3.0))
    atr_period  = int(params.get("atr_trail_period", 22))

    if ohlcv is None or "High" not in ohlcv.columns or len(ohlcv) < atr_period + 1:
        return signal  # can't compute ATR -> leave the rule's decision alone

    c_now = float(prices.iloc[-1])
    if position.entry_price <= 0:
        return signal
    unreal_pct = (c_now - position.entry_price) / position.entry_price * 100.0
    if unreal_pct < trigger_pct:
        return signal  # not yet in profit-runway zone — honour the rule's exit

    atr_v = float(_atr_raw(ohlcv, atr_period).iloc[-1])
    if pd.isna(atr_v) or atr_v <= 0:
        return signal
    chandelier = position.highest_close - atr_mult * atr_v

    if c_now <= chandelier:
        return StrategySignal(
            symbol=signal.symbol, direction="SELL", strength=signal.strength,
            price_at_signal=c_now, indicators={**signal.indicators, "trail_exit": True,
                                               "chandelier": round(chandelier, 2)},
            strategy_name=signal.strategy_name,
        )
    if signal.direction == "SELL":
        return StrategySignal(
            symbol=signal.symbol, direction="HOLD", strength=signal.strength,
            price_at_signal=c_now, indicators={**signal.indicators, "trail_hold": True,
                                               "chandelier": round(chandelier, 2)},
            strategy_name=signal.strategy_name,
        )
    return signal


# ── New policy path (exercised only by tests until Phase 1) ────────────────────

def _apply_policy(
    signal: StrategySignal,
    prices: pd.Series,
    ohlcv: Optional[pd.DataFrame],
    policy: ExitPolicy,
    position,
) -> StrategySignal:
    """Trail-first exit evaluation. Single-leg only: a triggered exit emits a
    full SELL. `partial` is parsed but not honoured (engine is all-or-nothing).

    Order of checks: regime_exit -> time_stop -> trend_fail -> trail. The rule's
    own BUY is never overridden; its SELL passes through if nothing else fires.
    """
    global _PARTIAL_WARNED
    if policy.partial and not _PARTIAL_WARNED:
        logger.debug("[exits] exit_policy 'partial' is not honoured yet (Phase 2); "
                     "treating as full exit at the first leg.")
        _PARTIAL_WARNED = True

    c_now = float(prices.iloc[-1])

    # 1. Regime exit
    if policy.regime_exit == "deep_bear" and _regime_is_deep_bear(ohlcv, prices):
        return _force_sell(signal, c_now, {"regime_exit": "deep_bear"})

    # 2. Time stop
    bars_held = int(getattr(position, "bars_held", 0) or 0)
    if policy.time_stop_bars is not None and bars_held >= int(policy.time_stop_bars):
        return _force_sell(signal, c_now, {"time_stop": bars_held})

    # 3. Trend failure
    if policy.trend_fail and policy.trend_fail != "none" and _trend_failed(policy.trend_fail, prices, ohlcv):
        return _force_sell(signal, c_now, {"trend_fail": policy.trend_fail})

    # 4. Trailing stop
    if policy.trail in ("atr", "chandelier"):
        line = _trail_line(policy, prices, ohlcv, position)
        if line is not None:
            if position.entry_price > 0:
                unreal_pct = (c_now - position.entry_price) / position.entry_price * 100.0
            else:
                unreal_pct = 0.0
            if unreal_pct >= policy.trigger_pct:
                if c_now <= line:
                    return _force_sell(signal, c_now, {"trail_exit": True,
                                                       "trail_line": round(line, 4)})
                if signal.direction == "SELL":
                    return _hold(signal, c_now, {"trail_hold": True,
                                                 "trail_line": round(line, 4)})

    return signal


def _trail_line(policy: ExitPolicy, prices: pd.Series, ohlcv, position) -> Optional[float]:
    if ohlcv is None or "High" not in ohlcv.columns or len(ohlcv) < policy.atr_period + 1:
        return None
    atr_v = float(_atr_raw(ohlcv, policy.atr_period).iloc[-1])
    if pd.isna(atr_v) or atr_v <= 0:
        return None
    if policy.trail == "chandelier":
        return float(position.highest_close) - policy.atr_mult * atr_v
    # "atr": trail off the current close
    return float(prices.iloc[-1]) - policy.atr_mult * atr_v


def _trend_failed(spec: str, prices: pd.Series, ohlcv) -> bool:
    c_now = float(prices.iloc[-1])
    if spec.startswith("ema_cross:"):
        try:
            fast_s, slow_s = spec.split(":", 1)[1].split(",")
            fast, slow = int(fast_s), int(slow_s)
        except Exception:
            return False
        if len(prices) < slow + 2:
            return False
        ef = prices.ewm(span=fast, adjust=False).mean()
        es = prices.ewm(span=slow, adjust=False).mean()
        return float(ef.iloc[-1]) < float(es.iloc[-1])
    if spec.startswith("ema:"):
        try:
            period = int(spec.split(":", 1)[1])
        except Exception:
            return False
        if len(prices) < period + 1:
            return False
        ema = prices.ewm(span=period, adjust=False).mean()
        return c_now < float(ema.iloc[-1])
    if spec == "supertrend":
        # Lightweight proxy: close below a wide ATR band off the recent high.
        if ohlcv is None or "High" not in ohlcv.columns or len(ohlcv) < 11:
            return False
        atr_v = float(_atr_raw(ohlcv, 10).iloc[-1])
        if pd.isna(atr_v) or atr_v <= 0:
            return False
        recent_high = float(ohlcv["High"].iloc[-10:].max())
        return c_now < recent_high - 3.0 * atr_v
    return False


def _regime_is_deep_bear(ohlcv, prices: pd.Series) -> bool:
    """Deep-bear proxy using the optional spy_close column, else the symbol's
    own price: close > 20% below its 200-bar SMA."""
    series = None
    if ohlcv is not None and "spy_close" in getattr(ohlcv, "columns", []):
        series = ohlcv["spy_close"].dropna()
    else:
        series = prices
    if series is None or len(series) < 200:
        return False
    sma200 = float(series.rolling(200).mean().iloc[-1])
    if sma200 <= 0:
        return False
    return float(series.iloc[-1]) < sma200 * 0.80


def _force_sell(signal: StrategySignal, c_now: float, extra: dict) -> StrategySignal:
    return StrategySignal(
        symbol=signal.symbol, direction="SELL", strength=signal.strength,
        price_at_signal=c_now, indicators={**signal.indicators, **extra},
        strategy_name=signal.strategy_name,
    )


def _hold(signal: StrategySignal, c_now: float, extra: dict) -> StrategySignal:
    return StrategySignal(
        symbol=signal.symbol, direction="HOLD", strength=signal.strength,
        price_at_signal=c_now, indicators={**signal.indicators, **extra},
        strategy_name=signal.strategy_name,
    )
