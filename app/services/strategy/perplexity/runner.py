from __future__ import annotations

"""
Runner for Perplexity strategies.
Evaluates all enabled strategies against live or historical OHLCV data.
"""

from typing import List

import pandas as pd

from app.services.market_regime import MarketRegime, get_current_regime
from app.services.performance_breakdown import bucket_atr_pct
from app.services.perplexity.suitability import is_strategy_suitable, load_suitability_config
from app.services.strategy.perplexity.base import PerplexitySignal, PerplexityStrategy
from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend,
    MaCrossoverRsi,
    BreakoutConsolidation,
    BollingerMeanReversionUptrend,
    FibPullbackSupport,
    RsiSwingReversal,
    SupertrendSwing,
    BollingerBandBreakout,
)
from app.services.strategy.perplexity.momentum_strategies import (
    PerpEngulfingVolumeSurge,
    PerpNarrowRangeBreakout,
    PerpThreeBarPush,
    PerpHammerShootingStar,
)
from app.services.strategy.perplexity.india_swing_strategies import (
    NiftyLeaderPullback,
    FiftyTwoWeekHighBreakout,
    VcpContractionBreakout,
)
from app.services.strategy.perplexity.india_advanced_strategies import (
    MomentumBreakout,
    TrendPullbackEma,
    TrendFollowingHHHL,
    SupportResistanceBounce,
    WyckoffSpringTest,
)

PERPLEXITY_STRATEGIES: List[PerplexityStrategy] = [
    EmaMeanReversionUptrend(),
    MaCrossoverRsi(),
    BreakoutConsolidation(),
    BollingerMeanReversionUptrend(),
    FibPullbackSupport(),
    RsiSwingReversal(),
    SupertrendSwing(),
    BollingerBandBreakout(),
    PerpEngulfingVolumeSurge(),
    PerpNarrowRangeBreakout(),
    PerpThreeBarPush(),
    PerpHammerShootingStar(),
    # India swing set (research_only until the Nifty-100+midcap backtest
    # justifies flipping each flag — see project_india_swing_revamp).
    NiftyLeaderPullback(),
    FiftyTwoWeekHighBreakout(),
    VcpContractionBreakout(),
    # Five advanced India strategies (spec-driven; research_only until backtested).
    MomentumBreakout(),
    TrendPullbackEma(),
    TrendFollowingHHHL(),
    SupportResistanceBounce(),
    WyckoffSpringTest(),
]


def get_perplexity_strategies() -> List[PerplexityStrategy]:
    """Active strategy list. DEFAULT = the bespoke classes above (unchanged).

    Only when ``use_unified_perplexity`` is explicitly enabled do we swap in the
    rule-backed adapters (Engine B as a display layer over rules.py). The flag
    defaults to False, so scanner / Perplexity-page / recommendations output is
    byte-identical to today until someone opts in.
    """
    try:
        from app.config import get_settings
        if getattr(get_settings(), "use_unified_perplexity", False):
            from app.services.strategy.perplexity.adapter import build_unified_adapters
            return build_unified_adapters()
    except Exception:
        pass
    return PERPLEXITY_STRATEGIES


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _current_atr(df: pd.DataFrame, period: int = 14) -> float:
    atr = _atr_series(df, period)
    v = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(df["Close"].iloc[-1]) * 0.02
    return v


def _live_benchmark_close(symbol: str) -> "pd.Series | None":
    """Latest benchmark index close series for RS-relative strategies (live path).

    India symbols rate vs ^NSEI, US vs SPY. This is the LIVE path, so using
    current data is correct (no point-in-time concern). The provider caches the
    fetch, so repeated scanner calls don't hammer the network. Returns None on
    any failure — RS strategies then degrade to "no RS" (HOLD), never crash.
    """
    try:
        from app.services.markets import is_india_symbol
        from app.services.market_data.provider import get_ohlcv
        ticker = "^NSEI" if is_india_symbol(symbol) else "SPY"
        return get_ohlcv(ticker, period="2y")["Close"]
    except Exception:
        return None


def run_perplexity_signal(
    symbol: str,
    df: pd.DataFrame,
    regime: MarketRegime | None = None,
) -> List[PerplexitySignal]:
    """Run all enabled Perplexity strategies on the given OHLCV DataFrame."""
    results = []
    if regime is None and not df.empty:
        regime = get_current_regime(df.index[-1])
    regime = regime or MarketRegime.BULL

    suitability_config = None
    try:
        suitability_config = load_suitability_config()
    except Exception:
        suitability_config = None

    volatility_bucket = bucket_atr_pct(_current_atr(df)) if not df.empty else "unknown"
    # Benchmark for RS-relative India strategies (Momentum_Breakout, Trend_Following).
    # Without this they get no benchmark and never fire. Live data is correct here.
    benchmark_close = _live_benchmark_close(symbol) if not df.empty else None

    for strategy in get_perplexity_strategies():
        # Skip strategies that are off entirely (RETIRE set). RESEARCH-ONLY
        # strategies ARE surfaced here so their signals appear in Live Signals /
        # Scanner for inspection — but they remain net-losing in backtest and
        # must NOT be auto-traded until validated. Going live (auto-trade)
        # requires a separate, explicit symbol→strategy assignment on the
        # Strategy page; surfacing a signal here does not place any order.
        if not strategy.enabled:
            continue
        try:
            sig = strategy.run(
                symbol,
                df,
                regime=regime,
                volatility_bucket=volatility_bucket,
                suitability_config=suitability_config,
                benchmark_close=benchmark_close,
            )
            if sig.direction == "BUY" and suitability_config is not None:
                allowed, reason = is_strategy_suitable(
                    strategy.name,
                    symbol,
                    regime.value if regime else None,
                    volatility_bucket,
                    suitability_config,
                )
                if not allowed:
                    sig = PerplexitySignal(
                        symbol=symbol,
                        strategy_name=strategy.name,
                        direction="HOLD",
                        reason=f"Blocked by suitability: {reason}",
                        suitability_blocked=True,
                        suitability_reason=reason,
                        volatility_bucket=volatility_bucket,
                    )
            results.append(sig)
        except Exception as exc:
            results.append(PerplexitySignal(
                symbol=symbol,
                strategy_name=strategy.name,
                direction="HOLD",
                reason=f"error: {exc}",
                volatility_bucket=volatility_bucket,
            ))
    return results
