from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.strategy.daytrading.autotrader import (
    AutoTraderConfig,
    get_manager,
)
from app.services.strategy.daytrading.market_open import (
    get_spy_regime,
    market_status,
)
from app.services.strategy.daytrading.runner import (
    get_provider_stats,
    get_session_symbols,
    run_backtest,
    run_backtest_all,
    run_scan,
    run_signals,
)
from app.services.strategy.daytrading.validation.walkforward import run_walkforward
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP
from app.services.strategy.daytrading.autotrader.native_entry import (
    SUPPORTED_NATIVE_STRATEGIES,
)

router = APIRouter(tags=["daytrading"])

# simple in-memory toggle store (resets on restart)
_disabled: set[str] = set()


@router.get("/signals/{symbol}")
def get_signals(symbol: str) -> dict[str, Any]:
    enabled = [s.name for s in ALL_STRATEGIES if s.name not in _disabled]
    return run_signals(symbol.upper(), enabled_strategies=enabled)


@router.get("/strategies")
def list_strategies() -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "timeframe": s.timeframe,
            "config": s.default_config,
            "enabled": s.name not in _disabled,
        }
        for s in ALL_STRATEGIES
    ]


@router.get("/data-source-status")
def data_source_status() -> dict[str, Any]:
    """Snapshot of bar-fetch provider usage since this API process started.

    Used by the Day Trading dashboard's provider-health badge. Counters reset
    on every API restart — this is a session-scoped view, not a historical
    audit. Read from the in-memory dict in ``runner._PROVIDER_COUNTERS``.
    """
    stats = get_provider_stats()
    primary = stats["primary_provider"]
    if primary == "twelvedata":
        status = "ok"
        label = "Twelve Data"
    elif primary == "webull":
        status = "degraded"
        label = "Webull (TD fallback)"
    elif primary == "yfinance":
        status = "degraded"
        label = "yfinance (both upstreams failed)"
    else:
        status = "idle"
        label = "no fetches yet"
    return {**stats, "status": status, "label": label}


@router.get("/backtest/{strategy}/{symbol}")
def backtest_strategy(
    strategy: str,
    symbol: str,
    period: str = Query("60d", description="60d | 90d | 730d"),
    initial_capital: float = Query(10_000.0),
    position_pct: float = Query(0.95),
) -> dict[str, Any]:
    if strategy not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    return run_backtest(
        symbol.upper(), strategy, period, initial_capital, position_pct
    )


@router.get("/backtest-all/{symbol}")
def backtest_all(
    symbol: str,
    period: str = Query("60d"),
    initial_capital: float = Query(10_000.0),
) -> list[dict[str, Any]]:
    return run_backtest_all(symbol.upper(), period, initial_capital)


@router.get("/scan")
def scan(
    symbols: str = Query("SPY,QQQ,AAPL,TSLA,NVDA"),
    strategies: str = Query("all"),
) -> list[dict[str, Any]]:
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    enabled = None if strategies == "all" else [s.strip() for s in strategies.split(",")]
    return run_scan(sym_list, enabled)


@router.get("/market-status")
def get_market_status() -> dict[str, Any]:
    status = market_status()
    regime, spy_vs_vwap, gap_pct, gap_type = get_spy_regime()
    return {
        **status,
        "regime": regime,
        "spy_vs_vwap_pct": spy_vs_vwap,
        "spy_gap_pct": gap_pct,
        "spy_gap_type": gap_type,
    }


@router.get("/scanner/watchlist")
def scanner_watchlist(
    max_symbols: int = Query(20, description="Max symbols to return"),
    universe: str = Query("", description="Comma-separated override universe; empty = full US-listed"),
    market_state: str = Query("", description="Force a market state (TREND_UP etc.); empty = auto"),
    min_price: float = Query(5.0, description="Minimum last price ($)"),
    max_price: float | None = Query(None, description="Maximum last price ($); null = no cap"),
    min_avg_volume: float = Query(1_000_000, description="Minimum 30-day average daily volume"),
    min_float: float | None = Query(None, description="Minimum shares float; null = no floor"),
    max_float: float | None = Query(None, description="Maximum shares float; null = no cap"),
    universe_max_symbols: int | None = Query(None, description="Cap total universe size; null = all (~5,800)"),
    run_native_precheck: bool = Query(
        True,
        description=(
            "Run the audited 4-strategy generate_signals() check on the top-K "
            "ranked candidates and promote names with active signals to the front."
        ),
    ),
    precheck_top_k: int = Query(
        10, ge=0, le=50,
        description="How many top-ranked candidates to pre-check (0 disables).",
    ),
) -> list[dict[str, Any]]:
    """Pre-market scanner: ranked watchlist with scores, tags, and strategy buckets.

    The default universe is the full US-listed common-stock universe
    (~5,800 names from NASDAQ Trader, refreshed daily). Use ``universe`` to
    pass a comma-separated override.
    """
    from app.services.strategy.daytrading.scanners import DayTradingScanner, DayTradingScannerConfig
    sym_list = [s.strip().upper() for s in universe.split(",") if s.strip()] or None
    cfg = DayTradingScannerConfig(
        min_price=min_price,
        max_price=max_price,
        min_avg_volume=min_avg_volume,
        min_float=min_float,
        max_float=max_float,
        universe_max_symbols=universe_max_symbols,
    )
    scanner = DayTradingScanner(config=cfg, universe=sym_list)
    state = market_state.strip() or None
    results = scanner.get_intraday_watchlist(
        max_symbols=max_symbols,
        market_state=state,
        run_native_precheck=run_native_precheck,
        precheck_top_k=precheck_top_k,
    )
    return [r.to_dict() for r in results]


@router.get("/scanner/metrics/{symbol}")
def scanner_symbol_metrics(symbol: str) -> dict[str, Any]:
    """Fetch raw scan metrics for a single symbol."""
    from app.services.strategy.daytrading.scanners import DayTradingScanner, DayTradingScannerConfig
    scanner = DayTradingScanner(config=DayTradingScannerConfig())
    metrics = scanner.fetch_metrics(symbol.upper())
    result = scanner.score_symbol(metrics)
    return result.to_dict()


@router.get("/backtest/walkforward/{strategy}/{symbol}")
def walkforward_backtest(
    strategy: str,
    symbol: str,
    years: int = Query(2, description="Years of history (max 2 due to 5m data limits)"),
    step_months: int = Query(3, description="Window step size in months"),
    initial_capital: float = Query(10_000.0),
    position_pct: float = Query(0.95),
) -> dict[str, Any]:
    """Walk-forward validation: IS/OOS windows with WFE scoring."""
    if strategy not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    result = run_walkforward(
        symbol=symbol.upper(),
        strategy_name=strategy,
        years=years,
        step_months=step_months,
        initial_capital=initial_capital,
        position_pct=position_pct,
    )
    return result.to_dict()


@router.post("/strategies/{name}/toggle")
def toggle_strategy(name: str, enabled: bool = Query(...)) -> dict[str, Any]:
    if name not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    if enabled:
        _disabled.discard(name)
    else:
        _disabled.add(name)
    return {"name": name, "enabled": enabled}


# ── Auto-trader switch ────────────────────────────────────────────────────────
# Flip ON to have the system day-trade automatically against a list of symbols
# using the active intraday strategies. Each symbol gets its own SingleStockTrader
# (polls 1m/5m/15m bars, enters/exits with the risk governor active, flattens at
# 3:45 PM ET). The whole switch auto-stops at 4:00 PM ET.


class AutoTraderStartRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Tickers to trade, e.g. ['TSLA','NVDA']")
    direction_mode: Literal["long_only", "short_only", "both"] = "long_only"
    trail_mode: Literal["ema", "atr", "candle"] = "atr"
    partial_tp: bool = True
    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.05, description="Fraction of capital risked per trade")
    max_daily_loss_pct: float = Field(2.0, gt=0, le=20.0)
    max_trades_per_day: int = Field(6, ge=1, le=50)
    max_consecutive_losses: int = Field(3, ge=1, le=10)
    initial_capital: float = Field(10_000.0, gt=0)
    broker_name: Literal["paper", "alpaca"] = "paper"
    entry_mode: Literal["legacy_entry_decider", "native_strategy"] = Field(
        "legacy_entry_decider",
        description=(
            "Entry-path mode. 'legacy_entry_decider' uses the composite scoring "
            "filter (default, current behavior). 'native_strategy' delegates to "
            "the same generate_signals() logic used in backtest for the audited "
            "strategies (BollingerMomentum, SupertrendTrend, EMAMomentum, ORBBreakout)."
        ),
    )
    native_strategies: list[str] | None = Field(
        None,
        description=(
            "Optional override for which strategies run when entry_mode="
            "'native_strategy'. Defaults to the audited 4-strategy set."
        ),
    )
    force: bool = Field(False, description="Arm even if outside regular trading hours")


@router.post("/autotrader/start")
def autotrader_start(req: AutoTraderStartRequest) -> dict[str, Any]:
    """Flip the day-trading switch ON. Spins up one trader per symbol."""
    cfg = AutoTraderConfig(
        symbols=req.symbols,
        direction_mode=req.direction_mode,
        trail_mode=req.trail_mode,
        partial_tp=req.partial_tp,
        risk_per_trade_pct=req.risk_per_trade_pct,
        max_daily_loss_pct=req.max_daily_loss_pct,
        max_trades_per_day=req.max_trades_per_day,
        max_consecutive_losses=req.max_consecutive_losses,
        initial_capital=req.initial_capital,
        broker_name=req.broker_name,
        entry_mode=req.entry_mode,
        native_strategies=req.native_strategies,
    )
    try:
        return get_manager().flip_on(cfg, force=req.force)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/autotrader/stop")
def autotrader_stop(
    flatten: bool = Query(False, description="Close open positions at market before stopping"),
) -> dict[str, Any]:
    """Flip the day-trading switch OFF. Pass flatten=true to also close open positions."""
    return get_manager().flip_off(flatten=flatten)


@router.get("/autotrader/status")
def autotrader_status() -> dict[str, Any]:
    """Snapshot of the switch, config, and every active trader."""
    return get_manager().status()


@router.get("/autotrader/decision-summary")
def autotrader_decision_summary(
    symbol: str | None = Query(None, description="Restrict to one symbol; omit for all"),
    limit: int | None = Query(
        None,
        description=(
            "Look at only the most-recent N decision_log entries per symbol. "
            "Omit for the full log (capped at 200 entries by the trader)."
        ),
        ge=1,
        le=200,
    ),
) -> dict[str, Any]:
    """Native-vs-legacy decision aggregation.

    Returns per-symbol agreement counts (both/native_only/legacy_only/neither),
    native rejection-category histogram, native winning-strategy frequency,
    legacy low-score reason histogram, and the last 20 disagreement cycles.
    """
    return get_manager().decision_summary(symbol=symbol, limit=limit)


@router.post("/autotrader/flatten")
def autotrader_flatten(
    symbol: str | None = Query(None, description="Symbol to flatten; omit to flatten ALL"),
) -> dict[str, Any]:
    """Force-close one or all open positions without stopping the loop."""
    return {"flattened": get_manager().flatten(symbol)}


# ── Scanner → AutoTrader bridge ───────────────────────────────────────────────
# Closes the manual copy-paste gap: pick today's best intraday candidates with
# the existing DayTradingScanner and feed them straight into the same
# manager.flip_on() flow used by /autotrader/start. No new execution path, no
# new risk controls — just symbol selection wired to symbol arming.


# Map scanner bucket → audited native strategy that fits it. Used when the
# caller asks us to filter scanner results to specific buckets. The buckets
# the scanner emits are: "ORB" | "VWAP" | "gap" | "momentum" | "none".
_BUCKET_TO_NATIVE_STRATEGY: dict[str, str] = {
    "ORB": "ORBBreakout",
    "momentum": "EMAMomentum",
    "VWAP": "SupertrendTrend",
    "gap": "BollingerMomentum",
}


class StartFromScannerRequest(BaseModel):
    max_symbols: int = Field(
        10, ge=1, le=50,
        description="How many top scanner picks to arm (post-regime adjustment).",
    )
    allowed_buckets: list[Literal["ORB", "VWAP", "gap", "momentum"]] | None = Field(
        None,
        description=(
            "Restrict scanner picks to these strategy buckets. None = keep all "
            "audited-strategy-mappable buckets (ORB/VWAP/gap/momentum)."
        ),
    )

    # AutoTrader knobs — defaults match the manual /autotrader/start defaults
    # except entry_mode, which we flip to native_strategy because that's the
    # whole point of using the scanner pipeline.
    direction_mode: Literal["long_only", "short_only", "both"] = "long_only"
    trail_mode: Literal["ema", "atr", "candle"] = "atr"
    partial_tp: bool = True
    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.05)
    max_daily_loss_pct: float = Field(2.0, gt=0, le=20.0)
    max_trades_per_day: int = Field(6, ge=1, le=50)
    max_consecutive_losses: int = Field(3, ge=1, le=10)
    initial_capital: float = Field(10_000.0, gt=0)
    broker_name: Literal["paper", "alpaca"] = "paper"
    entry_mode: Literal["legacy_entry_decider", "native_strategy"] = "native_strategy"
    native_strategies: list[str] | None = None

    # Scanner knobs — only the few that are obviously safe to pass through.
    market_state: str | None = Field(
        None,
        description="Force a market state (TREND_UP etc.); omit for brain-resolved.",
    )
    min_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Drop scanner picks whose adjusted_score is below this floor.",
    )
    require_native_signal: bool = Field(
        False,
        description=(
            "When True, only arm symbols whose native pre-check found at least "
            "one audited strategy with an accepted signal at the current bar. "
            "Strongest filter — if zero symbols match, the endpoint returns 409."
        ),
    )
    prefer_native_signal: bool = Field(
        True,
        description=(
            "Soft preference: when True, picks with native_signal_active=True "
            "are ordered ahead of picks without an active signal, even if their "
            "adjusted_score is lower. Tiebroken by best_native_confidence, then "
            "adjusted_score. Ignored when require_native_signal=True."
        ),
    )
    min_best_native_confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description=(
            "Optional confidence floor applied ONLY to picks where the native "
            "pre-check ran and produced a best_native_confidence. Picks without "
            "an active signal are not filtered by this; pair it with "
            "require_native_signal=True for a strict 'firing with conviction' run."
        ),
    )
    precheck_top_k: int = Field(
        10, ge=0, le=50,
        description="How many top-ranked candidates to pre-check (0 disables).",
    )

    force: bool = Field(False, description="Arm even if outside regular trading hours.")


@router.post("/autotrader/start-from-scanner")
def autotrader_start_from_scanner(req: StartFromScannerRequest) -> dict[str, Any]:
    """Pick today's best intraday candidates with the day-trading scanner and
    arm the autotrader on them in one shot.

    Reuses ``DayTradingScanner.get_intraday_watchlist`` for selection and the
    existing ``AutoTraderManager.flip_on`` for arming — no parallel execution
    path, no separate risk controls.
    """
    # Lazy import to mirror the existing /scanner/watchlist route and avoid
    # paying scanner import cost on app startup.
    from app.services.strategy.daytrading.scanners import (
        DayTradingScanner,
        DayTradingScannerConfig,
    )

    scanner = DayTradingScanner(config=DayTradingScannerConfig())
    # Over-fetch slightly so post-filter we still have enough candidates. The
    # native pre-check runs inside get_intraday_watchlist on the top-K of the
    # returned list, so we ask for at least precheck_top_k so the active-signal
    # surface isn't wasted when max_symbols is small.
    fetch_n = max(req.max_symbols * 2, req.max_symbols, req.precheck_top_k)
    raw = scanner.get_intraday_watchlist(
        max_symbols=fetch_n,
        market_state=req.market_state,
        run_native_precheck=(req.precheck_top_k > 0 or req.require_native_signal),
        precheck_top_k=max(req.precheck_top_k, req.max_symbols if req.require_native_signal else 0),
    )

    allowed = set(req.allowed_buckets) if req.allowed_buckets else set(_BUCKET_TO_NATIVE_STRATEGY)

    # ── Step 1: filter ────────────────────────────────────────────────────────
    # Hard rules first — anything that fails here cannot be armed.
    candidates: list[Any] = []
    for r in raw:
        if r.recommended_strategy_bucket not in allowed:
            continue
        if r.adjusted_score < req.min_score:
            continue
        if req.require_native_signal and not r.native_signal_active:
            continue
        # Confidence floor applies only to picks that actually have a confidence
        # number. A scanner row without an active signal carries
        # best_native_confidence=None and is exempt — pair with
        # require_native_signal=True if you want strict gating.
        if (
            req.min_best_native_confidence is not None
            and r.best_native_confidence is not None
            and r.best_native_confidence < req.min_best_native_confidence
        ):
            continue
        candidates.append(r)

    # ── Step 2: rank with explicit selection policy ───────────────────────────
    # Priority (descending):
    #   1. native_signal_active=True (when prefer_native_signal=True)
    #   2. best_native_confidence (None treated as 0.0)
    #   3. adjusted_score
    # When prefer_native_signal=False we drop tier (1) and rank purely by
    # confidence then adjusted_score — useful for pre-market broad arming
    # where the pre-check fields are mostly empty anyway.
    def _selection_key(p: Any) -> tuple[int, float, float]:
        tier = 1 if (req.prefer_native_signal and p.native_signal_active) else 0
        conf = p.best_native_confidence if p.best_native_confidence is not None else 0.0
        return (tier, conf, p.adjusted_score)

    candidates.sort(key=_selection_key, reverse=True)

    # ── Step 3: take top N ────────────────────────────────────────────────────
    picks = candidates[: req.max_symbols]

    if not picks:
        # No usable candidates today — fail clearly. The caller (UI / scheduler)
        # can decide whether to retry, widen filters, or stand down.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Scanner returned no candidates that match the filters "
                f"(allowed_buckets={sorted(allowed)}, min_score={req.min_score}, "
                f"require_native_signal={req.require_native_signal}, "
                f"min_best_native_confidence={req.min_best_native_confidence}, "
                f"market_state={req.market_state or 'auto'}). Nothing armed."
            ),
        )

    symbols = [p.symbol for p in picks]

    # Build the same AutoTraderConfig the manual /autotrader/start route uses.
    cfg = AutoTraderConfig(
        symbols=symbols,
        direction_mode=req.direction_mode,
        trail_mode=req.trail_mode,
        partial_tp=req.partial_tp,
        risk_per_trade_pct=req.risk_per_trade_pct,
        max_daily_loss_pct=req.max_daily_loss_pct,
        max_trades_per_day=req.max_trades_per_day,
        max_consecutive_losses=req.max_consecutive_losses,
        initial_capital=req.initial_capital,
        broker_name=req.broker_name,
        entry_mode=req.entry_mode,
        native_strategies=req.native_strategies or list(SUPPORTED_NATIVE_STRATEGIES),
    )
    try:
        start_result = get_manager().flip_on(cfg, force=req.force)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    scanner_summary = [
        {
            "symbol": p.symbol,
            "score": p.score,
            "adjusted_score": p.adjusted_score,
            "bucket": p.recommended_strategy_bucket,
            "tags": p.tags,
            "mapped_native_strategy": _BUCKET_TO_NATIVE_STRATEGY.get(
                p.recommended_strategy_bucket
            ),
            "native_signal_active": p.native_signal_active,
            "active_native_strategies": list(p.active_native_strategies),
            "best_native_strategy": p.best_native_strategy,
            "best_native_side": p.best_native_side,
            "best_native_confidence": (
                round(p.best_native_confidence, 3)
                if p.best_native_confidence is not None else None
            ),
        }
        for p in picks
    ]

    return {
        "selected_symbols": symbols,
        "scanner_summary": scanner_summary,
        "scanner_candidates_considered": len(raw),
        "market_state": req.market_state or "auto",
        "autotrader": start_result,
    }
