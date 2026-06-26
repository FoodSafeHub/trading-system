from __future__ import annotations

"""
Scanner service — the core engine that:
  1. Builds a candidate universe
  2. Applies liquidity filters
  3. Runs all strategies on each passing symbol
  4. Scores and ranks candidates
  5. Persists results to SQLite
  6. Optionally auto-trades the top-ranked symbol
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

from app.config import get_settings
from app.db import SessionLocal
from app.models.scan_results import ScanResult
from app.schemas.scanner import ScanConfig, ScanResultOut, ScanSummary
from app.services.market_data.provider import get_ohlcv, get_price_series
from app.services.scanner.scan_filters import apply_filters, fetch_float_shares, passes_float_filter
from app.services.scanner.universe_service import get_universe
from app.services.strategy.engine import StrategyEngine, load_strategies_from_config
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES, run_perplexity_signal

logger = logging.getLogger(__name__)

_engine = StrategyEngine()

# ── Scoring weights ──────────────────────────────────────────────────────────
# Final score = weighted sum, capped at 100.
_W_CONSENSUS = 40   # points per agreeing strategy (up to 2 strategies = 80 pts max from this)
_W_PERPLEXITY = 15  # bonus for perplexity strategy agreement
_W_VOLUME = 10      # bonus for high relative volume
_W_TREND = 10       # bonus for price above SMA50


def _score_candidate(
    strategies_agreeing: int,
    perplexity_count: int,
    avg_volume: float | None,
    price: float | None,
    df,
) -> float:
    score = 0.0

    # Consensus score — more strategies agreeing = higher score
    score += min(strategies_agreeing * _W_CONSENSUS, 60)

    # Perplexity bonus
    score += min(perplexity_count * _W_PERPLEXITY, 15)

    # Volume bonus — if avg volume is very high (>2M) award extra points
    if avg_volume and avg_volume > 2_000_000:
        score += _W_VOLUME
    elif avg_volume and avg_volume > 1_000_000:
        score += _W_VOLUME // 2

    # Trend bonus — price above 50-day SMA
    if df is not None and not df.empty and price and len(df) >= 50:
        sma50 = float(df["Close"].iloc[-50:].mean())
        if price > sma50:
            score += _W_TREND

    return min(round(score, 1), 100.0)


# ── Signal-scan strategy identity ────────────────────────────────────────────
# Maps a generic strategy TYPE (the stable identifier the picker uses) to the
# label suffix the scanner stores in votes via cfg.name = f"{symbol}_{Label}".
# Built once from the factory so it can't drift from _make_generic_configs.
def _generic_type_to_label() -> dict[str, str]:
    out: dict[str, str] = {}
    for cfg in _make_generic_configs("AAPL"):
        # cfg.name == "AAPL_<Label>"; strip the "AAPL_" prefix to recover <Label>.
        label = cfg.name[len("AAPL_"):] if cfg.name.startswith("AAPL_") else cfg.name
        out[cfg.type] = label
    return out


def _strategy_matches(stored_name: str, selected: set[str]) -> bool:
    """True if a vote's stored strategy name matches any selected identifier.

    `stored_name` is what the scan recorded in votes:
      * generic  -> "{symbol}_{Label}" (e.g. "AAPL_RSI2_Mean_Reversion")
      * perplexity -> "perplexity:{Name}" (e.g. "perplexity:RSI_Swing_Reversal")
    `selected` holds the picker identifiers:
      * generic  -> the TYPE (e.g. "rsi2_mean_reversion")
      * perplexity -> "perplexity:{Name}"
    """
    if not stored_name:
        return False
    # Perplexity: exact label match.
    if stored_name.startswith("perplexity:"):
        return stored_name in selected
    # Generic: stored name is "{symbol}_{label}", so match the exact "_{label}"
    # suffix (not a bare endswith, which could misfire if one label were a suffix
    # of another future label).
    for stype, label in _generic_type_to_label().items():
        if stype in selected and stored_name.endswith("_" + label):
            return True
    return False


def run_scan(config: ScanConfig) -> ScanSummary:
    """
    Execute a full scan and return a ScanSummary with top candidates.
    This is the main entry point called by the API and scheduler.
    """
    t_start = time.time()
    scan_run_id = str(uuid.uuid4())[:12]
    scanned_at = datetime.now(tz=timezone.utc)
    settings = get_settings()

    logger.info("[scanner] Starting scan — universe=%s run_id=%s", config.universe, scan_run_id)

    # ── Step 1: Build universe ───────────────────────────────────────────────
    all_symbols = get_universe(config.universe, config.custom_symbols)
    all_symbols = list(all_symbols)
    logger.info("[scanner] Universe size: %s symbols", len(all_symbols))

    total_scanned = 0
    total_passed_filters = 0
    total_float_rejected = 0

    # ── Shares-float filter setup ────────────────────────────────────────────
    # Float fetching is slow/rate-limited, so it runs ONLY for symbols that
    # already cleared the cheap price/volume gates, and reuses a disk cache that
    # is shared with the day-trading scanner and refreshed daily. India (Upstox)
    # symbols have no float feed, so the band is skipped for them.
    float_filter_active = bool(config.min_float or config.max_float)
    float_cache: dict = {}
    float_cache_dirty = False
    if float_filter_active:
        try:
            from app.services.strategy.daytrading.scanners.universe import load_float_cache
            float_cache = load_float_cache()
        except Exception as e:
            logger.warning("[scanner] Could not load float cache: %s", e)
        logger.info(
            "[scanner] Float filter active: %s–%s shares (%d cached)",
            f"{config.min_float/1e6:,.1f}M" if config.min_float else "0",
            f"{config.max_float/1e6:,.1f}M" if config.max_float else "∞",
            len(float_cache),
        )

    try:
        from app.services.markets import is_india_symbol
    except Exception:
        def is_india_symbol(_s: str) -> bool:  # type: ignore
            return False

    # votes[symbol][direction] = {strategies: [], perplexity_count: int, df, price, avg_vol}
    votes: dict = {}

    # ── Step 2: Process in batches ───────────────────────────────────────────
    for i in range(0, len(all_symbols), config.batch_size):
        batch = all_symbols[i: i + config.batch_size]
        for symbol in batch:
            total_scanned += 1
            try:
                df = get_ohlcv(symbol, period="1y")
            except Exception as e:
                logger.debug("[scanner] %s: data fetch failed — %s", symbol, e)
                continue

            # ── Step 3: Liquidity filters ────────────────────────────────────
            filt = apply_filters(
                symbol, df,
                min_price=config.min_price,
                min_avg_volume=config.min_avg_volume,
                max_price=config.max_price,
            )
            if not filt.passed:
                logger.debug("[scanner] %s filtered: %s", symbol, filt.reason)
                continue

            # ── Shares-float band (US only; slow fetch, gated behind liquidity)
            if float_filter_active and not is_india_symbol(symbol):
                fshares = float_cache.get(symbol)
                if fshares is None:
                    fshares = fetch_float_shares(symbol)
                    float_cache[symbol] = fshares
                    float_cache_dirty = True
                ok, why = passes_float_filter(
                    fshares, min_float=config.min_float, max_float=config.max_float
                )
                if not ok:
                    total_float_rejected += 1
                    logger.debug("[scanner] %s float-filtered: %s", symbol, why)
                    continue

            total_passed_filters += 1
            votes[symbol] = {
                "directions": defaultdict(list),
                "perplexity_counts": defaultdict(int),
                "df": df,
                "price": filt.price,
                "avg_vol": filt.avg_volume,
            }

            # ── Step 4a: Run Bollinger/classic strategies ─────────────────────
            bollinger_configs = [c for c in load_strategies_from_config()
                                 if c.enabled and c.symbol.upper() == symbol.upper()]
            if not bollinger_configs:
                # Symbol not in strategies.json — run generic strategy types
                bollinger_configs = _make_generic_configs(symbol)

            try:
                prices = df["Close"].dropna()
                for cfg in bollinger_configs:
                    sigs = _engine.run(cfg, prices)
                    for s in sigs:
                        if s.direction != "HOLD":
                            votes[symbol]["directions"][s.direction].append(cfg.name)
            except Exception as e:
                logger.debug("[scanner] %s bollinger error: %s", symbol, e)

            # ── Step 4b: Run Perplexity strategies ───────────────────────────
            try:
                if len(df) >= 220:
                    perp_sigs = run_perplexity_signal(symbol, df)
                    for sig in perp_sigs:
                        if sig.direction != "HOLD":
                            votes[symbol]["directions"][sig.direction].append(
                                f"perplexity:{sig.strategy_name}"
                            )
                            votes[symbol]["perplexity_counts"][sig.direction] += 1
            except Exception as e:
                logger.debug("[scanner] %s perplexity error: %s", symbol, e)

        # Small pause between batches to avoid rate-limiting yfinance
        if i + config.batch_size < len(all_symbols):
            time.sleep(0.5)

    # Persist any newly-fetched floats so the next scan (and the day-trading
    # scanner) reuse them instead of re-hitting yfinance.
    if float_cache_dirty:
        try:
            from app.services.strategy.daytrading.scanners.universe import save_float_cache
            save_float_cache(float_cache)
        except Exception as e:
            logger.warning("[scanner] Could not save float cache: %s", e)

    # ── Step 5: Score and rank ───────────────────────────────────────────────
    candidates: list[dict] = []

    # Signal mode: keep only the agreeing strategies the user selected (ANY match).
    signal_mode = (config.scan_mode == "signal")
    selected_strats = set(config.signal_strategies or [])

    for symbol, data in votes.items():
        for direction, strategy_list in data["directions"].items():
            if signal_mode:
                # Strategy-first lens: narrow the agreeing list to the user's
                # picks. The symbol qualifies (ANY) iff at least one selected
                # strategy fired this direction; score/reason reflect only those.
                strategy_list = [
                    s for s in strategy_list if _strategy_matches(s, selected_strats)
                ]
            if len(strategy_list) < 1:
                continue
            perp_count = data["perplexity_counts"].get(direction, 0)
            score = _score_candidate(
                strategies_agreeing=len(strategy_list),
                perplexity_count=perp_count,
                avg_volume=data["avg_vol"],
                price=data["price"],
                df=data["df"],
            )
            if signal_mode:
                reason = f"{', '.join(strategy_list[:3])} fired {direction}"
                if len(strategy_list) > 3:
                    reason += f" +{len(strategy_list)-3} more"
            else:
                reason = f"{len(strategy_list)} strategies agree: {', '.join(strategy_list[:3])}"
                if len(strategy_list) > 3:
                    reason += f" +{len(strategy_list)-3} more"

            candidates.append({
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "strategies_agreeing": len(strategy_list),
                "strategy_name": strategy_list[0],
                "price": data["price"],
                "avg_vol": data["avg_vol"],
                "reason": reason,
                "indicators_json": None,
            })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Apply direction filter BEFORE the top-N slice so the top window is
    # filled exclusively with the requested side. Without this, SELLs can
    # crowd out BUYs (or vice versa) when one side dominates a given day.
    wanted_dir = (config.scan_direction or "ANY").upper()
    if wanted_dir in {"BUY", "SELL"}:
        candidates = [c for c in candidates if str(c.get("direction", "")).upper() == wanted_dir]

    top = candidates[: config.top_n]

    logger.info(
        "[scanner] %s symbols scanned, %s passed filters (%s float-rejected), "
        "%s matches (direction=%s), top %s selected",
        total_scanned, total_passed_filters, total_float_rejected,
        len(candidates), wanted_dir, len(top),
    )

    # ── Step 6: Persist to DB ────────────────────────────────────────────────
    saved_rows: list[ScanResult] = []
    with SessionLocal() as db:
        for c in top:
            row = ScanResult(
                scan_run_id=scan_run_id,
                scanned_at=scanned_at,
                universe=config.universe,
                symbol=c["symbol"],
                strategy_name=c["strategy_name"],
                direction=c["direction"],
                score=c["score"],
                strategies_agreeing=c["strategies_agreeing"],
                price=c["price"],
                avg_volume=c["avg_vol"],
                reason=c["reason"],
                indicators_json=c["indicators_json"],
                auto_traded=False,
            )
            db.add(row)
            saved_rows.append(row)
        db.commit()
        for row in saved_rows:
            db.refresh(row)

    # ── Step 7: Optional auto-trade top candidate ────────────────────────────
    # If auto_trade_direction is set (BUY/SELL), pick the top-ranked candidate
    # matching that side. ANY keeps the old behavior. Avoids the user flipping
    # the toggle and accidentally taking a SELL when they wanted a BUY signal.
    if config.auto_trade_top and top:
        wanted = (config.auto_trade_direction or "ANY").upper()
        if wanted == "ANY":
            _auto_trade_top(top[0], scan_run_id, scanned_at)
        else:
            match = next((c for c in top if str(c.get("direction", "")).upper() == wanted), None)
            if match is not None:
                _auto_trade_top(match, scan_run_id, scanned_at)
            else:
                logger.info("[scanner] auto-trade skipped: no %s candidate in top %d",
                            wanted, len(top))

    duration = round(time.time() - t_start, 1)

    top_out = [
        ScanResultOut(
            id=row.id,
            scan_run_id=row.scan_run_id,
            scanned_at=row.scanned_at,
            universe=row.universe,
            symbol=row.symbol,
            strategy_name=row.strategy_name,
            direction=row.direction,
            score=row.score,
            strategies_agreeing=row.strategies_agreeing,
            price=row.price,
            avg_volume=row.avg_volume,
            reason=row.reason,
            auto_traded=row.auto_traded,
        )
        for row in saved_rows
    ]

    return ScanSummary(
        scan_run_id=scan_run_id,
        scanned_at=scanned_at,
        universe=config.universe,
        total_scanned=total_scanned,
        total_passed_filters=total_passed_filters,
        total_matches=len(candidates),
        top_candidates=top_out,
        duration_seconds=duration,
    )


def _auto_trade_top(candidate: dict, scan_run_id: str, scanned_at: datetime) -> None:
    """DISCOVERY-ONLY: record the scan candidate as a Signal row and notify;
    do NOT place any broker order.

    The scanner is for *discovery*. The scheduler (`_run_cycle`, every 15
    minutes) is the SOLE execution authority -- it reads assignments, sizes
    BUYs per the per-symbol cap (held-aware), and routes SELLs through the
    tight trail. Letting the scanner also place orders meant two parallel
    paths could fire on the same signal with different caps and different
    SELL policies (the path that flattened BNY on 2026-06-04 14:16 ET).

    What this function does now:
      * writes a Signal row with `source='scanner'` and `acted_on=False`
        (so it shows up in the Recent Signals feed with a clear "scanner
        discovery only" provenance);
      * emits a notification with the same provenance;
      * marks the matching ScanResult as auto_traded=False -- the column
        is kept for back-compat with the dashboard table, but it now
        means "scanner saw this", not "the broker filled this".

    It deliberately does NOT call get_broker / ExecutionService / place_order.
    Any future MARKET/TRAIL order on this candidate will come from the
    scheduler's next cycle, gated by the user's assignment row.
    """
    symbol = candidate["symbol"]
    direction = candidate["direction"]
    price = candidate["price"] or 0.0
    strategy_label = "scanner:" + (candidate.get("strategy_name") or "top")

    # Persist discovery signal -- acted_on=False because the scanner doesn't
    # execute. The scheduler decides whether to act on its own next cycle.
    try:
        from app.models.signals import Signal
        with SessionLocal() as db:
            sig = Signal(
                strategy_name=strategy_label[:128],
                symbol=symbol.upper(),
                direction=direction,
                strength=1.0,
                price_at_signal=price or None,
                acted_on=False,   # DISCOVERY only; scheduler is execution authority
            )
            db.add(sig)
            db.commit()
    except Exception as exc:
        logger.warning("[scanner] Could not persist discovery signal for %s: %s", symbol, exc)

    try:
        from app.services.notifications.bus import notify_signal
        notify_signal(
            symbol=symbol, direction=direction,
            strategy=strategy_label,
            source="scanner",
            price=price or None,
            extra=(
                f"Score: {candidate.get('score', 0):.0f} -- DISCOVERY ONLY, "
                f"scheduler executes on its 15-min cycle if assigned."
            ),
        )
    except Exception:
        pass

    # auto_traded=False so the dashboard distinguishes discovery from a real fill.
    try:
        with SessionLocal() as db:
            rows = db.query(ScanResult).filter_by(
                scan_run_id=scan_run_id, symbol=symbol
            ).all()
            for r in rows:
                r.auto_traded = False
            db.commit()
    except Exception:
        pass

    logger.info(
        "[scanner] DISCOVERY %s %s @ ~$%.2f (score=%.1f) -- no order placed; "
        "scheduler will act on its next cycle if symbol is assigned.",
        direction, symbol, price, candidate.get("score", 0),
    )


def _make_generic_configs(symbol: str):
    """Create a set of generic strategy configs for a symbol not in strategies.json.

    Returns all 5 new strategy types so the backtest page can run the same
    set on any user-typed ticker (e.g. BRK-B) and the scanner's signals are
    directly reproducible there.

    If a per-symbol calibrated profile exists (saved from the Backtest page's
    Optimize Filters flow), its param overrides are merged onto the factory
    defaults here — so the scheduler, the live scanner, and backtests all run
    the tuned params with no extra wiring. The rule logic is never changed;
    only the params dict it receives.
    """
    from app.services.strategy.models import StrategyConfig
    from app.services.backtest.scanner_profiles import get_param_overrides

    def _p(strategy_type: str, defaults: dict) -> dict:
        merged = dict(defaults)
        merged.update(get_param_overrides(strategy_type, symbol))
        return merged

    return [
        StrategyConfig(
            name=f"{symbol}_RSI2_Mean_Reversion",
            symbol=symbol,
            type="rsi2_mean_reversion",
            enabled=True,
            params=_p("rsi2_mean_reversion",
                    {"rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 70,
                    "sma_trend": 200, "exit_sma": 5, "hard_stop_pct": 5.0,
                    "take_profit_pct": 8.0, "max_hold_bars": 10, "atr_skip_threshold": 5.0}),
        ),
        StrategyConfig(
            name=f"{symbol}_EMA_MACD_Crossover",
            symbol=symbol,
            type="ema_macd_crossover",
            enabled=True,
            params=_p("ema_macd_crossover",
                    {"ema_fast": 9, "ema_slow": 21, "macd_fast": 12, "macd_slow": 26,
                    "macd_signal": 9, "rsi_period": 14, "rsi_min": 45, "rsi_max": 65,
                    "vol_ratio_min": 1.1, "atr_stop_multiplier": 1.5,
                    "atr_tp_multiplier": 2.0, "max_hold_bars": 20}),
        ),
        StrategyConfig(
            name=f"{symbol}_BB_Squeeze_Breakout",
            symbol=symbol,
            type="bb_squeeze_breakout",
            enabled=True,
            params=_p("bb_squeeze_breakout",
                    {"bb_period": 20, "bb_std": 2.0, "squeeze_bars": 5,
                    "vol_ratio_min": 1.3, "rsi_period": 14,
                    "rsi_entry_min": 50, "rsi_overbought": 80}),
        ),
        StrategyConfig(
            name=f"{symbol}_Pullback_EMA50",
            symbol=symbol,
            type="pullback_ema50",
            enabled=True,
            params=_p("pullback_ema50",
                    {"ema_trend": 50, "ema_slope_bars": 5, "price_ema_proximity_pct": 1.0,
                    "rsi_period": 14, "rsi_min": 35, "rsi_max": 55, "wick_ratio_min": 0.4,
                    "exit_rsi": 65, "exit_extension_pct": 3.0, "hard_stop_pct": 2.0,
                    "max_hold_bars": 20, "bear_skip_threshold_pct": 10.0}),
        ),
        StrategyConfig(
            name=f"{symbol}_VIX_Spike_Reversal",
            symbol=symbol,
            type="vix_spike_reversal",
            enabled=True,
            params=_p("vix_spike_reversal",
                    {"atr_period": 14, "atr_spike_threshold": 3.0, "atr_exit_threshold": 2.0,
                    "rsi_period": 14, "rsi_entry_max": 30, "rsi_exit": 55,
                    "bb_pos_max": 0.15, "wick_ratio_min": 0.5,
                    "prior_decline_pct": 2.0, "prior_decline_bars": 3}),
        ),
    ]


def _make_generic_configs_full(symbol: str):
    """Same as _make_generic_configs but also includes the 2 legacy strategy
    types (Bollinger Mean Reversion + Fibonacci Pullback). Used by the
    Custom Symbol backtest comparison so the user sees all 7 strategies
    available for AAPL/MSFT/etc. on any ticker.

    Kept SEPARATE from _make_generic_configs because the live scanner is
    tuned around the 5 regime-aware strategies — pulling in legacy ones
    there would slow scans and reintroduce noisier signals.
    """
    from app.services.strategy.models import StrategyConfig
    base = _make_generic_configs(symbol)
    legacy = [
        StrategyConfig(
            name=f"Legacy_{symbol}_BB_Mean_Reversion",
            symbol=symbol,
            type="bollinger",
            enabled=True,
            params={"bb_period": 20, "bb_std": 2.0, "rsi_low": 35, "rsi_high": 50},
        ),
        StrategyConfig(
            name=f"Legacy_{symbol}_Fib_Pullback",
            symbol=symbol,
            type="fib_pullback",
            enabled=True,
            params={"lookback": 60, "rsi_low": 40, "rsi_high": 55, "fib_tolerance": 0.015},
        ),
    ]
    return base + legacy


def _make_unified_configs(symbol: str):
    """Phase 1 PARALLEL config set — the 6 consolidated daily strategies, each
    paired with a default exit_policy so the exit-overlay layer manages
    trail/time/trend-fail/regime exits. Used ONLY by opt-in callers (e.g. the
    Backtest Compare-All `strategy_set=unified` path). The live scanner and
    scheduler are untouched and keep using _make_generic_configs.

    Per-symbol calibration is inherited from predecessor types via the alias
    map (get_param_overrides resolves new->old), so tuning saved under the old
    names is reused without re-calibration.
    """
    from app.services.strategy.models import StrategyConfig
    from app.services.backtest.scanner_profiles import get_param_overrides

    def _p(strategy_type: str, defaults: dict) -> dict:
        merged = dict(defaults)
        merged.update(get_param_overrides(strategy_type, symbol))
        return merged

    specs = [
        ("rsi2_reversion", "RSI2_Reversion", {
            "rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 65,
            "sma_trend": 200, "exit_sma": 5, "atr_skip_threshold": 5.0, "hard_stop_pct": 2.5,
            "exit_policy": {"trail": "atr", "atr_mult": 1.5, "trigger_pct": 3.0, "time_stop_bars": 8},
        }),
        ("trend_pullback", "Trend_Pullback", {
            "ema_trend": 50, "ema_slope_bars": 5, "atr_proximity_mult": 1.2, "rsi_period": 14,
            "rsi_min": 35, "rsi_max": 55, "wick_ratio_min": 0.4, "exit_rsi": 65,
            "exit_extension_pct": 3.0, "bear_skip_threshold_pct": 10.0, "stop_atr_mult": 1.5,
            "exit_policy": {"trail": "chandelier", "atr_mult": 3.0, "time_stop_bars": 20, "trend_fail": "ema:50"},
        }),
        ("squeeze_breakout", "Squeeze_Breakout", {
            "bb_period": 20, "bb_std": 2.0, "squeeze_lookback": 20, "squeeze_percentile": 0.40,
            "rsi_entry_min": 50, "vol_ratio_min": 1.3, "break_atr_frac": 0.10, "exit_ema": 20,
            "stop_atr_mult": 2.0,
            "exit_policy": {"trail": "chandelier", "atr_mult": 3.0, "time_stop_bars": 15, "trend_fail": "ema:20"},
        }),
        ("momentum_breakout", "Momentum_Breakout", {
            "donchian": 20, "donchian_exit": 10, "ema_fast": 9, "ema_slow": 21,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "vol_ratio_min": 1.2,
            "sma_trend": 200, "stop_atr_mult": 2.5,
            "exit_policy": {"trail": "chandelier", "atr_mult": 3.0, "trend_fail": "ema_cross:9,21", "regime_exit": "deep_bear"},
        }),
        ("panic_reversal", "Panic_Reversal", {
            "atr_period": 14, "atr_spike_threshold": 3.0, "atr_exit_threshold": 2.0,
            "rsi_period": 14, "rsi_entry_max": 30, "rsi_exit": 55, "bb_pos_max": 0.20,
            "wick_ratio_min": 0.5, "prior_decline_pct": 2.0, "prior_decline_bars": 3, "hard_stop_pct": 4.0,
            "exit_policy": {"trail": "atr", "atr_mult": 1.5, "trigger_pct": 2.0, "time_stop_bars": 8},
        }),
        ("trend_follow", "Trend_Follow", {
            "st_period": 10, "st_multiplier": 3.0, "sma_trend": 200, "adx_min": 20.0,
            "exit_policy": {"trail": "chandelier", "atr_mult": 4.5, "trend_fail": "supertrend", "regime_exit": "deep_bear"},
        }),
    ]
    return [
        StrategyConfig(name=f"{symbol}_{label}", symbol=symbol, type=stype,
                       enabled=True, params=_p(stype, defaults))
        for stype, label, defaults in specs
    ]


def get_latest_results(limit: int = 50) -> list[ScanResultOut]:
    """Return the most recent scan results from DB."""
    with SessionLocal() as db:
        rows = (
            db.query(ScanResult)
            .order_by(ScanResult.scanned_at.desc(), ScanResult.score.desc())
            .limit(limit)
            .all()
        )
        return [
            ScanResultOut(
                id=r.id,
                scan_run_id=r.scan_run_id,
                scanned_at=r.scanned_at,
                universe=r.universe,
                symbol=r.symbol,
                strategy_name=r.strategy_name,
                direction=r.direction,
                score=r.score,
                strategies_agreeing=r.strategies_agreeing,
                price=r.price,
                avg_volume=r.avg_volume,
                reason=r.reason,
                auto_traded=r.auto_traded,
            )
            for r in rows
        ]
