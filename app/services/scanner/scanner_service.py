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
from app.services.scanner.scan_filters import apply_filters
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
            )
            if not filt.passed:
                logger.debug("[scanner] %s filtered: %s", symbol, filt.reason)
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

    # ── Step 5: Score and rank ───────────────────────────────────────────────
    candidates: list[dict] = []

    for symbol, data in votes.items():
        for direction, strategy_list in data["directions"].items():
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
    top = candidates[: config.top_n]

    logger.info("[scanner] %s symbols scanned, %s passed filters, %s matches, top %s selected",
                total_scanned, total_passed_filters, len(candidates), len(top))

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
    if config.auto_trade_top and top:
        _auto_trade_top(top[0], scan_run_id, scanned_at)

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
    """Place a paper/live order for the top-ranked scan candidate if risk checks pass."""
    import asyncio
    from app.schemas.orders import OrderRequest
    from app.services.brokers.factory import get_broker
    from app.services.execution.service import ExecutionService
    from app.services.risk.engine import RiskEngine

    symbol = candidate["symbol"]
    direction = candidate["direction"]
    price = candidate["price"] or 0.0

    risk = RiskEngine()
    order_req = OrderRequest(
        symbol=symbol,
        side=direction,
        order_type="MARKET",
        quantity=1,
        source="scanner",
    )
    check = risk.check(order_req, estimated_price=price)
    if not check.passed:
        logger.info("[scanner] Auto-trade blocked for %s: %s", symbol, check.blocked_reason)
        return

    try:
        loop = asyncio.new_event_loop()
        broker = get_broker()
        loop.run_until_complete(broker.authenticate())
        accounts = loop.run_until_complete(broker.get_accounts())
        account_id = accounts[0].account_id if accounts else ""
        svc = ExecutionService(broker)
        # Stamp a Signal row so Recent Fills can show the strategy that fired.
        sig_id: int | None = None
        try:
            from app.models.signals import Signal
            with SessionLocal() as db:
                sig = Signal(
                    strategy_name=("scanner:" + (candidate.get("strategy_name") or "top"))[:128],
                    symbol=symbol.upper(),
                    direction=direction,
                    strength=1.0,
                    price_at_signal=price or None,
                    acted_on=True,
                )
                db.add(sig)
                db.commit()
                db.refresh(sig)
                sig_id = sig.id
        except Exception:
            sig_id = None
        loop.run_until_complete(svc.execute(order_req, account_id=account_id, signal_id=sig_id))
        loop.close()

        # Mark as auto-traded
        with SessionLocal() as db:
            rows = db.query(ScanResult).filter_by(
                scan_run_id=scan_run_id, symbol=symbol
            ).all()
            for r in rows:
                r.auto_traded = True
            db.commit()

        logger.info("[scanner] Auto-traded %s %s @ ~$%.2f (score=%.1f)",
                    direction, symbol, price, candidate["score"])
    except Exception as e:
        logger.error("[scanner] Auto-trade failed for %s: %s", symbol, e)


def _make_generic_configs(symbol: str):
    """Create a set of generic strategy configs for a symbol not in strategies.json."""
    from app.services.strategy.models import StrategyConfig
    return [
        StrategyConfig(
            name=f"{symbol}_RSI2_Mean_Reversion",
            symbol=symbol,
            type="rsi2_mean_reversion",
            enabled=True,
            params={"rsi_period": 2, "rsi_entry_threshold": 10, "rsi_exit_threshold": 70,
                    "sma_trend": 200, "exit_sma": 5, "hard_stop_pct": 5.0,
                    "take_profit_pct": 8.0, "max_hold_bars": 10, "atr_skip_threshold": 5.0},
        ),
        StrategyConfig(
            name=f"{symbol}_EMA_MACD_Crossover",
            symbol=symbol,
            type="ema_macd_crossover",
            enabled=True,
            params={"ema_fast": 9, "ema_slow": 21, "macd_fast": 12, "macd_slow": 26,
                    "macd_signal": 9, "rsi_period": 14, "rsi_min": 45, "rsi_max": 65,
                    "vol_ratio_min": 1.1, "atr_stop_multiplier": 1.5,
                    "atr_tp_multiplier": 2.0, "max_hold_bars": 20},
        ),
        StrategyConfig(
            name=f"{symbol}_Pullback_EMA50",
            symbol=symbol,
            type="pullback_ema50",
            enabled=True,
            params={"ema_trend": 50, "ema_slope_bars": 5, "price_ema_proximity_pct": 1.0,
                    "rsi_period": 14, "rsi_min": 35, "rsi_max": 55, "wick_ratio_min": 0.4,
                    "exit_rsi": 65, "exit_extension_pct": 3.0, "hard_stop_pct": 2.0,
                    "max_hold_bars": 20, "bear_skip_threshold_pct": 10.0},
        ),
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
