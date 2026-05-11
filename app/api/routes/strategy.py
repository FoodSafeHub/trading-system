from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from collections import defaultdict

from app.config import get_settings
from app.schemas.orders import OrderRequest
from app.services.brokers.factory import get_broker
from app.services.execution.service import ExecutionService
from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.ema import compute_ema
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma
from app.services.market_data.provider import get_ohlcv, get_price_series
from app.services.strategy.engine import StrategyEngine, load_strategies_from_config
from app.services.strategy.scheduler import get_scheduler_status, set_scheduler_system_flags

router = APIRouter(prefix="/strategy", tags=["strategy"])
_engine = StrategyEngine()


@router.post("/run")
async def run_strategy_cycle():
    """
    Manually trigger one full strategy cycle with signal consensus filtering.
    An order is only placed when min_signal_agreement strategies agree on the
    same symbol + direction.
    """
    settings = get_settings()
    min_agree = int(settings.min_signal_agreement)
    configs = load_strategies_from_config()
    broker = get_broker()
    await broker.authenticate()
    accounts = await broker.get_accounts()
    account_id = accounts[0].account_id if accounts else ""
    svc = ExecutionService(broker)

    # ── Step 1: collect all signals ──────────────────────────
    raw_results = []
    # votes[symbol][direction] = list of strategy names
    votes: dict = defaultdict(lambda: defaultdict(list))

    for config in configs:
        if not config.enabled:
            continue
        try:
            prices = get_price_series(config.symbol, period="1y", use_cache=False)
            signals = _engine.run(config, prices)
            for s in signals:
                raw_results.append({
                    "strategy": config.name,
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "indicators": s.indicators,
                    "price": s.price_at_signal,
                })
                if s.direction != "HOLD":
                    votes[s.symbol][s.direction].append(config.name)
        except Exception as exc:
            raw_results.append({"strategy": config.name, "error": str(exc)})

    # ── Step 2: place orders only where consensus is met ─────
    orders_placed: dict = {}  # (symbol, direction) → order_status
    consensus_info: dict = {}  # (symbol, direction) → agreeing strategies

    for symbol, directions in votes.items():
        for direction, agreeing in directions.items():
            count = len(agreeing)
            consensus_info[(symbol, direction)] = {"count": count, "strategies": agreeing}
            if count >= min_agree:
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=1,
                )
                order = await svc.execute(order_req, account_id=account_id)
                orders_placed[(symbol, direction)] = order.status if order else "risk_blocked"

    # ── Step 3: annotate results with consensus info ─────────
    results = []
    for r in raw_results:
        if "error" in r:
            results.append(r)
            continue
        key = (r["symbol"], r["direction"])
        info = consensus_info.get(key, {})
        agree_count = info.get("count", 0)
        agree_names = info.get("strategies", [])
        order_status = orders_placed.get(key)

        consensus_met = agree_count >= min_agree and r["direction"] != "HOLD"
        results.append({
            **r,
            "agreement": f"{agree_count}/{min_agree} needed",
            "agreeing_strategies": agree_names,
            "consensus_met": consensus_met,
            "order_status": order_status if consensus_met else (
                f"waiting — only {agree_count}/{min_agree} agree" if r["direction"] != "HOLD" else None
            ),
        })

    return {
        "results": results,
        "count": len(results),
        "min_signal_agreement": min_agree,
        "orders_placed": len(orders_placed),
    }


@router.get("/configs")
def list_configs():
    """Return active strategy configurations."""
    configs = load_strategies_from_config()
    return [{"name": c.name, "symbol": c.symbol, "type": c.type, "enabled": c.enabled} for c in configs]


@router.get("/scheduler")
def scheduler_status():
    """Return current scheduler state."""
    return get_scheduler_status()


@router.post("/scheduler/config")
def update_scheduler_config(run_bollinger: bool | None = None, run_perplexity: bool | None = None):
    """
    Toggle which strategy systems participate in the auto-scheduler cycle.
    Changes take effect on the next scheduled cycle — no restart required.
    """
    set_scheduler_system_flags(run_bollinger, run_perplexity)
    return get_scheduler_status()


@router.get("/chart/{symbol}")
def chart_data(symbol: str, period: str = "3mo"):
    """
    Return OHLCV + indicator data for charting.
    period: yfinance period string (1mo, 3mo, 6mo, 1y)
    """
    try:
        symbol = symbol.upper()
        df = get_ohlcv(symbol, period=period)
        closes = df["Close"].dropna()

        def _safe(fn, *args):
            try:
                return fn(*args).values.tolist()
            except Exception:
                return [None] * len(closes)

        try:
            bb = compute_bollinger(closes, 20, 2.0)
            bb_upper  = bb.upper.values.tolist()
            bb_middle = bb.middle.values.tolist()
            bb_lower  = bb.lower.values.tolist()
        except Exception:
            bb_upper = bb_middle = bb_lower = [None] * len(closes)

        sma10 = _safe(compute_sma, closes, 10)
        sma30 = _safe(compute_sma, closes, 30)
        rsi14 = _safe(compute_rsi, closes, 14)
        ema9  = _safe(compute_ema, closes, 9)
        try:
            macd_result = compute_macd(closes, 12, 26, 9)
            macd_line = macd_result.macd.values.tolist()
            macd_sig  = macd_result.signal.values.tolist()
            macd_hist = macd_result.histogram.values.tolist()
        except Exception:
            macd_line = macd_sig = macd_hist = [None] * len(closes)

        dates = [str(d)[:10] for d in df.index]

        return {
            "symbol": symbol,
            "dates": dates,
            "open":  [round(v, 4) if v == v else None for v in df["Open"].tolist()],
            "high":  [round(v, 4) if v == v else None for v in df["High"].tolist()],
            "low":   [round(v, 4) if v == v else None for v in df["Low"].tolist()],
            "close": [round(v, 4) if v == v else None for v in df["Close"].tolist()],
            "volume":[int(v) if v == v else None for v in df["Volume"].tolist()],
            "indicators": {
                "sma10":  [round(v, 4) if v is not None and v == v else None for v in sma10],
                "sma30":  [round(v, 4) if v is not None and v == v else None for v in sma30],
                "ema9":   [round(v, 4) if v is not None and v == v else None for v in ema9],
                "rsi14":  [round(v, 4) if v is not None and v == v else None for v in rsi14],
                "macd":        [round(v, 4) if v is not None and v == v else None for v in macd_line],
                "macd_signal": [round(v, 4) if v is not None and v == v else None for v in macd_sig],
                "macd_hist":   [round(v, 4) if v is not None and v == v else None for v in macd_hist],
                "bb_upper":  [round(v, 4) if v is not None and v == v else None for v in bb_upper],
                "bb_middle": [round(v, 4) if v is not None and v == v else None for v in bb_middle],
                "bb_lower":  [round(v, 4) if v is not None and v == v else None for v in bb_lower],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
