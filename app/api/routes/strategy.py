from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from collections import defaultdict

from app.config import get_settings
from app.db import SessionLocal
from app.models.signals import Signal
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

    # Fetch current positions once so SELL consensus can size to the actual
    # holding instead of blindly trying to sell 1 share of nothing.
    current_positions: dict[str, float] = {}
    try:
        positions = await broker.get_positions(account_id)
        for pos in positions:
            current_positions[pos.symbol.upper()] = pos.quantity
    except Exception as exc:
        # Can't risk a SELL burst on stale data — log loud and continue.
        # The SELL branch below will skip any symbol not in current_positions.
        import logging
        logging.getLogger(__name__).warning(
            "Could not fetch positions for /strategy/run: %s — all SELLs will be skipped",
            exc,
        )

    for symbol, directions in votes.items():
        for direction, agreeing in directions.items():
            count = len(agreeing)
            consensus_info[(symbol, direction)] = {"count": count, "strategies": agreeing}
            if count >= min_agree:
                if direction == "SELL":
                    held = current_positions.get(symbol.upper(), 0.0)
                    if held < 1.0:
                        orders_placed[(symbol, direction)] = f"skipped_no_position (held={held:.4f})"
                        continue
                    qty = held
                else:
                    qty = 1.0  # manual cycle defaults to 1 share for BUYs
                order_req = OrderRequest(
                    symbol=symbol,
                    side=direction,  # type: ignore[arg-type]
                    order_type="MARKET",
                    quantity=qty,
                    source="scheduler",
                )
                # Stamp a Signal row so Recent Fills can render the strategy name.
                sig_id: int | None = None
                try:
                    with SessionLocal() as db:
                        sig = Signal(
                            strategy_name=("consensus:" + "+".join(agreeing))[:128],
                            symbol=symbol.upper(),
                            direction=direction,
                            strength=1.0,
                            acted_on=True,
                        )
                        db.add(sig)
                        db.commit()
                        db.refresh(sig)
                        sig_id = sig.id
                except Exception:
                    sig_id = None
                order = await svc.execute(order_req, account_id=account_id, signal_id=sig_id)
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
    """Return OHLCV + indicators + fundamentals for charting."""
    import math
    import pandas as pd
    import yfinance as yf

    def _r(v):
        try:
            return round(float(v), 4) if v is not None and not math.isnan(float(v)) else None
        except Exception:
            return None

    def _safe_list(series, n):
        try:
            return [_r(v) for v in series.values.tolist()]
        except Exception:
            return [None] * n

    try:
        symbol = symbol.upper()
        df = get_ohlcv(symbol, period=period)
        n = len(df)
        closes = df["Close"]
        highs  = df["High"]
        lows   = df["Low"]

        # ── Standard indicators ──────────────────────────────────
        sma10 = _safe_list(compute_sma(closes, 10), n)
        sma20 = _safe_list(compute_sma(closes, 20), n)
        sma50 = _safe_list(compute_sma(closes, 50), n)
        sma200= _safe_list(compute_sma(closes, 200), n)
        ema9  = _safe_list(compute_ema(closes, 9), n)
        ema21 = _safe_list(compute_ema(closes, 21), n)
        ema50 = _safe_list(compute_ema(closes, 50), n)
        ema200= _safe_list(compute_ema(closes, 200), n)
        rsi14 = _safe_list(compute_rsi(closes, 14), n)

        try:
            bb = compute_bollinger(closes, 20, 2.0)
            bb_upper  = [_r(v) for v in bb.upper.values.tolist()]
            bb_middle = [_r(v) for v in bb.middle.values.tolist()]
            bb_lower  = [_r(v) for v in bb.lower.values.tolist()]
        except Exception:
            bb_upper = bb_middle = bb_lower = [None] * n

        try:
            macd_result = compute_macd(closes, 12, 26, 9)
            macd_line = [_r(v) for v in macd_result.macd.values.tolist()]
            macd_sig  = [_r(v) for v in macd_result.signal.values.tolist()]
            macd_hist = [_r(v) for v in macd_result.histogram.values.tolist()]
        except Exception:
            macd_line = macd_sig = macd_hist = [None] * n

        # ── ATR ──────────────────────────────────────────────────
        try:
            prev_close = closes.shift(1)
            tr = pd.concat([
                highs - lows,
                (highs - prev_close).abs(),
                (lows  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr14 = [_r(v) for v in tr.ewm(alpha=1/14, adjust=False).mean().values.tolist()]
        except Exception:
            atr14 = [None] * n

        # ── Stochastic %K/%D (14,3) ──────────────────────────────
        try:
            low14  = lows.rolling(14).min()
            high14 = highs.rolling(14).max()
            stoch_k = ((closes - low14) / (high14 - low14) * 100).rolling(3).mean()
            stoch_d = stoch_k.rolling(3).mean()
            stoch_k_list = [_r(v) for v in stoch_k.values.tolist()]
            stoch_d_list = [_r(v) for v in stoch_d.values.tolist()]
        except Exception:
            stoch_k_list = stoch_d_list = [None] * n

        # ── VWAP (daily rolling) ─────────────────────────────────
        try:
            typical = (highs + lows + closes) / 3
            cum_vol = df["Volume"].cumsum()
            cum_tp_vol = (typical * df["Volume"]).cumsum()
            vwap = [_r(v) for v in (cum_tp_vol / cum_vol).values.tolist()]
        except Exception:
            vwap = [None] * n

        # ── OBV ──────────────────────────────────────────────────
        try:
            direction = closes.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (direction * df["Volume"]).cumsum()
            obv_list = [_r(v) for v in obv.values.tolist()]
        except Exception:
            obv_list = [None] * n

        # ── Supertrend (10, 3) ───────────────────────────────────
        try:
            atr_period, factor = 10, 3.0
            prev_c = closes.shift(1)
            tr_s = pd.concat([highs - lows, (highs - prev_c).abs(), (lows - prev_c).abs()], axis=1).max(axis=1)
            atr_s = tr_s.ewm(alpha=1/atr_period, adjust=False).mean()
            hl2   = (highs + lows) / 2
            upper = hl2 + factor * atr_s
            lower = hl2 - factor * atr_s
            st_line = [None] * n
            trend   = [None] * n  # 1=bull, -1=bear
            for i in range(1, n):
                prev_upper = upper.iloc[i-1]
                prev_lower = lower.iloc[i-1]
                upper.iloc[i] = min(upper.iloc[i], prev_upper) if closes.iloc[i-1] > prev_lower else upper.iloc[i]
                lower.iloc[i] = max(lower.iloc[i], prev_lower) if closes.iloc[i-1] < prev_upper else lower.iloc[i]
                if trend[i-1] == -1:
                    trend[i] = 1 if closes.iloc[i] > upper.iloc[i-1] else -1
                else:
                    trend[i] = -1 if closes.iloc[i] < lower.iloc[i-1] else 1
                st_line[i] = _r(lower.iloc[i]) if trend[i] == 1 else _r(upper.iloc[i])
            trend[0] = 1
            st_line[0] = _r(lower.iloc[0])
        except Exception:
            st_line = [None] * n
            trend   = [None] * n

        # ── Fundamentals via yfinance ────────────────────────────
        fundamentals = {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            def _fi(key):
                v = info.get(key)
                try:
                    return round(float(v), 2) if v is not None and not math.isnan(float(v)) else None
                except Exception:
                    return None
            def _fs(key):
                return info.get(key) or None

            fundamentals = {
                "company_name":      _fs("longName") or _fs("shortName"),
                "sector":            _fs("sector"),
                "industry":          _fs("industry"),
                "market_cap":        _fi("marketCap"),
                "pe_ratio":          _fi("trailingPE"),
                "forward_pe":        _fi("forwardPE"),
                "peg_ratio":         _fi("pegRatio"),
                "eps":               _fi("trailingEps"),
                "dividend_yield":    _fi("dividendYield"),
                "beta":              _fi("beta"),
                "52w_high":          _fi("fiftyTwoWeekHigh"),
                "52w_low":           _fi("fiftyTwoWeekLow"),
                "avg_volume":        info.get("averageVolume"),
                "float_shares":      info.get("floatShares"),
                "short_ratio":       _fi("shortRatio"),
                "revenue":           _fi("totalRevenue"),
                "profit_margin":     _fi("profitMargins"),
                "debt_to_equity":    _fi("debtToEquity"),
                "roe":               _fi("returnOnEquity"),
                "analyst_target":    _fi("targetMeanPrice"),
                "analyst_rating":    _fs("recommendationKey"),
            }
        except Exception:
            pass

        dates = [str(d)[:10] for d in df.index]
        return {
            "symbol": symbol,
            "dates":  dates,
            "open":   [_r(v) for v in df["Open"].tolist()],
            "high":   [_r(v) for v in df["High"].tolist()],
            "low":    [_r(v) for v in df["Low"].tolist()],
            "close":  [_r(v) for v in df["Close"].tolist()],
            "volume": [int(v) if v == v else None for v in df["Volume"].tolist()],
            "indicators": {
                "sma10":    sma10,   "sma20":    sma20,
                "sma50":    sma50,   "sma200":   sma200,
                "ema9":     ema9,    "ema21":    ema21,
                "ema50":    ema50,   "ema200":   ema200,
                "rsi14":    rsi14,
                "macd":     macd_line, "macd_signal": macd_sig, "macd_hist": macd_hist,
                "bb_upper": bb_upper,  "bb_middle":   bb_middle, "bb_lower": bb_lower,
                "atr14":    atr14,
                "stoch_k":  stoch_k_list, "stoch_d": stoch_d_list,
                "vwap":     vwap,
                "obv":      obv_list,
                "supertrend": st_line,
                "supertrend_trend": trend,
            },
            "fundamentals": fundamentals,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
