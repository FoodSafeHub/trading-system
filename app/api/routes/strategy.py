from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from collections import defaultdict

from app.config import get_settings
from app.db import SessionLocal
from app.models.assignments import SymbolStrategyAssignment
from app.models.signals import Signal
from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.ema import compute_ema
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma
from app.services.market_data.provider import get_ohlcv, get_price_series
from app.services.markets import is_india_symbol
from app.services.strategy.engine import StrategyEngine, load_strategies_from_config
from app.services.strategy.scheduler import (
    get_scheduler_status,
    run_once as _scheduler_run_once,
    set_scheduler_system_flags,
)

router = APIRouter(prefix="/strategy", tags=["strategy"])
_engine = StrategyEngine()


@router.post("/run")
async def run_strategy_cycle():
    """
    DISCOVERY ONLY. Manually trigger one full strategy cycle with consensus
    counting. Writes signal rows to the DB so Recent Signals updates and the
    "would_fire" preview is accurate. Does NOT place any broker orders.

    The auto-scheduler (`_run_cycle`, every 15 min) is the SOLE execution
    authority for the system. This endpoint is for previewing what the next
    scheduler cycle would do; it never short-circuits that cycle.

    Previously this endpoint placed real orders (tight-trail SELLs and
    hardcoded qty=1 MARKET BUYs) on consensus. That created a second
    execution path that didn't know about per-symbol assignment caps and
    could fire out of band with the scheduler -- the same architectural
    pattern that allowed scanner_service to flatten BNY by surprise. Both
    side-paths are now closed.
    """
    settings = get_settings()
    min_agree = int(settings.min_signal_agreement)
    configs = load_strategies_from_config()

    # Symbols with an enabled per-symbol assignment are evaluated by the
    # scheduler on its assigned-strategy path; we still surface their consensus
    # vote here for visibility, but the scheduler is the only thing that ever
    # turns those into orders. Mirrors scheduler.py.
    with SessionLocal() as db:
        assigned_symbols = {
            sym for (sym,) in db.query(SymbolStrategyAssignment.symbol)
            .filter_by(enabled=True)
            .all()
        }

    # ── Step 1: collect all signals ──────────────────────────
    raw_results = []
    # votes[symbol][direction] = list of strategy names
    votes: dict = defaultdict(lambda: defaultdict(list))

    for config in configs:
        if not config.enabled or config.symbol in assigned_symbols:
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

    # ── Step 2: record consensus-met signals as discovery rows ─────
    # No broker calls. The scheduler picks these up on its next cycle if the
    # symbol is assigned; otherwise they're observability only.
    orders_placed: dict = {}  # (symbol, direction) → status string
    consensus_info: dict = {}  # (symbol, direction) → agreeing strategies

    for symbol, directions in votes.items():
        for direction, agreeing in directions.items():
            count = len(agreeing)
            consensus_info[(symbol, direction)] = {"count": count, "strategies": agreeing}
            if count < min_agree:
                continue

            signal_px = float(
                next(
                    (r["price"] for r in raw_results
                     if r.get("symbol") == symbol and r.get("direction") == direction),
                    0,
                )
            ) or None
            try:
                with SessionLocal() as db:
                    sig = Signal(
                        strategy_name=("consensus:" + "+".join(agreeing))[:128],
                        symbol=symbol.upper(),
                        direction=direction,
                        strength=1.0,
                        price_at_signal=signal_px,
                        # DISCOVERY ONLY -- scheduler decides whether to execute
                        acted_on=False,
                    )
                    db.add(sig)
                    db.commit()
            except Exception:
                pass

            orders_placed[(symbol, direction)] = (
                "discovery_only - scheduler executes on next 15-min cycle"
            )

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
        # Kept for response-shape back-compat. Now means "consensus-met signals
        # recorded as discovery rows", NOT "orders sent to broker".
        "orders_placed": len(orders_placed),
        "execution_authority": "scheduler",
        "note": (
            "Discovery only -- this endpoint records signals; the auto-scheduler "
            "(every 15 min) is the sole order-placement path."
        ),
    }


@router.get("/configs")
def list_configs():
    """Return active strategy configurations."""
    configs = load_strategies_from_config()
    return [{"name": c.name, "symbol": c.symbol, "type": c.type, "enabled": c.enabled} for c in configs]


@router.post("/scheduler/run_now")
def run_scheduler_now(dry_run: bool = True):
    """Trigger one full scheduler cycle synchronously (all assigned symbols + consensus pool).

    dry_run=true  (default) — evaluates every strategy, writes signal rows so
                  Recent Signals updates, but places NO orders and does NOT
                  touch the kill switch. Safe to call while the live scheduler
                  is running.
    dry_run=false — live run; places real orders.
    """
    import threading
    done = threading.Event()
    exc_holder: list[Exception] = []

    def _run():
        try:
            _scheduler_run_once(force=True, dry_run=dry_run)
        except Exception as e:
            exc_holder.append(e)
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    done.wait(timeout=170)
    if exc_holder:
        raise HTTPException(status_code=500, detail=str(exc_holder[0]))
    return {"status": "ok", "dry_run": dry_run, "message": "Scheduler cycle complete"}


@router.get("/scheduler")
def scheduler_status():
    """Return current scheduler state."""
    return get_scheduler_status()


@router.post("/scheduler/config")
def update_scheduler_config(run_bollinger: bool | None = None,
                            run_perplexity: bool | None = None,
                            tape_gate: bool | None = None):
    """
    Toggle which strategy systems participate in the auto-scheduler cycle,
    and the tape-health (knife-veto) BUY gate.
    Changes take effect on the next scheduled cycle — no restart required.
    """
    set_scheduler_system_flags(run_bollinger, run_perplexity, tape_gate)
    return get_scheduler_status()


@router.get("/chart/{symbol}")
def chart_data(symbol: str, period: str = "3mo", interval: str = "1d",
               fundamentals: bool = False):
    """Return OHLCV + indicators (+ fundamentals when requested) for charting.

    `fundamentals=false` (the default) skips the slow yfinance `.info` call —
    the Charts page is technical-only now, so nothing requests it.

    `interval` accepts 1d / 1wk / 1mo / 1h. yfinance handles all four natively;
    Upstox (India) supports 1d/1wk/1h but not 1mo, so monthly India data is
    fetched weekly and resampled to month-end bars here.
    """
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
        # India + monthly: provider's Upstox path has no 1mo unit. Fetch weekly
        # and resample to calendar month-end OHLCV so the chart still works.
        if interval == "1mo" and is_india_symbol(symbol):
            wk = get_ohlcv(symbol, period=period, interval="1wk")
            df = wk.resample("ME").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna(subset=["Close"])
            df.attrs["source"] = wk.attrs.get("source", "upstox")
        else:
            df = get_ohlcv(symbol, period=period, interval=interval)
        n = len(df)
        closes = df["Close"]
        highs  = df["High"]
        lows   = df["Low"]

        # ── Standard indicators ──────────────────────────────────
        # Long-window MAs raise on short periods (e.g. SMA(200) on a 3mo window
        # of ~60 bars). Guard each so a short chart degrades to a flat [None]
        # line instead of failing the whole request with a 400.
        def _ma(fn, length):
            try:
                return _safe_list(fn(closes, length), n)
            except Exception:
                return [None] * n

        sma10 = _ma(compute_sma, 10)
        sma20 = _ma(compute_sma, 20)
        sma50 = _ma(compute_sma, 50)
        sma200= _ma(compute_sma, 200)
        ema9  = _ma(compute_ema, 9)
        ema21 = _ma(compute_ema, 21)
        ema50 = _ma(compute_ema, 50)
        ema200= _ma(compute_ema, 200)
        rsi14 = _ma(compute_rsi, 14)

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

        # ── AlphaTrend (Kivanc Ozbilgic, AP=14, coeff=1) ─────────
        # ATR(14, SMA) band ratcheted by a momentum filter: MFI(14) when the
        # symbol has volume, RSI(14) otherwise (indices/FX). The classic
        # buy/sell read is the AlphaTrend line crossing its own 2-bar lag.
        try:
            _ap, _coeff = 14, 1.0
            prev_c_at = closes.shift(1)
            tr_at = pd.concat([highs - lows, (highs - prev_c_at).abs(),
                               (lows - prev_c_at).abs()], axis=1).max(axis=1)
            atr_at = tr_at.rolling(_ap).mean()
            vols = df["Volume"].fillna(0)
            if float(vols.sum()) > 0:
                tp_at = (highs + lows + closes) / 3
                mf = tp_at * vols
                pos_mf = mf.where(tp_at > tp_at.shift(1), 0.0).rolling(_ap).sum()
                neg_mf = mf.where(tp_at < tp_at.shift(1), 0.0).rolling(_ap).sum()
                mom = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, math.nan))
            else:
                mom = compute_rsi(closes, _ap)
            up_t = lows - atr_at * _coeff
            dn_t = highs + atr_at * _coeff
            at_vals: list[float] = [math.nan] * n
            for i in range(n):
                prev_at = at_vals[i - 1] if i > 0 else math.nan
                a, m_ = float(atr_at.iloc[i]), float(mom.iloc[i]) if mom.iloc[i] == mom.iloc[i] else math.nan
                if math.isnan(a) or math.isnan(m_):
                    at_vals[i] = prev_at
                    continue
                if m_ >= 50:
                    cand = float(up_t.iloc[i])
                    at_vals[i] = cand if math.isnan(prev_at) else max(cand, prev_at)
                else:
                    cand = float(dn_t.iloc[i])
                    at_vals[i] = cand if math.isnan(prev_at) else min(cand, prev_at)
            alphatrend = [_r(v) for v in at_vals]
            # 2-bar lag of the same line — the classic AlphaTrend signal line.
            alphatrend_sig = [None, None] + alphatrend[:-2] if n > 2 else [None] * n
            # 1 = bullish, -1 = bearish; a flat ratchet (line == lag) keeps the
            # previous direction so the coloured line doesn't blink to neutral.
            at_trend: list[int | None] = [None] * n
            for i in range(n):
                if alphatrend[i] is None or alphatrend_sig[i] is None:
                    continue
                if alphatrend[i] > alphatrend_sig[i]:
                    at_trend[i] = 1
                elif alphatrend[i] < alphatrend_sig[i]:
                    at_trend[i] = -1
                else:
                    at_trend[i] = at_trend[i - 1] if i > 0 else None
        except Exception:
            alphatrend = alphatrend_sig = [None] * n
            at_trend = [None] * n

        # ── Fundamentals via yfinance (opt-in) ───────────────────
        fundamentals_out = {}
        if fundamentals:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
            except Exception:
                info = {}
            def _fi(key):
                v = info.get(key)
                try:
                    return round(float(v), 2) if v is not None and not math.isnan(float(v)) else None
                except Exception:
                    return None
            def _fs(key):
                return info.get(key) or None

            fundamentals_out = {
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

        dates = [str(d)[:10] for d in df.index]

        # ── Technical buy/sell signal engine ─────────────────────
        # Derived purely from the indicator arrays above, so chart markers
        # always agree with the plotted lines. `kind` keys let the UI toggle
        # each layer independently.
        closes_l = [_r(v) for v in df["Close"].tolist()]
        tech_signals: list[dict] = []

        def _sig(i, side, kind, code, why):
            tech_signals.append({
                "date": dates[i], "side": side, "kind": kind,
                "label": code, "price": closes_l[i], "reason": why,
            })

        def _cross(a, b, i):
            """1 = a crossed above b at bar i, -1 = crossed below, 0 = none."""
            if None in (a[i], b[i], a[i-1], b[i-1]):
                return 0
            if a[i] > b[i] and a[i-1] <= b[i-1]:
                return 1
            if a[i] < b[i] and a[i-1] >= b[i-1]:
                return -1
            return 0

        for i in range(1, n):
            c = _cross(alphatrend, alphatrend_sig, i)
            if c:
                _sig(i, "BUY" if c > 0 else "SELL", "alphatrend", "AT",
                     "AlphaTrend crossed " + ("above" if c > 0 else "below") + " its 2-bar lag")
            if trend[i] is not None and trend[i-1] is not None and trend[i] != trend[i-1]:
                _sig(i, "BUY" if trend[i] == 1 else "SELL", "supertrend", "ST",
                     f"Supertrend flipped {'bullish' if trend[i] == 1 else 'bearish'}")
            c = _cross(ema9, ema21, i)
            if c:
                _sig(i, "BUY" if c > 0 else "SELL", "ema_cross", "EMA",
                     f"EMA 9 crossed {'above' if c > 0 else 'below'} EMA 21")
            c = _cross(macd_line, macd_sig, i)
            if c:
                _sig(i, "BUY" if c > 0 else "SELL", "macd_cross", "MACD",
                     f"MACD crossed {'above' if c > 0 else 'below'} signal line")
            if None not in (rsi14[i], rsi14[i-1]):
                if rsi14[i-1] < 30 <= rsi14[i]:
                    _sig(i, "BUY", "rsi", "RSI", f"RSI reclaimed 30 ({rsi14[i]:.0f})")
                elif rsi14[i-1] > 70 >= rsi14[i]:
                    _sig(i, "SELL", "rsi", "RSI", f"RSI lost 70 ({rsi14[i]:.0f})")

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
                "alphatrend": alphatrend,
                "alphatrend_signal": alphatrend_sig,
                "alphatrend_trend": at_trend,
            },
            "tech_signals": tech_signals,
            "fundamentals": fundamentals_out,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
