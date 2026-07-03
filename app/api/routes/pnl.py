from __future__ import annotations

"""Realized + unrealized P/L endpoints.

All endpoints compute on the fly off the orders table — no persisted P/L
state, no caching. Cost is one orders query + one signals query per call;
fine for the ~10s of K orders this system is sized for. If that ever
becomes a hot path we'll add a persisted realized_trades table (Pass 2).
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_serializer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.schemas._serializers import serialize_et
from app.services.brokers.factory import get_broker
from app.services.pnl.aggregate import (
    by_strategy,
    by_symbol,
    equity_curve,
    open_position_pnl,
    summarize,
)
from app.services.pnl.fifo import compute_fifo
from app.services.pnl.store import load_closed_trades, sync_realized_trades

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl", tags=["pnl"])


# ── Response models ────────────────────────────────────────────────────────

class GroupSummaryOut(BaseModel):
    key: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    total_realized_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float]
    best_trade: float
    worst_trade: float
    avg_hold_days: float


class ClosedTradeOut(BaseModel):
    symbol: str
    quantity: float
    buy_price: float
    sell_price: float
    buy_at: datetime
    sell_at: datetime
    realized_pnl: float
    realized_pct: float
    hold_days: float
    broker: str
    buy_order_id: int
    sell_order_id: int
    buy_strategy: Optional[str]
    sell_strategy: Optional[str]
    is_paper: bool
    # Trailing stop audit fields — populated when the exit was via a trailing stop.
    signal_price: Optional[float] = None    # price when SELL signal fired (from Signal row)
    signal_at: Optional[datetime] = None    # timestamp of the SELL signal
    trail_pct: Optional[float] = None       # trail % placed at signal time (from Order row)
    trail_captured_pct: Optional[float] = None  # (sell_price - signal_price) / signal_price * 100
    exit_type: Optional[str] = None         # "trail" | "market" | "unknown"
    # Approach C peak-capture audit: the durable high-water mark reached after
    # the signal (from trail_peaks), and how much of the available run-up
    # (signal → peak) the trail actually kept.
    peak_price: Optional[float] = None
    capture_efficiency_pct: Optional[float] = None  # (exit-signal)/(peak-signal)*100

    @field_serializer("buy_at", "sell_at", "signal_at")
    def _ser_ts(self, dt: datetime | None) -> str | None:
        return serialize_et(dt) if dt else None


class EquityPointOut(BaseModel):
    at: datetime
    realized_pnl: float
    peak_pnl: float
    drawdown: float       # peak_pnl - realized_pnl, always >= 0
    drawdown_pct: float | None  # drawdown / peak_pnl when peak > 0, else None

    @field_serializer("at")
    def _ser_at(self, dt: datetime) -> str | None:
        return serialize_et(dt)


class OpenPositionOut(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    last_price: Optional[float]
    market_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pct: Optional[float]
    broker: str
    is_paper: bool


class SummaryOut(BaseModel):
    realized: GroupSummaryOut
    total_unrealized_pnl: float
    open_position_count: int
    closed_trade_count: int
    has_live_prices: bool


class OpenTrailOut(BaseModel):
    """An OPEN position whose tight-trail SELL stop is currently armed.

    The strategy already fired a SELL signal and the scheduler placed a
    protective STOP / TRAILING_STOP, but the position has not exited yet.
    This lets the user watch — in real time — whether the trail is letting
    them ride extra upside above the signal price, or sitting below it.
    """
    symbol: str
    quantity: float
    avg_cost: float
    last_price: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pct: Optional[float]
    broker: str
    is_paper: bool

    # Armed trail details (joined from the resting SELL Order + its Signal)
    signal_price: Optional[float]      # price when the SELL signal fired
    signal_at: Optional[datetime]      # when it fired
    signal_strategy: Optional[str]     # which strategy fired it
    order_type: Optional[str]          # "STOP" (floor) | "TRAILING_STOP" (native %)
    stop_price: Optional[float]        # current stop trigger (STOP orders only)
    trail_pct: Optional[float]         # trail width % (TRAILING_STOP / config)
    armed_at: Optional[datetime]       # when the trail order was submitted
    days_armed: Optional[float]        # days since the trail was armed

    # Computed live status
    move_since_signal_pct: Optional[float]  # (last - signal) / signal × 100
    trail_helping: Optional[bool]           # True if last_price > signal_price now

    # Estimated trail trigger when no order is resting yet (SIGNAL_ONLY): the
    # level the trail WOULD sit at = max(signal floor, last × (1 - trail_pct)).
    # Floor = signal × (1 + 0.25% buffer). Lets the user see the protective
    # level before the reconciliation actually arms the broker order.
    est_trail_trigger: Optional[float] = None

    # Durable high-water mark the trail ratchets off (from trail_peaks). This is
    # the authoritative peak since the SELL signal — the stop holds at
    # peak × (1 - trail_pct%) and does NOT drop when price pulls back below it.
    peak_price: Optional[float] = None
    peak_at: Optional[datetime] = None


# ── Helpers ────────────────────────────────────────────────────────────────

_last_broker_sync_ts: float = 0.0
# On-read reconcile throttle. One PnL page load fires ~7 endpoints over
# 10-60s, so a 10s throttle re-paid the full broker order-listing fan-out
# several times per page view. 45s means at most one sync per page load
# while a broker-side close still reflects within a minute.
_BROKER_SYNC_THROTTLE_SECONDS = 45.0

# ── Fan-out caches ───────────────────────────────────────────────────────────
# Broker positions / working stops / quotes are each needed by several PnL
# endpoints that the dashboard fires back-to-back on one page load. Without a
# cache, EVERY endpoint re-authenticated and re-fetched from every broker
# serially (summary 15s, open-trails 22s measured). A short TTL makes one page
# load pay each fan-out once. A failing broker (expired Zerodha token, Webull
# 429) is negative-cached for 5 min so it stops adding a doomed round-trip to
# every call.
import threading as _threading
import time as _time_mod

_FANOUT_TTL_SECONDS = 15.0
_BROKER_DOWN_SECONDS = 300.0
_fanout_cache: dict = {}
_fanout_lock = _threading.Lock()
_broker_down_until: dict[str, float] = {}


def _cache_get(key):
    with _fanout_lock:
        hit = _fanout_cache.get(key)
        if hit and hit[0] > _time_mod.monotonic():
            return hit[1]
    return None


def _cache_put(key, value) -> None:
    with _fanout_lock:
        _fanout_cache[key] = (_time_mod.monotonic() + _FANOUT_TTL_SECONDS, value)


def _broker_usable(broker) -> bool:
    name = (getattr(broker, "name", "") or "?").lower()
    return _broker_down_until.get(name, 0.0) <= _time_mod.monotonic()


def _mark_broker_down(broker) -> None:
    name = (getattr(broker, "name", "") or "?").lower()
    _broker_down_until[name] = _time_mod.monotonic() + _BROKER_DOWN_SECONDS
    logger.info("[pnl] broker %s marked down for %.0fs (skipping in fan-outs)",
                name, _BROKER_DOWN_SECONDS)


def _map_brokers_parallel(brokers: list, fn) -> list:
    """Run fn(broker) for each usable broker concurrently; collect non-None
    results in broker order. fn must swallow-and-return-None on failure."""
    import concurrent.futures as _fut
    usable = [b for b in brokers if _broker_usable(b)]
    if not usable:
        return []
    if len(usable) == 1:
        r = fn(usable[0])
        return [r] if r is not None else []
    with _fut.ThreadPoolExecutor(max_workers=len(usable)) as ex:
        results = list(ex.map(fn, usable))
    return [r for r in results if r is not None]


def _maybe_sync_broker_orders() -> None:
    """Pull fresh broker fills into the orders table, throttled.

    PnL/Recent-Fills read from filled Order rows, so a close that happened at
    the broker (native trailing-stop fill, manual close, an India order) only
    shows up once its fill is reconciled into the DB. The 15-min scheduler job
    does this in the background, but viewing the page should reflect reality
    NOW — so we run the reconcile on read, throttled to at most once per
    _BROKER_SYNC_THROTTLE_SECONDS so rapid page refreshes don't hammer the
    broker. Best-effort: any failure is swallowed (the page still renders the
    last-known state).
    """
    global _last_broker_sync_ts
    import time as _time
    now = _time.monotonic()
    if now - _last_broker_sync_ts < _BROKER_SYNC_THROTTLE_SECONDS:
        return
    _last_broker_sync_ts = now
    try:
        from app.services.reconciliation.order_sync import sync_broker_orders_once
        sync_broker_orders_once()
    except Exception as exc:
        logger.debug("[pnl] on-read broker order sync skipped: %s", exc)


def _refresh(db: Session):
    """Sync new round-trips into realized_trades, then return (closed, fifo).

    The FifoResult is reused for open lots (no separate query). The closed list
    is the canonical persisted history — same numbers across calls regardless of
    what the live walk would produce.
    """
    # Pull any broker-side fills (closes) into the orders table first, so the
    # round-trip sync below sees freshly-closed positions automatically.
    _maybe_sync_broker_orders()
    _inserted, fifo = sync_realized_trades(db)
    closed = load_closed_trades(db)

    # Drop user-excluded symbols from EVERY PnL view (closed list here; open lots
    # below). For holdings managed outside this system so they don't skew P/L.
    settings = get_settings()
    excluded = settings.pnl_excluded
    if excluded:
        closed = [c for c in closed if (getattr(c, "symbol", "") or "").upper() not in excluded]
        fifo.open_lots = [l for l in fifo.open_lots if (l.symbol or "").upper() not in excluded]
    # Paper fills (tests, manual dry-runs) are not real money — without this a
    # paper BUY lingers forever as a phantom "open position" on the PnL page.
    if settings.pnl_exclude_paper:
        closed = [c for c in closed if not getattr(c, "is_paper", False)]
        fifo.open_lots = [l for l in fifo.open_lots if not l.is_paper]
    return closed, fifo


def _resolve_last_prices(symbols: list[str]) -> dict[str, float]:
    """Pull last/ask/bid for the given symbols across all position-holding brokers.

    Returns {} on total failure — the unrealized columns will surface as None
    rather than 0, so the UI can tell "we don't know" from "zero". India symbols
    only quote on Zerodha, so we query every position broker and take the first
    price found per symbol (stopping once all symbols are resolved).
    """
    if not symbols:
        return {}
    cache_key = ("last_prices", tuple(sorted(s.upper() for s in symbols)))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    import asyncio
    from app.services.brokers.factory import get_position_brokers

    def _one(broker):
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(broker.authenticate())
                return loop.run_until_complete(broker.get_quotes(list(symbols)))
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("[pnl] %s quote fetch failed: %s", getattr(broker, "name", "?"), exc)
            _mark_broker_down(broker)
            return None

    out: dict[str, float] = {}
    for quotes in _map_brokers_parallel(get_position_brokers(), _one):
        for sym, q in (quotes or {}).items():
            price = getattr(q, "last", None) or getattr(q, "ask", None) or getattr(q, "bid", None)
            if price and price > 0:
                out.setdefault(sym.upper(), float(price))
    _cache_put(cache_key, out)
    return out


def _naive(dt):
    """Strip tzinfo for safe comparison between naive and aware datetimes."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def _peak_since(symbol: str, since_dt) -> float:
    """Highest traded price for `symbol` since `since_dt` (the SELL signal time).

    Used to estimate where a trailing stop WOULD sit before it's armed. A real
    trailing stop ratchets off the PEAK since arming — never the current price —
    so the estimate must use the high-water mark, not the live quote. Returns
    0.0 on any failure (caller falls back to current price / floor).
    """
    try:
        from app.services.market_data.provider import get_ohlcv
        df = get_ohlcv(symbol, period="3mo")
        if df is None or df.empty:
            return 0.0
        col = "High" if "High" in df.columns else "Close"
        if since_dt is not None:
            try:
                sliced = df.loc[str(_naive(since_dt))[:10]:]
                if not sliced.empty:
                    return float(sliced[col].max())
            except Exception:
                pass
        return float(df[col].max())
    except Exception:
        return 0.0


def _broker_resting_sell_stops() -> dict[str, dict]:
    """Query EVERY position-holding broker for WORKING SELL STOP/TRAILING_STOP orders.

    The scheduler places these on the broker; they may not all be mirrored in
    the local DB (e.g. scanner-path SELLs). This is the source of truth for
    "what trail is actually resting right now". Enumerates all brokers that could
    hold a position (global route + per-assignment overrides — Webull, Zerodha)
    so a trail resting on a non-default broker isn't reported as NOT ARMED.
    Returns {symbol: {...}}.
    """
    cached = _cache_get("resting_stops")
    if cached is not None:
        return cached

    import asyncio
    from app.services.brokers.factory import get_position_brokers

    def _one(broker):
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(broker.authenticate())
                accts = loop.run_until_complete(broker.get_accounts())
                acct_id = accts[0].account_id if accts else ""
                return loop.run_until_complete(broker.list_orders(acct_id, status="working")) or []
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("[pnl] %s working-orders fetch failed: %s", getattr(broker, "name", "?"), exc)
            _mark_broker_down(broker)
            return None

    result: dict[str, dict] = {}
    for orders in _map_brokers_parallel(get_position_brokers(), _one):
        for o in orders or []:
            side = (getattr(o, "side", "") or "").upper()
            otype = (getattr(o, "order_type", "") or "").upper()
            sym = (getattr(o, "symbol", "") or "").upper()
            if side != "SELL" or otype not in ("STOP", "TRAILING_STOP") or not sym:
                continue
            if sym in result:
                continue  # keep first (broker returns newest-ish; good enough)
            raw = getattr(o, "raw", {}) or {}
            result[sym] = {
                "order_type": otype,
                "stop_price": getattr(o, "stop_price", None) or raw.get("stopPrice"),
                "trail_value": getattr(o, "trail_value", None) or raw.get("stopPriceOffset"),
                "broker_order_id": getattr(o, "broker_order_id", None),
            }
    _cache_put("resting_stops", result)
    return result


def _broker_positions() -> dict[str, dict]:
    """Query EVERY position-holding broker for currently-held LONG positions.

    The FIFO ledger only knows positions the BOT opened (local order history).
    Manual/external buys (placed directly in the broker app) never get a local
    BUY row, so they're invisible to the ledger — but they're real holdings the
    assigned strategy still trades. This is the source of truth for "what do we
    actually hold". Enumerates all brokers that could hold a position (global
    route + per-assignment overrides — Webull, Zerodha) so a symbol held on a
    non-default broker isn't silently omitted. Returns {symbol: {quantity,
    avg_cost, broker}}.
    """
    cached = _cache_get("broker_positions")
    if cached is not None:
        return cached

    import asyncio
    from app.services.brokers.factory import get_position_brokers

    settings = get_settings()
    excluded = settings.pnl_excluded
    brokers = [
        b for b in get_position_brokers()
        if not (settings.pnl_exclude_paper and (getattr(b, "name", "") or "").lower() == "paper")
    ]

    def _one(broker):
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(broker.authenticate())
                accts = loop.run_until_complete(broker.get_accounts())
                acct_id = accts[0].account_id if accts else ""
                return loop.run_until_complete(broker.get_positions(acct_id)) or []
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("[pnl] %s positions fetch failed: %s", getattr(broker, "name", "?"), exc)
            _mark_broker_down(broker)
            return None

    result: dict[str, dict] = {}
    for positions in _map_brokers_parallel(brokers, _one):
        for p in positions or []:
            sym = (getattr(p, "symbol", "") or "").upper()
            qty = getattr(p, "quantity", 0) or 0
            if not sym or qty <= 0 or sym in excluded or sym in result:
                continue
            result[sym] = {
                "quantity": float(qty),
                "avg_cost": getattr(p, "average_cost", None),
                "broker": getattr(p, "broker", None),
            }
    _cache_put("broker_positions", result)
    return result


def _merged_open_positions(fifo) -> tuple[list[dict], float]:
    """Open-position rows merged from the FIFO ledger AND live broker positions.

    Single source of truth for "what is open right now" so /summary's count and
    /open-positions' table never disagree. FIFO rows keep their cost basis;
    broker-only holdings (MANUAL/external buys with no local BUY row) get a
    synthesized row — broker avg_cost when reported, else 0.0 — so they're at
    least visible with live market value. Returns (rows, total_unrealized).
    """
    broker_positions = _broker_positions()
    fifo_symbols = {l.symbol.upper() for l in fifo.open_lots}
    all_held = sorted(fifo_symbols | set(broker_positions.keys()))
    if not all_held:
        return [], 0.0

    last_prices = _resolve_last_prices(all_held)
    rows, total_unrealized = open_position_pnl(fifo.open_lots, last_prices)
    open_by_symbol = {r["symbol"].upper(): r for r in rows}

    for sym, bp in broker_positions.items():
        if sym in open_by_symbol:
            continue  # FIFO row already has richer cost-basis data
        last_px = last_prices.get(sym)
        avg_cost = bp.get("avg_cost")
        qty = bp["quantity"]
        mkt_val = round(last_px * qty, 2) if last_px else None
        upnl = upnl_pct = None
        if avg_cost and last_px:
            upnl = round((last_px - float(avg_cost)) * qty, 2)
            upnl_pct = round((last_px - float(avg_cost)) / float(avg_cost) * 100, 2)
            total_unrealized += upnl
        open_by_symbol[sym] = {
            "symbol": sym,
            "quantity": qty,
            "avg_cost": float(avg_cost) if avg_cost else 0.0,
            "last_price": last_px,
            "market_value": mkt_val,
            "unrealized_pnl": upnl,
            "unrealized_pct": upnl_pct,
            "broker": bp.get("broker") or "",
            "is_paper": False,
        }

    return list(open_by_symbol.values()), total_unrealized


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/summary", response_model=SummaryOut)
def pnl_summary(
    include_unrealized: bool = True,
    db: Session = Depends(get_db),
):
    """Top-line numbers: realized so far, unrealized on open positions, counts.

    Open-position count + unrealized P/L include MANUAL/external broker buys
    (not just FIFO-ledger positions), so the headline matches the open-positions
    table. include_unrealized=False keeps the legacy FIFO-only, no-quote path.
    """
    closed, fifo = _refresh(db)
    realized = summarize(closed, key="ALL")
    if include_unrealized:
        open_rows, total_unrealized = _merged_open_positions(fifo)
    else:
        open_rows, total_unrealized = open_position_pnl(fifo.open_lots, {})
    has_live = any(r.get("last_price") for r in open_rows)
    return SummaryOut(
        realized=GroupSummaryOut(**asdict(realized)),
        total_unrealized_pnl=total_unrealized,
        open_position_count=len(open_rows),
        closed_trade_count=len(closed),
        has_live_prices=has_live,
    )


@router.get("/by-symbol", response_model=List[GroupSummaryOut])
def pnl_by_symbol(db: Session = Depends(get_db)):
    """Realized P/L grouped by symbol, best to worst."""
    closed, _ = _refresh(db)
    return [GroupSummaryOut(**asdict(g)) for g in by_symbol(closed)]


@router.get("/by-strategy", response_model=List[GroupSummaryOut])
def pnl_by_strategy(db: Session = Depends(get_db)):
    """Realized P/L grouped by the BUY-side strategy attribution."""
    closed, _ = _refresh(db)
    return [GroupSummaryOut(**asdict(g)) for g in by_strategy(closed)]


@router.get("/closed-trades", response_model=List[ClosedTradeOut])
def pnl_closed_trades(
    symbol: str | None = None,
    strategy: str | None = None,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    """The realized round-trip log. Most recent SELL first.

    For trailing-stop exits, also returns the Approach C peak-capture audit
    fields: signal_price (price when the SELL signal fired), signal_at, trail_pct,
    peak_price (the high-water mark the trail ratcheted off, from trail_peaks),
    capture_efficiency_pct ((exit-signal)/(peak-signal) — share of the available
    run-up the trail kept), trail_captured_pct, and a normalised exit_type
    ("trail" for a bot STOP / native TRAILING_STOP, "market" otherwise).
    """
    if limit < 1 or limit > 5000:
        raise HTTPException(400, "limit must be between 1 and 5000")
    closed, _ = _refresh(db)
    rows = closed
    if symbol:
        sym = symbol.upper().strip()
        rows = [t for t in rows if t.symbol == sym]
    if strategy:
        rows = [t for t in rows if (t.buy_strategy or "(unattributed)") == strategy]
    rows = sorted(rows, key=lambda t: t.sell_at, reverse=True)[:limit]

    # Build a lookup: sell_order_id → (signal_price, signal_at, trail_pct, exit_type)
    # by joining Order → Signal for every sell order in this result set.
    from app.models.orders import Order
    from app.models.signals import Signal

    sell_ids = [t.sell_order_id for t in rows if t.sell_order_id]
    trail_info: dict[int, dict] = {}
    # Durable peaks keyed by signal_id (what the trail ratcheted off — Approach C).
    peak_by_signal: dict[int, float] = {}
    if sell_ids:
        sell_orders = db.query(Order).filter(Order.id.in_(sell_ids)).all()
        sig_ids = [o.signal_id for o in sell_orders if o.signal_id]
        signals_by_id: dict[int, Signal] = {}
        if sig_ids:
            for sig in db.query(Signal).filter(Signal.id.in_(sig_ids)).all():
                signals_by_id[sig.id] = sig
            try:
                from app.models.trail_peaks import TrailPeak
                for tp in db.query(TrailPeak).filter(TrailPeak.signal_id.in_(sig_ids)).all():
                    if tp.signal_id is not None and tp.peak_price:
                        # Keep the highest if multiple rows exist for a signal.
                        peak_by_signal[tp.signal_id] = max(
                            peak_by_signal.get(tp.signal_id, 0.0), float(tp.peak_price)
                        )
            except Exception:
                peak_by_signal = {}

        for o in sell_orders:
            sig = signals_by_id.get(o.signal_id) if o.signal_id else None
            # Normalise exit_type: the Approach C trail rests as a bot-managed
            # STOP or a native TRAILING_STOP — both are "trail" exits. A plain
            # MARKET fill (or a reconciled broker close) is a "market" exit.
            otype = (o.order_type or "").upper()
            if otype in ("STOP", "TRAILING_STOP"):
                exit_type = "trail"
            elif otype == "MARKET":
                exit_type = "market"
            else:
                exit_type = otype.lower() or "unknown"
            trail_info[o.id] = {
                "signal_price": float(sig.price_at_signal) if sig and sig.price_at_signal else None,
                "signal_at":    sig.created_at if sig else None,
                "trail_pct":    float(o.trail_value) if getattr(o, "trail_value", None) else None,
                "exit_type":    exit_type,
                "peak_price":   peak_by_signal.get(o.signal_id) if o.signal_id else None,
            }

    result = []
    for t in rows:
        info = trail_info.get(t.sell_order_id, {})
        sp = info.get("signal_price")
        peak = info.get("peak_price")
        exit_p = t.sell_price
        # trail_captured_pct: how much extra (%) the stock moved from signal → exit.
        trail_captured = None
        if sp and sp > 0 and exit_p:
            trail_captured = round((exit_p - sp) / sp * 100, 2)
        # capture_efficiency_pct: of the run-up that was actually AVAILABLE after
        # the signal (signal → peak), how much did the trail keep? 100% = exited
        # at the peak; 0% = exited at the signal; <0% = exited below the signal.
        # Only meaningful when the peak rose above the signal.
        capture_eff = None
        if sp and peak and exit_p and (peak - sp) > 1e-9:
            capture_eff = round((exit_p - sp) / (peak - sp) * 100, 1)
        result.append(ClosedTradeOut(
            symbol=t.symbol, quantity=t.quantity, buy_price=t.buy_price,
            sell_price=t.sell_price, buy_at=t.buy_at, sell_at=t.sell_at,
            realized_pnl=t.realized_pnl, realized_pct=t.realized_pct,
            hold_days=t.hold_days, broker=t.broker,
            buy_order_id=t.buy_order_id, sell_order_id=t.sell_order_id,
            buy_strategy=t.buy_strategy, sell_strategy=t.sell_strategy,
            is_paper=t.is_paper,
            signal_price=sp,
            signal_at=info.get("signal_at"),
            trail_pct=info.get("trail_pct"),
            trail_captured_pct=trail_captured,
            exit_type=info.get("exit_type"),
            peak_price=peak,
            capture_efficiency_pct=capture_eff,
        ))
    return result


@router.get("/equity-curve", response_model=List[EquityPointOut])
def pnl_equity_curve(bucket: str = "trade", db: Session = Depends(get_db)):
    """Cumulative realized P/L over time + running drawdown.

    `bucket=trade` (default) emits one point per closed trade.
    `bucket=day` collapses to one point per calendar day (last close of each
    day) — cleaner for long histories.
    """
    if bucket not in {"trade", "day"}:
        raise HTTPException(400, "bucket must be 'trade' or 'day'")
    closed, _ = _refresh(db)
    points = equity_curve(closed)

    if bucket == "day" and points:
        by_day: dict = {}
        for p in points:
            key = p.at.date()
            by_day[key] = p  # last point of each day wins (curve is monotonic in time)
        points = sorted(by_day.values(), key=lambda p: p.at)

    out: list[EquityPointOut] = []
    peak = 0.0
    for p in points:
        if p.realized_pnl > peak:
            peak = p.realized_pnl
        dd = max(peak - p.realized_pnl, 0.0)
        dd_pct = (dd / peak) if peak > 0 else None
        out.append(EquityPointOut(
            at=p.at,
            realized_pnl=p.realized_pnl,
            peak_pnl=peak,
            drawdown=dd,
            drawdown_pct=dd_pct,
        ))
    return out


@router.post("/reconcile")
def pnl_reconcile(lookback_days: int = 30):
    """Force a broker→DB fill reconcile, widening the lookback window.

    The routine 15-min job (and the on-read sync) only look back
    settings.order_sync_lookback_days (default 7). A manual close older than
    that — never ingested while the app was down, say — stays invisible to
    realized P/L. Hit this with a larger window (default 30d, max 365) to pull
    those orphan fills in and materialize the round-trips. Returns the sync
    summary (orders updated, orphan fills ingested, round-trips materialized).
    """
    lookback_days = max(1, min(int(lookback_days), 365))
    from app.services.reconciliation.order_sync import sync_broker_orders_once
    result = sync_broker_orders_once(lookback_days=lookback_days)
    result["lookback_days"] = lookback_days
    return result


@router.get("/open-positions", response_model=List[OpenPositionOut])
def pnl_open_positions(db: Session = Depends(get_db)):
    """Open long lots aggregated per symbol, with unrealized P/L vs last quote.

    Held positions come from TWO sources (see _merged_open_positions): the FIFO
    ledger AND live broker positions, so MANUAL/external buys placed directly in
    the broker app are not silently omitted.
    """
    _, fifo = _refresh(db)
    rows, _total = _merged_open_positions(fifo)
    return [OpenPositionOut(**r) for r in rows]


@router.get("/open-trails", response_model=List[OpenTrailOut])
def pnl_open_trails(db: Session = Depends(get_db)):
    """Open positions whose tight-trail SELL stop is currently armed.

    Live counterpart to the closed-trade Trail Stop Audit: the strategy fired
    a SELL signal and the scheduler armed a protective STOP / TRAILING_STOP,
    but the position has NOT exited yet. Shows whether the trail is currently
    letting the position ride above the signal price (helping) or sitting
    below it (hurting), so the user can track the in-progress effect.
    """
    from app.models.orders import Order
    from app.models.signals import Signal

    _, fifo = _refresh(db)

    # Source of held positions is TWO-fold:
    #   1. FIFO ledger (positions the bot opened) — carries cost basis / P&L.
    #   2. Live broker positions — catches MANUAL/external buys the ledger never
    #      saw. Without this, a held symbol with a SELL signal but no local BUY
    #      (e.g. COLL bought in the broker app) is silently absent from this
    #      table even though the assigned strategy is actively trading it.
    broker_positions = _broker_positions()

    fifo_symbols = {l.symbol.upper() for l in fifo.open_lots}
    all_held = sorted(fifo_symbols | set(broker_positions.keys()))
    if not all_held:
        return []

    last_prices = _resolve_last_prices(all_held)
    rows, _total = open_position_pnl(fifo.open_lots, last_prices)
    open_by_symbol = {r["symbol"].upper(): r for r in rows}

    # Synthesize a position row for broker-only holdings (no FIFO cost basis).
    for sym, bp in broker_positions.items():
        if sym in open_by_symbol:
            continue  # FIFO row already has richer cost-basis data
        last_px = last_prices.get(sym)
        avg_cost = bp.get("avg_cost")
        upnl = upnl_pct = None
        if avg_cost and last_px:
            upnl = round((last_px - float(avg_cost)) * bp["quantity"], 2)
            upnl_pct = round((last_px - float(avg_cost)) / float(avg_cost) * 100, 2)
        open_by_symbol[sym] = {
            "symbol": sym,
            "quantity": bp["quantity"],
            "avg_cost": avg_cost,
            "last_price": last_px,
            "unrealized_pnl": upnl,
            "unrealized_pct": upnl_pct,
            "broker": bp.get("broker"),
            "is_paper": False,
        }

    symbols = list(open_by_symbol.keys())

    # Earliest open (buy) time per held symbol — the SELL-signal price we anchor
    # to is the FIRST SELL fired AT/AFTER the position was opened, not the most
    # recent one. That first signal is when the strategy decided to exit and is
    # the price the trail should be measured against.
    earliest_buy: dict[str, datetime] = {}
    for lot in fifo.open_lots:
        cur = earliest_buy.get(lot.symbol)
        if cur is None or lot.buy_at < cur:
            earliest_buy[lot.symbol] = lot.buy_at

    # ── Source 1: local DB resting SELL stops (preferred — carries signal link)
    db_resting = (
        db.query(Order)
        .filter(
            Order.symbol.in_(symbols),
            Order.side == "SELL",
            Order.order_type.in_(["STOP", "TRAILING_STOP"]),
            Order.status.in_(["submitted", "working"]),
        )
        .order_by(Order.created_at.desc())
        .all()
    )
    db_by_symbol: dict[str, Order] = {}
    for o in db_resting:
        db_by_symbol.setdefault(o.symbol, o)

    # ── Source 2: broker working SELL stops (catches trails not in local DB,
    # e.g. scanner-path SELLs). Source of truth for what's actually resting.
    broker_stops = _broker_resting_sell_stops()

    # ── Anchor signal price: the FIRST SELL signal fired AT/AFTER the position
    # was opened, per held symbol. This is the price the strategy first flagged
    # the exit at — the price the trail should be measured against (not the most
    # recent re-fire, and NOT a SELL from some other strategy).
    #
    # The qualifying signal MUST come from the symbol's ASSIGNED strategy (the
    # scheduler stores it as the bare name or "scanner:<name>"), and must have
    # fired AFTER the position was acquired (earliest open lot).
    # Shared helper — same anchor logic the scheduler's trail reconcile uses,
    # so the table and the actual armed trails never diverge.
    from app.services.strategy.trail_anchor import (
        assigned_strategy_map, assigned_trail_pct_map, first_assigned_sell_signals,
    )

    assigned_strat = assigned_strategy_map(db, symbols)
    assigned_trail_pct = assigned_trail_pct_map(db, symbols)
    fallback_sig: dict[str, Signal] = first_assigned_sell_signals(
        db, assigned_strat, earliest_buy,
    )

    # Durable high-water marks the trail ratchets off (one per symbol; latest
    # row wins). This is the persisted peak the scheduler advances each cycle —
    # the authoritative value the stop is measured against.
    trail_peak_by_symbol: dict[str, tuple] = {}
    try:
        from app.models.trail_peaks import TrailPeak
        if symbols:
            for tp in (
                db.query(TrailPeak)
                .filter(TrailPeak.symbol.in_([s.upper() for s in symbols]))
                .order_by(TrailPeak.id.asc())
                .all()
            ):
                trail_peak_by_symbol[tp.symbol.upper()] = (tp.peak_price, tp.peak_at)
    except Exception:
        trail_peak_by_symbol = {}

    # Show held positions whose ASSIGNED strategy fired a SELL after acquisition
    # (whether or not a trail is currently resting), PLUS any symbol that already
    # has a resting trail order in the DB or on the broker.
    sell_signal_symbols = set(fallback_sig.keys())
    armed_symbols = set(db_by_symbol) | set(broker_stops) | sell_signal_symbols

    now = datetime.utcnow()
    out: list[OpenTrailOut] = []
    for sym in armed_symbols:
        pos = open_by_symbol.get(sym)
        if not pos:
            continue

        o = db_by_symbol.get(sym)
        bstop = broker_stops.get(sym)

        # Trail order details: prefer the DB row, then the broker order, else
        # signal-only (SELL fired but no resting trail detected yet).
        if o is not None:
            order_type = o.order_type
            stop_price = float(o.stop_price) if o.stop_price else None
            trail_pct  = float(o.trail_value) if getattr(o, "trail_value", None) else None
            armed_at   = o.submitted_at or o.created_at
        elif bstop is not None:
            order_type = bstop.get("order_type")
            try:
                stop_price = float(bstop["stop_price"]) if bstop.get("stop_price") else None
            except Exception:
                stop_price = None
            try:
                trail_pct = float(bstop["trail_value"]) if bstop.get("trail_value") else None
            except Exception:
                trail_pct = None
            armed_at = None
        else:
            # Signal fired but no resting trail order found in DB or on broker.
            order_type = "SIGNAL_ONLY"
            stop_price = trail_pct = armed_at = None

        # Signal price/time/strategy ALWAYS come from the FIRST SELL signal
        # fired after the position opened (the anchor the user cares about),
        # not the most recent re-fire or the order's linked signal.
        sig = fallback_sig.get(sym)
        sig_price = float(sig.price_at_signal) if (sig and sig.price_at_signal) else None

        last_px = pos.get("last_price")
        move_pct = helping = None
        if sig_price and sig_price > 0 and last_px:
            move_pct = round((last_px - sig_price) / sig_price * 100, 2)
            helping = last_px > sig_price

        # Days since the trail was armed, or — when no trail is resting yet
        # (SIGNAL_ONLY) — days since the first SELL signal fired.
        _ref_time = armed_at or (sig.created_at if sig else None)
        days_armed = None
        if _ref_time is not None:
            try:
                days_armed = round((now - _naive(_ref_time)).total_seconds() / 86400, 1)
            except Exception:
                days_armed = None

        # Estimated trail trigger — what the trail WOULD sit at if armed now.
        # A trailing stop RATCHETS off the PEAK since the signal, never the
        # current price, so it can only move UP. Mirrors tighten_trail_on_sell:
        #   floor  = signal × (1 + 0.25%)        (never park below the signal)
        #   native = peak_since_signal × (1 - trail_pct%)
        #   trigger = max(floor, native)
        # Only meaningful while SIGNAL_ONLY; once a real order rests the
        # broker-managed trigger (stop_price / trail_pct) applies.
        # Durable persisted peak (authoritative; the scheduler advances it each
        # cycle). Prefer it over a fresh OHLCV lookup so the table matches the
        # exact value the live trail measures against.
        _persisted = trail_peak_by_symbol.get(sym)
        peak_price = float(_persisted[0]) if _persisted else None
        peak_at = _persisted[1] if _persisted else None

        est_trail_trigger = None
        if order_type == "SIGNAL_ONLY" and sig_price and sig_price > 0:
            # None = assignment left trail unset (live trail derives it from ATR);
            # for this display estimate fall back to 2% since we don't compute ATR here.
            _tp = assigned_trail_pct.get(sym) or 2.0
            floor = sig_price * 1.0025
            # Use the persisted peak when present; else reconstruct from OHLCV.
            peak = peak_price or 0.0
            if peak <= 0:
                sig_time = sig.created_at if sig else None
                peak = _peak_since(sym, sig_time)
            if peak <= 0:
                # No history — fall back to the larger of last/signal so we
                # never imply a trail below where price has actually been.
                peak = max(last_px or 0.0, sig_price)
            native = peak * (1.0 - _tp / 100.0)
            est_trail_trigger = round(max(floor, native), 2)
            if trail_pct is None:
                trail_pct = _tp

        out.append(OpenTrailOut(
            symbol=sym,
            quantity=pos.get("quantity"),
            avg_cost=pos.get("avg_cost"),
            last_price=last_px,
            unrealized_pnl=pos.get("unrealized_pnl"),
            unrealized_pct=pos.get("unrealized_pct"),
            broker=pos.get("broker"),
            is_paper=pos.get("is_paper", False),
            signal_price=sig_price,
            signal_at=sig.created_at if sig else None,
            signal_strategy=sig.strategy_name if sig else None,
            order_type=order_type,
            stop_price=stop_price,
            trail_pct=trail_pct,
            armed_at=armed_at,
            days_armed=days_armed,
            move_since_signal_pct=move_pct,
            trail_helping=helping,
            est_trail_trigger=est_trail_trigger,
            peak_price=peak_price,
            peak_at=peak_at,
        ))

    out.sort(key=lambda r: (r.move_since_signal_pct or -999), reverse=True)
    return out
