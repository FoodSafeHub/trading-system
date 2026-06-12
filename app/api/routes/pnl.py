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
    exit_type: Optional[str] = None         # "trailing_stop" | "market" | "stop" | "unknown"

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


# ── Helpers ────────────────────────────────────────────────────────────────

def _refresh(db: Session):
    """Sync new round-trips into realized_trades, then return (closed, fifo).

    The FifoResult is reused for open lots (no separate query). The closed list
    is the canonical persisted history — same numbers across calls regardless of
    what the live walk would produce.
    """
    _inserted, fifo = sync_realized_trades(db)
    closed = load_closed_trades(db)
    return closed, fifo


def _resolve_last_prices(symbols: list[str]) -> dict[str, float]:
    """Pull last/ask/bid from the active broker for the given symbols.

    Returns {} on any failure — the unrealized columns will surface as None
    rather than 0, so the UI can tell "we don't know" from "zero".
    """
    if not symbols:
        return {}
    import asyncio
    try:
        broker = get_broker()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(broker.authenticate())
            quotes = loop.run_until_complete(broker.get_quotes(symbols))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[pnl] quote fetch failed: %s", exc)
        return {}

    out: dict[str, float] = {}
    for sym, q in (quotes or {}).items():
        price = getattr(q, "last", None) or getattr(q, "ask", None) or getattr(q, "bid", None)
        if price and price > 0:
            out[sym.upper()] = float(price)
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
    """Query the active broker for WORKING SELL STOP/TRAILING_STOP orders.

    The scheduler places these on the broker; they may not all be mirrored in
    the local DB (e.g. scanner-path SELLs). This is the source of truth for
    "what trail is actually resting right now". Returns {symbol: {...}}.
    """
    import asyncio
    result: dict[str, dict] = {}
    try:
        broker = get_broker()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(broker.authenticate())
            accts = loop.run_until_complete(broker.get_accounts())
            acct_id = accts[0].account_id if accts else ""
            orders = loop.run_until_complete(broker.list_orders(acct_id, status="working"))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[pnl] broker working-orders fetch failed: %s", exc)
        return {}

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
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/summary", response_model=SummaryOut)
def pnl_summary(
    include_unrealized: bool = True,
    db: Session = Depends(get_db),
):
    """Top-line numbers: realized so far, unrealized on open positions, counts."""
    closed, fifo = _refresh(db)
    realized = summarize(closed, key="ALL")
    last_prices: dict[str, float] = {}
    if include_unrealized and fifo.open_lots:
        last_prices = _resolve_last_prices(sorted({l.symbol for l in fifo.open_lots}))
    open_rows, total_unrealized = open_position_pnl(fifo.open_lots, last_prices)
    return SummaryOut(
        realized=GroupSummaryOut(**asdict(realized)),
        total_unrealized_pnl=total_unrealized,
        open_position_count=len(open_rows),
        closed_trade_count=len(closed),
        has_live_prices=bool(last_prices),
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

    For trailing-stop exits, also returns signal_price (price when the SELL
    signal fired), signal_at (when it fired), trail_pct (the % trail placed),
    trail_captured_pct (extra gain between signal price and actual exit), and
    exit_type so the dashboard can render the trail stop audit.
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
    if sell_ids:
        sell_orders = db.query(Order).filter(Order.id.in_(sell_ids)).all()
        order_by_id = {o.id: o for o in sell_orders}
        sig_ids = [o.signal_id for o in sell_orders if o.signal_id]
        signals_by_id: dict[int, Signal] = {}
        if sig_ids:
            for sig in db.query(Signal).filter(Signal.id.in_(sig_ids)).all():
                signals_by_id[sig.id] = sig

        for o in sell_orders:
            sig = signals_by_id.get(o.signal_id) if o.signal_id else None
            exit_type = (o.order_type or "unknown").lower()
            trail_info[o.id] = {
                "signal_price": float(sig.price_at_signal) if sig and sig.price_at_signal else None,
                "signal_at":    sig.created_at if sig else None,
                "trail_pct":    float(o.trail_value) if getattr(o, "trail_value", None) else None,
                "exit_type":    exit_type,
            }

    result = []
    for t in rows:
        info = trail_info.get(t.sell_order_id, {})
        sp = info.get("signal_price")
        # trail_captured_pct: how much extra (%) the stock moved from signal → exit
        trail_captured = None
        if sp and sp > 0 and t.sell_price:
            trail_captured = round((t.sell_price - sp) / sp * 100, 2)
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


@router.get("/open-positions", response_model=List[OpenPositionOut])
def pnl_open_positions(db: Session = Depends(get_db)):
    """Open long lots aggregated per symbol, with unrealized P/L vs last quote."""
    _, fifo = _refresh(db)
    last_prices: dict[str, float] = {}
    if fifo.open_lots:
        last_prices = _resolve_last_prices(sorted({l.symbol for l in fifo.open_lots}))
    rows, _total = open_position_pnl(fifo.open_lots, last_prices)
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
    if not fifo.open_lots:
        return []

    last_prices = _resolve_last_prices(sorted({l.symbol for l in fifo.open_lots}))
    rows, _total = open_position_pnl(fifo.open_lots, last_prices)
    open_by_symbol = {r["symbol"]: r for r in rows}
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
    from app.models.assignments import SymbolStrategyAssignment

    assigned_strat: dict[str, str] = {}
    assigned_trail_pct: dict[str, float] = {}   # per-symbol tight_trail_pct (default 2%)
    if symbols:
        for a in (
            db.query(SymbolStrategyAssignment)
            .filter(
                SymbolStrategyAssignment.symbol.in_(symbols),
                SymbolStrategyAssignment.enabled == True,  # noqa: E712
            )
            .all()
        ):
            assigned_strat[a.symbol.upper()] = a.strategy_name
            assigned_trail_pct[a.symbol.upper()] = float(a.tight_trail_pct or 2.0)

    fallback_sig: dict[str, Signal] = {}
    if assigned_strat:
        # Anchor to the SCANNER discovery stream for the assigned strategy
        # ("scanner:<assigned_strategy>") — this is the feed shown on the
        # Notifications page and is the signal price the user references.
        scanner_names = {f"scanner:{nm}": sym
                         for sym, nm in assigned_strat.items()}

        all_sells = (
            db.query(Signal)
            .filter(
                Signal.symbol.in_(list(assigned_strat.keys())),
                Signal.direction == "SELL",
                Signal.price_at_signal.isnot(None),
                Signal.strategy_name.in_(list(scanner_names.keys())),
            )
            .order_by(Signal.created_at.asc())   # earliest first
            .all()
        )
        for s in all_sells:
            sym = s.symbol.upper()
            if sym in fallback_sig:
                continue  # already have the FIRST qualifying signal
            # Confirm this scanner signal belongs to THIS symbol's assigned strategy.
            if scanner_names.get(s.strategy_name) != sym:
                continue
            # Only signals fired AFTER the position was acquired.
            buy_t = earliest_buy.get(sym)
            if buy_t is None or _naive(s.created_at) >= _naive(buy_t):
                fallback_sig[sym] = s

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
        est_trail_trigger = None
        if order_type == "SIGNAL_ONLY" and sig_price and sig_price > 0:
            _tp = assigned_trail_pct.get(sym, 2.0)
            floor = sig_price * 1.0025
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
        ))

    out.sort(key=lambda r: (r.move_since_signal_pct or -999), reverse=True)
    return out
