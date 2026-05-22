from __future__ import annotations

"""Aggregations and roll-ups over the FIFO output.

Pure functions on top of ClosedTrade/OpenLot lists — no DB access, no I/O.
Keeps fifo.py focused on the walk and these helpers focused on math.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from app.services.pnl.fifo import ClosedTrade, OpenLot


@dataclass
class GroupSummary:
    """Aggregated realized stats over a slice of closed trades."""
    key: str                 # symbol, strategy name, etc.
    trade_count: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    total_realized_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float | None  # None when there are no losses
    best_trade: float
    worst_trade: float
    avg_hold_days: float


def summarize(trades: list[ClosedTrade], key: str = "ALL") -> GroupSummary:
    """Compute realized-trade metrics over a slice of ClosedTrades."""
    if not trades:
        return GroupSummary(
            key=key, trade_count=0, win_count=0, loss_count=0, win_rate_pct=0.0,
            total_realized_pnl=0.0, avg_pnl=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=None, best_trade=0.0, worst_trade=0.0, avg_hold_days=0.0,
        )

    pnls = [t.realized_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = sum(pnls)
    gross_win = sum(wins)
    gross_loss = -sum(losses)  # positive magnitude
    pf = (gross_win / gross_loss) if gross_loss > 0 else None

    return GroupSummary(
        key=key,
        trade_count=len(trades),
        win_count=len(wins),
        loss_count=len(losses),
        win_rate_pct=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        total_realized_pnl=total,
        avg_pnl=total / len(trades),
        avg_win=(gross_win / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        profit_factor=pf,
        best_trade=max(pnls),
        worst_trade=min(pnls),
        avg_hold_days=sum(t.hold_days for t in trades) / len(trades),
    )


def by_symbol(trades: list[ClosedTrade]) -> list[GroupSummary]:
    """Group closed trades by symbol and summarize each."""
    buckets: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in trades:
        buckets[t.symbol].append(t)
    out = [summarize(ts, key=sym) for sym, ts in buckets.items()]
    out.sort(key=lambda g: g.total_realized_pnl, reverse=True)
    return out


def by_strategy(trades: list[ClosedTrade]) -> list[GroupSummary]:
    """Group closed trades by the BUY-side strategy attribution.

    We attribute realized P/L to whichever strategy opened the position —
    that's where the call came from. Trades whose BUY had no signal_id
    (manual orders or pre-migration history) bucket into "(unattributed)".
    """
    buckets: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in trades:
        key = t.buy_strategy or "(unattributed)"
        buckets[key].append(t)
    out = [summarize(ts, key=k) for k, ts in buckets.items()]
    out.sort(key=lambda g: g.total_realized_pnl, reverse=True)
    return out


@dataclass
class EquityPoint:
    at: datetime
    realized_pnl: float       # cumulative realized P/L through this point


def equity_curve(trades: list[ClosedTrade]) -> list[EquityPoint]:
    """Cumulative realized P/L over time. Empty list when there are no closes."""
    ordered = sorted(trades, key=lambda t: t.sell_at)
    out: list[EquityPoint] = []
    running = 0.0
    for t in ordered:
        running += t.realized_pnl
        out.append(EquityPoint(at=t.sell_at, realized_pnl=running))
    return out


def open_position_pnl(
    open_lots: list[OpenLot],
    last_prices: dict[str, float],
) -> tuple[list[dict], float]:
    """Combine open lots with live prices to compute per-symbol unrealized P/L.

    Returns (rows, total_unrealized). Rows are dicts (not a dataclass) because
    they pass straight through to the Pydantic response model.
    """
    by_sym: dict[str, dict] = {}
    for lot in open_lots:
        agg = by_sym.setdefault(lot.symbol, {
            "symbol": lot.symbol,
            "quantity": 0.0,
            "cost_basis": 0.0,
            "broker": lot.broker,
            "is_paper": lot.is_paper,
        })
        agg["quantity"] += lot.quantity
        agg["cost_basis"] += lot.quantity * lot.buy_price

    rows: list[dict] = []
    total_unrealized = 0.0
    for sym, agg in by_sym.items():
        qty = agg["quantity"]
        if qty <= 1e-9:
            continue
        avg_cost = agg["cost_basis"] / qty
        last = float(last_prices.get(sym, 0.0) or 0.0)
        mkt_val = last * qty if last > 0 else None
        unrealized = (last - avg_cost) * qty if last > 0 else None
        unrealized_pct = ((last - avg_cost) / avg_cost * 100.0) if (last > 0 and avg_cost > 0) else None
        if unrealized is not None:
            total_unrealized += unrealized
        rows.append({
            "symbol": sym,
            "quantity": qty,
            "avg_cost": avg_cost,
            "last_price": last if last > 0 else None,
            "market_value": mkt_val,
            "unrealized_pnl": unrealized,
            "unrealized_pct": unrealized_pct,
            "broker": agg["broker"],
            "is_paper": agg["is_paper"],
        })
    rows.sort(key=lambda r: (r.get("unrealized_pnl") or 0.0), reverse=True)
    return rows, total_unrealized
