"""Shared helper: the anchor SELL signal for a held position's tight trail.

Single source of truth used by BOTH the scheduler's trail reconciliation and
the PnL "Armed Tight Trails" table, so they never diverge.

The anchor is the FIRST SELL signal from the symbol's ASSIGNED strategy
(stored as ``scanner:<strategy>``) fired AT OR AFTER the position was acquired.
That is the price the strategy first flagged the exit at — the level the trail
should be measured against and the trigger that arms Approach C.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.assignments import SymbolStrategyAssignment
from app.models.signals import Signal


def _naive(dt):
    """Strip tzinfo for safe comparison between naive and aware datetimes."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def assigned_strategy_map(db: Session, symbols: list[str]) -> dict[str, str]:
    """Return {SYMBOL: assigned_strategy_name} for the enabled assignments."""
    out: dict[str, str] = {}
    if not symbols:
        return out
    for a in (
        db.query(SymbolStrategyAssignment)
        .filter(
            SymbolStrategyAssignment.symbol.in_(symbols),
            SymbolStrategyAssignment.enabled == True,  # noqa: E712
        )
        .all()
    ):
        out[a.symbol.upper()] = a.strategy_name
    return out


def assigned_trail_pct_map(db: Session, symbols: list[str], default: float = 2.0) -> dict[str, float]:
    """Return {SYMBOL: tight_trail_pct} (assignment value, else `default`)."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    for a in (
        db.query(SymbolStrategyAssignment)
        .filter(
            SymbolStrategyAssignment.symbol.in_(symbols),
            SymbolStrategyAssignment.enabled == True,  # noqa: E712
        )
        .all()
    ):
        out[a.symbol.upper()] = float(a.tight_trail_pct or default)
    return out


def first_assigned_sell_signals(
    db: Session,
    assigned_strat: dict[str, str],
    earliest_buy: dict[str, datetime],
) -> dict[str, Signal]:
    """For each symbol, the FIRST ``scanner:<assigned_strategy>`` SELL signal
    fired at/after the position was acquired.

    Parameters
    ----------
    assigned_strat : {SYMBOL: assigned_strategy_name}
    earliest_buy   : {SYMBOL: earliest open-lot buy time} (anchor cutoff)

    Returns {SYMBOL: Signal} only for symbols whose assigned strategy actually
    fired a qualifying SELL — symbols with none are absent.
    """
    result: dict[str, Signal] = {}
    if not assigned_strat:
        return result

    # Map the scanner-stream strategy name → symbol so we can confirm ownership.
    scanner_names = {f"scanner:{nm}": sym for sym, nm in assigned_strat.items()}

    rows = (
        db.query(Signal)
        .filter(
            Signal.symbol.in_(list(assigned_strat.keys())),
            Signal.direction == "SELL",
            Signal.price_at_signal.isnot(None),
            Signal.strategy_name.in_(list(scanner_names.keys())),
        )
        .order_by(Signal.created_at.asc())  # earliest first
        .all()
    )
    for s in rows:
        sym = s.symbol.upper()
        if sym in result:
            continue  # already have the FIRST qualifying signal
        if scanner_names.get(s.strategy_name) != sym:
            continue  # not this symbol's assigned strategy
        buy_t = earliest_buy.get(sym)
        if buy_t is None or _naive(s.created_at) >= _naive(buy_t):
            result[sym] = s
    return result
