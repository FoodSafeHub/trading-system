"""
Contract tests for cash_aware US broker routing.

Under trade_routing="cash_aware", a default-broker US BUY must route to the US
broker (Schwab/Webull) with the most fundable cash, and the per-strategy cap
must aggregate held qty across US brokers so a position split across accounts
can't pyramid past its cap.
"""
from __future__ import annotations

import app.services.strategy.scheduler as sched


def test_pick_us_broker_prefers_most_cash():
    budgets = {"schwab": 100.0, "webull": 5000.0}
    assert sched._pick_us_broker_by_cash(lambda n: budgets[n]) == "webull"

    budgets = {"schwab": 9000.0, "webull": 20.0}
    assert sched._pick_us_broker_by_cash(lambda n: budgets[n]) == "schwab"


def test_pick_us_broker_ties_go_to_schwab():
    # Equal cash → first in _US_BROKERS (Schwab), the historical default.
    assert sched._pick_us_broker_by_cash(lambda n: 500.0) == "schwab"


def test_pick_us_broker_falls_back_when_cash_unreadable():
    def _boom(_name):
        raise RuntimeError("balance fetch failed")

    assert sched._pick_us_broker_by_cash(_boom) == "schwab"


def test_us_held_qty_aggregates_across_brokers(monkeypatch):
    # A symbol whose lots landed on Schwab AND Webull (plus a legacy default
    # key) must sum to the combined position for the cap check.
    ledger = {
        ("AAPL", "scanner", "s", "schwab"): 3.0,
        ("AAPL", "scanner", "s", "webull"): 4.0,
        ("AAPL", "scanner", "s", "default"): 1.0,
    }
    from app.services.strategy import strategy_ledger

    monkeypatch.setattr(
        strategy_ledger, "get_held",
        lambda sym, sysn, strat, bkey: ledger.get((sym, sysn, strat, bkey), 0.0),
    )
    assert sched._us_held_qty("AAPL", "scanner", "s") == 8.0


def test_us_held_qty_zero_when_nothing_held(monkeypatch):
    from app.services.strategy import strategy_ledger

    monkeypatch.setattr(strategy_ledger, "get_held", lambda *a, **k: 0.0)
    assert sched._us_held_qty("AAPL", "scanner", "s") == 0.0


def test_resolve_routing_cash_aware_sees_both_for_reads():
    from app.services.brokers.factory import _resolve_routing

    # For reads/global broker, cash_aware resolves the same US set as "both"
    # so positions/quotes see both brokers; the single-broker PICK is per-BUY.
    assert _resolve_routing("cash_aware", "schwab") == ["schwab", "webull"]
