"""
Regression test: the live autotrader must translate its 4-value action vocabulary
(BUY / SELL / SELL_SHORT / BUY_COVER) into the broker's BUY/SELL.

Bug guarded against: _submit_via_execution_service used to reject any action
not in ("BUY","SELL"), so SELL_SHORT (open short) and BUY_COVER (close short)
fell through and were silently dropped — live short entries and covers never
reached the broker, leaving short positions stuck open. Paper mode (broker=None)
masked this because it fills at last price without going through the submit path.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.strategy.daytrading.autotrader.single_stock_trader import (
    SingleStockTrader,
)


def _trader_with_capturing_exec():
    """Build a trader whose ExecutionService records the OrderRequest it receives."""
    captured = {}

    class _FakeExecResult:
        fill_price = 100.0

    async def _execute(order_req, account_id="", estimated_price=0.0):
        captured["side"] = order_req.side
        captured["symbol"] = order_req.symbol
        captured["qty"] = order_req.quantity
        return _FakeExecResult()

    exec_svc = MagicMock()
    exec_svc.execute = _execute

    broker = MagicMock()
    broker.is_paper = False  # force the live submit path

    t = SingleStockTrader(
        symbol="AAPL",
        broker=broker,
        execution_service=exec_svc,
        account_id="acct1",
    )
    # Seed a price so _last_price() > 0 and the fill path returns a real number.
    import pandas as pd
    t._df_5m = pd.DataFrame({"Close": [100.0]})
    return t, captured


@pytest.mark.parametrize("action,expected_side", [
    ("BUY", "BUY"),
    ("SELL", "SELL"),
    ("SELL_SHORT", "SELL"),   # open short -> sell order
    ("BUY_COVER", "BUY"),     # cover short -> buy order
])
def test_action_maps_to_broker_side(action, expected_side):
    t, captured = _trader_with_capturing_exec()
    fill = t._submit_via_execution_service(action, qty=10, is_exit=action in ("SELL", "BUY_COVER"))
    assert fill == 100.0, "order should have filled, not been dropped"
    assert captured["side"] == expected_side


def test_short_entry_is_not_silently_dropped():
    # The exact regression: SELL_SHORT used to return 0.0 (no order placed).
    t, captured = _trader_with_capturing_exec()
    fill = t._submit_via_execution_service("SELL_SHORT", qty=5, is_exit=False)
    assert fill > 0.0
    assert captured.get("side") == "SELL"


def test_cover_is_not_silently_dropped():
    t, captured = _trader_with_capturing_exec()
    fill = t._submit_via_execution_service("BUY_COVER", qty=5, is_exit=True)
    assert fill > 0.0
    assert captured.get("side") == "BUY"


def test_unknown_action_still_rejected():
    t, captured = _trader_with_capturing_exec()
    fill = t._submit_via_execution_service("WIGGLE", qty=5, is_exit=False)
    assert fill == 0.0
    assert "side" not in captured  # never reached the broker
