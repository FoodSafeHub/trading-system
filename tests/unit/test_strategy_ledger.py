"""Per-strategy share ledger.

A symbol can be held by several strategy assignments at once, but the broker
reports only one aggregate position. The ledger tracks each strategy's own lot so
a SELL fired by one strategy touches only that strategy's shares.

These tests pin:
  - BUY adds to a strategy's held qty and volume-weights avg_price.
  - SELL subtracts and clamps at 0 (never goes negative).
  - Two strategies on the same symbol keep independent lots.
  - order_sync's _attribute_fill_to_ledger maps a filled order to the right
    assignment via signal -> strategy_name, and leaves unmapped fills alone.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.assignments import SymbolStrategyAssignment
from app.models.orders import Order
from app.models.signals import Signal
from app.models.strategy_positions import StrategyPosition
from app.services.strategy import strategy_ledger


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def _held(db, symbol, system, strat, broker="default"):
    return strategy_ledger.get_held(symbol, system, strat, broker, db=db)


def test_buy_adds_and_weights_avg_price(db):
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "BUY", 100, price=10.0, db=db)
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "BUY", 100, price=20.0, db=db)
    row = db.query(StrategyPosition).filter_by(
        symbol="AAPL", system="bollinger", strategy_name="X", broker="default"
    ).one()
    assert row.held_qty == 200
    assert row.avg_price == pytest.approx(15.0)


def test_sell_subtracts_and_clamps_at_zero(db):
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "BUY", 100, price=10.0, db=db)
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "SELL", 60, price=11.0, db=db)
    assert _held(db, "AAPL", "bollinger", "X") == 40
    # Oversell: clamps to 0, never negative.
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "SELL", 999, price=11.0, db=db)
    assert _held(db, "AAPL", "bollinger", "X") == 0


def test_two_strategies_same_symbol_are_independent(db):
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "BUY", 200, price=10.0, db=db)
    strategy_ledger.apply_fill("AAPL", "scanner", "Y", "BUY", 100, price=10.0, db=db)
    # X sells its whole lot — Y is untouched.
    strategy_ledger.apply_fill("AAPL", "bollinger", "X", "SELL", 200, price=12.0, db=db)
    assert _held(db, "AAPL", "bollinger", "X") == 0
    assert _held(db, "AAPL", "scanner", "Y") == 100


def test_attribute_fill_maps_order_to_assignment(db):
    db.add(SymbolStrategyAssignment(
        symbol="MSFT", system="scanner", strategy_name="Y", enabled=True,
        broker="default", assigned_at=datetime.now(tz=timezone.utc),
    ))
    sig = Signal(strategy_name="Y", symbol="MSFT", direction="BUY",
                 created_at=datetime.now(tz=timezone.utc))
    db.add(sig)
    db.commit()
    order = Order(broker="paper", symbol="MSFT", side="BUY", order_type="MARKET",
                  quantity=50, status="filled", fill_price=30.0, signal_id=sig.id)
    db.add(order)
    db.commit()

    from app.services.reconciliation.order_sync import _attribute_fill_to_ledger
    _attribute_fill_to_ledger(db, order)
    db.commit()
    assert _held(db, "MSFT", "scanner", "Y") == 50


def test_attribute_fill_strips_system_prefix_from_label(db):
    # Regression: scanner/perplexity signals store a PREFIXED label
    # ("scanner:NAME"), but the assignment + the scheduler's ledger reads use the
    # BARE name. If attribution doesn't strip the prefix, the ledger stays 0 and
    # every BUY re-buys the full cap (the NVDA over-buy bug).
    db.add(SymbolStrategyAssignment(
        symbol="NVDA", system="scanner", strategy_name="Squeeze", enabled=True,
        broker="default", assigned_at=datetime.now(tz=timezone.utc),
    ))
    sig = Signal(strategy_name="scanner:Squeeze", symbol="NVDA", direction="BUY",
                 created_at=datetime.now(tz=timezone.utc))
    db.add(sig)
    db.commit()
    order = Order(broker="paper", symbol="NVDA", side="BUY", order_type="MARKET",
                  quantity=3, status="filled", fill_price=100.0, signal_id=sig.id)
    db.add(order)
    db.commit()

    from app.services.reconciliation.order_sync import _attribute_fill_to_ledger
    _attribute_fill_to_ledger(db, order)
    db.commit()
    # Ledger must be keyed by the BARE (system, name) the scheduler reads with.
    assert _held(db, "NVDA", "scanner", "Squeeze") == 3


def test_attribute_fill_ignores_unmapped_order(db):
    # No signal link -> orphan/manual fill -> ledger untouched.
    order = Order(broker="paper", symbol="TSLA", side="SELL", order_type="MARKET",
                  quantity=10, status="filled", fill_price=200.0, signal_id=None)
    db.add(order)
    db.commit()
    from app.services.reconciliation.order_sync import _attribute_fill_to_ledger
    _attribute_fill_to_ledger(db, order)
    db.commit()
    assert db.query(StrategyPosition).count() == 0
