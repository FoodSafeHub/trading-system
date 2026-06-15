"""Approach C peak-capture audit fields on /pnl/closed-trades.

The Trail Stop Audit panel reports, per closed trade: the signal price, the PEAK
the trail ratcheted off (from trail_peaks), and capture efficiency =
(exit - signal) / (peak - signal). It also normalises exit_type so a bot-managed
STOP (the Approach C floored trail) counts as a "trail" exit, not "market".
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orders import Order
from app.models.signals import Signal
from app.models.trail_peaks import TrailPeak
from app.api.routes.pnl import pnl_closed_trades


@pytest.fixture
def db_session():
    from app.models import orders, realized_trades, signals, trail_peaks  # noqa: F401
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def _seed_amal(db, *, sell_order_type, signal_price=43.55, peak=45.23, exit_px=44.2388):
    t_buy = datetime(2026, 6, 3, tzinfo=timezone.utc)
    t_sell = datetime(2026, 6, 15, tzinfo=timezone.utc)
    # SELL signal the trail was anchored to.
    sig = Signal(
        strategy_name="scanner:Legacy_AMAL_Fib_Pullback", symbol="AMAL",
        direction="SELL", strength=1.0, price_at_signal=signal_price, acted_on=True,
    )
    db.add(sig)
    db.flush()
    db.add(Order(
        id=1, broker="schwab", broker_order_id="b1", symbol="AMAL", side="BUY",
        order_type="MARKET", quantity=12, status="filled", is_paper=False,
        fill_price=41.275, filled_at=t_buy,
    ))
    db.add(Order(
        id=2, broker="schwab", broker_order_id="s2", symbol="AMAL", side="SELL",
        order_type=sell_order_type, quantity=12, status="filled", is_paper=False,
        fill_price=exit_px, filled_at=t_sell, signal_id=sig.id,
    ))
    db.add(TrailPeak(symbol="AMAL", signal_id=sig.id, signal_price=signal_price,
                     peak_price=peak, peak_at=t_sell))
    db.commit()


def test_capture_efficiency_and_trail_exit_type(db_session):
    with db_session() as db:
        _seed_amal(db, sell_order_type="STOP")  # bot-managed floored trail
        rows = pnl_closed_trades(db=db)

    audited = [r for r in rows if r.symbol == "AMAL"]
    assert audited, "expected an AMAL closed trade"
    r = audited[0]
    assert r.signal_price == pytest.approx(43.55)
    assert r.peak_price == pytest.approx(45.23)
    # (44.2388 - 43.55) / (45.23 - 43.55) * 100 = 41.0%
    assert r.capture_efficiency_pct == pytest.approx(41.0, abs=0.5)
    # A bot-managed STOP is a TRAIL exit, not "market".
    assert r.exit_type == "trail"


def test_native_trailing_stop_is_also_trail(db_session):
    with db_session() as db:
        _seed_amal(db, sell_order_type="TRAILING_STOP")
        rows = pnl_closed_trades(db=db)
    r = [x for x in rows if x.symbol == "AMAL"][0]
    assert r.exit_type == "trail"


def test_no_peak_means_no_efficiency(db_session):
    """A close with a signal but no recorded peak still returns, with
    capture_efficiency_pct = None (it won't appear in the audit panel)."""
    with db_session() as db:
        # Seed without a TrailPeak row.
        sig = Signal(strategy_name="scanner:X", symbol="KO", direction="SELL",
                     strength=1.0, price_at_signal=78.0, acted_on=True)
        db.add(sig); db.flush()
        db.add(Order(id=1, broker="schwab", broker_order_id="b", symbol="KO",
                     side="BUY", order_type="MARKET", quantity=5, status="filled",
                     is_paper=False, fill_price=78.0,
                     filled_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        db.add(Order(id=2, broker="schwab", broker_order_id="s", symbol="KO",
                     side="SELL", order_type="MARKET", quantity=5, status="filled",
                     is_paper=False, fill_price=81.0, signal_id=sig.id,
                     filled_at=datetime(2026, 6, 10, tzinfo=timezone.utc)))
        db.commit()
        rows = pnl_closed_trades(db=db)
    r = [x for x in rows if x.symbol == "KO"][0]
    assert r.peak_price is None
    assert r.capture_efficiency_pct is None
    assert r.exit_type == "market"
