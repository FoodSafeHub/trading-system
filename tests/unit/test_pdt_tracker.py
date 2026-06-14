"""
Tests for the PDT (Pattern Day Trader) tracker.

The guard restricts ONLY real (non-paper) margin accounts under $25k to 3 day
trades in a rolling 5-business-day window. Cash accounts, funded (>=$25k)
accounts, and paper accounts are exempt. A day trade is an open+close of the
same symbol on the same trading day; overnight holds do not count.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from app.services.strategy.daytrading.brain.pdt_tracker import (
    PDT_MAX_DAY_TRADES,
    PDTTracker,
    _last_n_business_days,
)

# A fixed Wednesday so the rolling window is deterministic and weekend-free.
WED = date(2026, 6, 10)


def _rt(d: datetime, sell: datetime | None = None, symbol: str = "AAPL") -> dict:
    return {"symbol": symbol, "buy_at": d, "sell_at": sell or d}


# ── Exemptions ─────────────────────────────────────────────────────────────────

def test_paper_is_exempt():
    t = PDTTracker(account_type="margin", equity=1_000, is_paper=True)
    d = t.check_can_open_day_trade([], as_of=WED)
    assert d.allowed
    assert not d.status.guard_active
    assert "paper" in d.status.exempt_reason


def test_cash_account_is_exempt():
    t = PDTTracker(account_type="cash", equity=1_000, is_paper=False)
    assert not t.build_status([], as_of=WED).guard_active


def test_funded_margin_is_exempt():
    t = PDTTracker(account_type="margin", equity=25_000, is_paper=False)
    assert not t.build_status([], as_of=WED).guard_active
    t2 = PDTTracker(account_type="margin", equity=99_999, is_paper=False)
    assert not t2.build_status([], as_of=WED).guard_active


# ── Active guard counting ──────────────────────────────────────────────────────

def _sub25k() -> PDTTracker:
    return PDTTracker(account_type="margin", equity=10_000, is_paper=False)


def test_guard_active_for_sub25k_margin():
    assert _sub25k().build_status([], as_of=WED).guard_active


def test_three_day_trades_blocks_fourth():
    rts = [_rt(datetime(2026, 6, 10, 10)) for _ in range(3)]
    d = _sub25k().check_can_open_day_trade(rts, symbol="AAPL", as_of=WED)
    assert not d.allowed
    assert d.status.day_trades_used == 3
    assert d.status.day_trades_remaining == 0
    assert "PDT BLOCK" in d.reason


def test_two_day_trades_still_allowed():
    rts = [_rt(datetime(2026, 6, 10, 10)) for _ in range(2)]
    d = _sub25k().check_can_open_day_trade(rts, as_of=WED)
    assert d.allowed
    assert d.status.day_trades_remaining == 1


def test_day_trades_counted_across_window_days():
    rts = [
        _rt(datetime(2026, 6, 8, 10)),   # Mon
        _rt(datetime(2026, 6, 9, 10)),   # Tue
        _rt(datetime(2026, 6, 10, 10)),  # Wed
    ]
    status = _sub25k().build_status(rts, as_of=WED)
    assert status.day_trades_used == 3
    assert status.per_day == {"2026-06-08": 1, "2026-06-09": 1, "2026-06-10": 1}


def test_overnight_hold_is_not_a_day_trade():
    # buy Tue, sell Wed -> not a day trade.
    rts = [_rt(datetime(2026, 6, 9, 10), sell=datetime(2026, 6, 10, 10))]
    status = _sub25k().build_status(rts, as_of=WED)
    assert status.day_trades_used == 0


def test_old_day_trade_rolls_off_window():
    # 2026-06-01 (Mon) is more than 5 business days before Wed 6/10.
    rts = [_rt(datetime(2026, 6, 1, 10))]
    status = _sub25k().build_status(rts, as_of=WED)
    assert status.day_trades_used == 0


def test_iso_string_timestamps_are_accepted():
    rts = [{"symbol": "AAPL", "buy_at": "2026-06-10T10:00:00", "sell_at": "2026-06-10T11:00:00"}]
    status = _sub25k().build_status(rts, as_of=WED)
    assert status.day_trades_used == 1


# ── Window helper ──────────────────────────────────────────────────────────────

def test_window_skips_weekends():
    days = _last_n_business_days(WED, 5)
    assert len(days) == 5
    # Wed 6/10 window: Thu 6/4, Fri 6/5, Mon 6/8, Tue 6/9, Wed 6/10
    assert days[0] == date(2026, 6, 4)
    assert days[-1] == WED
    assert all(d.weekday() < 5 for d in days)


def test_max_day_trades_constant():
    assert PDT_MAX_DAY_TRADES == 3
