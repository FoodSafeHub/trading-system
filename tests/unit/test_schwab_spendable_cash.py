"""
Contract tests for SchwabBroker._spendable_cash — the figure the scheduler's
cash gate uses to avoid submitting BUYs Schwab rejects for "not enough cash".

The key bug: cashBalance OVER-reports for a CASH account because it includes
UNSETTLED sell proceeds a cash account can't spend on a new BUY. The gate must
see SETTLED cash only.
"""
from __future__ import annotations

from app.services.brokers.schwab import SchwabBroker


def _sec(acct_type, **balances):
    return {"type": acct_type, "currentBalances": balances}


def test_cash_account_excludes_unsettled():
    # cashAvailableForTrading looks like 1357 but 413 is unsettled → 944 settled.
    sec = _sec(
        "CASH",
        cashBalance=1357.87,
        cashAvailableForTrading=1357.87,
        unsettledCash=413.83,
        cashAvailableForWithdrawal=944.04,
    )
    assert abs(SchwabBroker._spendable_cash(sec) - 944.04) < 0.01


def test_cash_account_all_settled():
    sec = _sec(
        "CASH",
        cashBalance=500.0,
        cashAvailableForTrading=500.0,
        unsettledCash=0.0,
    )
    assert SchwabBroker._spendable_cash(sec) == 500.0


def test_cash_account_never_negative():
    # Defensive: unsettled somehow exceeds available → clamp to 0, not negative.
    sec = _sec("CASH", cashAvailableForTrading=100.0, unsettledCash=250.0)
    assert SchwabBroker._spendable_cash(sec) == 0.0


def test_margin_account_uses_buying_power():
    sec = _sec("MARGIN", buyingPower=8000.0, cashBalance=1000.0, cashAvailableForTrading=1000.0)
    assert SchwabBroker._spendable_cash(sec) == 8000.0


def test_falls_back_to_cash_balance_when_fields_missing():
    # A schema change that drops the available fields must not blank the budget.
    sec = _sec("CASH", cashBalance=321.0)
    assert SchwabBroker._spendable_cash(sec) == 321.0


def test_returns_none_when_nothing_present():
    assert SchwabBroker._spendable_cash({"type": "CASH", "currentBalances": {}}) is None
