"""
Contract tests for SchwabBroker._spendable_cash — the figure the scheduler's
cash gate uses to avoid submitting BUYs Schwab rejects for "not enough cash".

Contract (revised 2026-07): for CASH accounts the gate uses Schwab's own
`cashAvailableForTrading`, which INCLUDES unsettled T+1 sell proceeds — Schwab
accepts BUYs against them (a good-faith violation only occurs if the new
position is sold before those funds settle). The old settled-only policy
(available − unsettledCash) pinned the budget near zero on any day the system
traded, starving every BUY. MARGIN accounts still use buyingPower.
"""
from __future__ import annotations

from app.services.brokers.schwab import SchwabBroker


def _sec(acct_type, **balances):
    return {"type": acct_type, "currentBalances": balances}


def test_cash_account_includes_unsettled():
    # Unsettled T+1 proceeds are spendable: use Schwab's cashAvailableForTrading
    # as-is, NOT available − unsettled (which starved the gate to 944 here).
    sec = _sec(
        "CASH",
        cashBalance=1357.87,
        cashAvailableForTrading=1357.87,
        unsettledCash=413.83,
        cashAvailableForWithdrawal=944.04,
    )
    assert abs(SchwabBroker._spendable_cash(sec) - 1357.87) < 0.01


def test_cash_account_all_settled():
    sec = _sec(
        "CASH",
        cashBalance=500.0,
        cashAvailableForTrading=500.0,
        unsettledCash=0.0,
    )
    assert SchwabBroker._spendable_cash(sec) == 500.0


def test_cash_account_uses_available_for_trading_verbatim():
    # Even when unsettledCash exceeds the available figure, we trust Schwab's
    # own tradable number rather than deriving (and clamping) our own.
    sec = _sec("CASH", cashAvailableForTrading=100.0, unsettledCash=250.0)
    assert SchwabBroker._spendable_cash(sec) == 100.0


def test_margin_account_uses_buying_power():
    sec = _sec("MARGIN", buyingPower=8000.0, cashBalance=1000.0, cashAvailableForTrading=1000.0)
    assert SchwabBroker._spendable_cash(sec) == 8000.0


def test_falls_back_to_cash_balance_when_fields_missing():
    # A schema change that drops the available fields must not blank the budget.
    sec = _sec("CASH", cashBalance=321.0)
    assert SchwabBroker._spendable_cash(sec) == 321.0


def test_returns_none_when_nothing_present():
    assert SchwabBroker._spendable_cash({"type": "CASH", "currentBalances": {}}) is None
