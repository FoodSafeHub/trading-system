"""
Targeted tests for the per-symbol cap-aware sizing in scheduler._compute_quantity.

Pin the contract:
    * max_capital_usd is a TOTAL-position cap (includes shares already held)
    * max_shares is a TOTAL-position cap when max_capital_usd is empty
    * Already at/over the cap → returns 0 (no top-up)
    * Partial holding → returns the gap shares only
    * No held position → behaves like the prior implementation (full cap)
    * No cap at all → falls back to the configured account-level sizer

The risk-based path inside _compute_quantity calls calculate_position_size,
which depends on a settings instance — we focus the tests on the cap-only
branches (stop missing / unusable) so the contract is provable without
mocking the whole settings + risk stack.
"""
from __future__ import annotations

import pytest

from app.services.strategy.scheduler import _compute_quantity


# ── No-stop branch (cap-only sizing) ────────────────────────────────────────


class TestMaxCapitalUsd:
    def test_zero_held_returns_full_cap_worth(self):
        # max_cap $1000, entry $100 → cap implies 10 shares total
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=1000.0, max_shares=None, held_qty=0.0,
        )
        assert qty == 10.0

    def test_partial_held_returns_gap_only(self):
        # cap = 10 shares total; already hold 4 → top-up = 6
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=1000.0, max_shares=None, held_qty=4.0,
        )
        assert qty == 6.0

    def test_already_at_cap_returns_zero(self):
        # already at exactly the cap
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=1000.0, max_shares=None, held_qty=10.0,
        )
        assert qty == 0.0

    def test_already_over_cap_returns_zero(self):
        # over the cap (the brief: don't close, just refuse to add more)
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=1000.0, max_shares=None, held_qty=15.0,
        )
        assert qty == 0.0


class TestMaxShares:
    def test_zero_held_returns_full_cap(self):
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=None, max_shares=8.0, held_qty=0.0,
        )
        assert qty == 8.0

    def test_partial_held_returns_gap_only(self):
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=None, max_shares=8.0, held_qty=3.0,
        )
        assert qty == 5.0

    def test_already_at_or_over_cap_returns_zero(self):
        for held in (8.0, 9.5, 100.0):
            qty = _compute_quantity(
                "AAPL", entry=100.0, stop=None,
                max_capital_usd=None, max_shares=8.0, held_qty=held,
            )
            assert qty == 0.0, f"expected 0 when held={held}, got {qty}"


class TestCapPrecedence:
    def test_dollar_cap_wins_over_shares_cap(self):
        # max_cap_usd implies 10 shares; max_shares implies 20.
        # Dollar cap should bind first.
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=1000.0, max_shares=20.0, held_qty=0.0,
        )
        assert qty == 10.0


class TestNoCap:
    def test_no_cap_no_held_defaults_to_one_share(self):
        # No cap, no stop → previous behaviour: 1 share fallback (now gap-aware).
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=None, max_shares=None, held_qty=0.0,
        )
        assert qty == 1.0

    def test_no_cap_held_one_returns_zero(self):
        # Without an explicit cap the engine defaults the cap to "1 share".
        # If we already hold 1, no top-up.
        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=None,
            max_capital_usd=None, max_shares=None, held_qty=1.0,
        )
        assert qty == 0.0


# ── Risk-based path (stop present) ──────────────────────────────────────────


class TestRiskBranchHonoursHeld:
    def test_held_value_subtracted_from_cap(self, monkeypatch):
        """When a stop is present and the cap is partially consumed by held
        shares, the risk sizer must see the REDUCED cap so it can't return a
        quantity that would push us past the user's intent."""
        # Stub the risk sizer to echo its max_position_size_usd input back
        # so the test verifies what _compute_quantity passes it.
        from app.services.strategy import scheduler as sched

        recorded = {}

        class _StubResult:
            viable = True
            shares = 0.0
            risk_amount = 0.0

        def _fake_sizer(**kwargs):
            recorded["cap_seen"] = kwargs["max_position_size_usd"]
            r = _StubResult()
            # Return a small qty so the function returns a non-zero size,
            # but the test only cares about cap_seen.
            r.shares = 0.5
            r.risk_amount = 10.0
            return r

        monkeypatch.setattr(sched, "calculate_position_size", _fake_sizer)

        # max_cap = $1000; held=4 @ $100 = $400 already deployed.
        # Risk sizer must see remaining cap = $1000 - $400 = $600.
        _compute_quantity(
            "AAPL", entry=100.0, stop=95.0,
            max_capital_usd=1000.0, max_shares=None, held_qty=4.0,
        )
        assert recorded.get("cap_seen") == pytest.approx(600.0)

    def test_already_at_cap_short_circuits_risk_branch(self, monkeypatch):
        """If held already meets/exceeds the cap, _compute_quantity must
        return 0 WITHOUT calling the risk sizer at all."""
        from app.services.strategy import scheduler as sched

        called = {"yes": False}

        def _spy(**kwargs):
            called["yes"] = True
            class _R:
                viable = False
                shares = 0.0
                risk_amount = 0.0
            return _R()

        monkeypatch.setattr(sched, "calculate_position_size", _spy)

        qty = _compute_quantity(
            "AAPL", entry=100.0, stop=95.0,
            max_capital_usd=1000.0, max_shares=None, held_qty=15.0,
        )
        assert qty == 0.0
        assert called["yes"] is False, "risk sizer must not be invoked when held >= cap"
