"""Flat (qty 0) lots must never show as open positions.

Brokers (Schwab especially) keep recently-CLOSED lots in their positions array
with net quantity 0. Before the fix these rendered as "open" positions on the
home page — a symbol you already sold (e.g. ABNB) kept showing as held.

These tests pin the filter at two layers:
  - each broker adapter's get_positions() drops qty-0 lots, and
  - the /account/positions endpoint helper drops them as defense in depth.
"""
from __future__ import annotations

import pytest

from app.schemas.account import Position


def test_schwab_skips_zero_quantity_lots():
    from app.services.brokers.schwab import SchwabBroker

    raw = {
        "securitiesAccount": {
            "positions": [
                {  # held
                    "instrument": {"symbol": "AAPL"},
                    "longQuantity": 10, "shortQuantity": 0,
                    "averagePrice": 100.0, "marketValue": 1100.0,
                    "unrealizedProfitOrLoss": 100.0,
                },
                {  # SOLD — net qty 0, must be skipped
                    "instrument": {"symbol": "ABNB"},
                    "longQuantity": 0, "shortQuantity": 0,
                    "averagePrice": 130.0, "marketValue": 0.0,
                    "unrealizedProfitOrLoss": 0.0,
                },
            ]
        }
    }

    b = SchwabBroker.__new__(SchwabBroker)

    async def _fake_get(path, params=None):
        return raw

    async def _fake_hash():
        return "HASH"

    b._get = _fake_get          # type: ignore[attr-defined]
    b._get_account_hash = _fake_hash  # type: ignore[attr-defined]

    import asyncio
    positions = asyncio.run(b.get_positions("ACCT"))
    syms = {p.symbol for p in positions}
    assert "AAPL" in syms
    assert "ABNB" not in syms      # the sold lot is gone
    assert len(positions) == 1


def test_endpoint_helper_drops_flat_lots():
    from app.api.routes.account import _held_only

    rows = [
        Position(symbol="AAPL", quantity=10),
        Position(symbol="ABNB", quantity=0),     # sold
        Position(symbol="MSFT", quantity=-5),    # short — still a position
    ]
    out = _held_only(rows)
    syms = {p.symbol for p in out}
    assert syms == {"AAPL", "MSFT"}


def test_zerodha_skips_zero_quantity_holdings():
    from app.services.brokers.zerodha import ZerodhaBroker

    b = ZerodhaBroker.__new__(ZerodhaBroker)

    async def _fake_get(path, params=None):
        return [
            {"tradingsymbol": "RELIANCE", "quantity": 5, "average_price": 2900.0,
             "last_price": 3000.0, "pnl": 500.0},
            {"tradingsymbol": "AIIL", "quantity": 0, "average_price": 100.0,
             "last_price": 110.0, "pnl": 0.0},
        ]

    b._get = _fake_get  # type: ignore[attr-defined]

    import asyncio
    positions = asyncio.run(b.get_positions("ACCT"))
    syms = {p.symbol for p in positions}
    assert syms == {"RELIANCE"}
