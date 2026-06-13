"""Guard the Zerodha/Kite trailing-stop gap.

Kite Connect has NO native trailing-stop order type (Zerodha removed trailing
SL years ago). Before the fix, ZerodhaBroker._kite_order_type() fell through to
"MARKET" for any unknown type — so a protective TRAILING_STOP became an instant
MARKET SELL that liquidated the position it was meant to guard.

These tests pin the corrected behavior:
  - place_order() rejects TRAILING_STOP loudly (no silent MARKET conversion).
  - _kite_order_type() raises on unknown types instead of defaulting to MARKET.
  - STOP still maps to SL-M (the path the execution layer uses for India).
  - Broker capability flags (supports_native_trailing_stop) are correct, so the
    execution layer routes India to a static STOP and US to a native trail.
"""
from __future__ import annotations

import pytest

from app.schemas.orders import OrderRequest
from app.services.brokers.zerodha import ZerodhaBroker
from app.services.brokers.schwab import SchwabBroker
from app.services.brokers.webull import WebullBroker


class TestZerodhaOrderTypeMapping:
    def test_stop_maps_to_sl_m(self):
        assert ZerodhaBroker._kite_order_type("STOP") == "SL-M"

    def test_stop_limit_maps_to_sl(self):
        assert ZerodhaBroker._kite_order_type("STOP_LIMIT") == "SL"

    def test_market_and_limit_pass_through(self):
        assert ZerodhaBroker._kite_order_type("MARKET") == "MARKET"
        assert ZerodhaBroker._kite_order_type("LIMIT") == "LIMIT"

    def test_unknown_type_raises_not_market(self):
        # The old code returned "MARKET" here — the dangerous default.
        with pytest.raises(ValueError):
            ZerodhaBroker._kite_order_type("SOMETHING_NEW")

    def test_trailing_stop_is_not_silently_market(self):
        with pytest.raises(ValueError):
            ZerodhaBroker._kite_order_type("TRAILING_STOP")


class TestZerodhaPlaceOrderRejectsTrailingStop:
    async def test_place_order_rejects_trailing_stop(self):
        broker = ZerodhaBroker()
        order = OrderRequest(
            symbol="RELIANCE", side="SELL", order_type="TRAILING_STOP",
            quantity=10, trail_type="PERCENT", trail_value=3.0,
        )
        # Must raise BEFORE any network call — never silently become a MARKET sell.
        with pytest.raises(ValueError, match="no native TRAILING_STOP"):
            await broker.place_order(order, account_id="X")


class TestBrokerTrailingCapabilityFlags:
    def test_zerodha_has_no_native_trailing(self):
        assert ZerodhaBroker().supports_native_trailing_stop is False

    def test_schwab_has_native_trailing(self):
        assert SchwabBroker().supports_native_trailing_stop is True

    def test_webull_has_native_trailing(self):
        assert WebullBroker().supports_native_trailing_stop is True
