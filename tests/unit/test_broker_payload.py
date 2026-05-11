"""Unit tests for broker-specific payload mapping."""
import pytest
from app.services.brokers.schwab import SchwabBroker
from app.schemas.orders import OrderRequest


def _broker() -> SchwabBroker:
    return SchwabBroker()


class TestSchwabPayloadMapping:
    def test_market_buy_payload(self):
        broker = _broker()
        order = OrderRequest(symbol="AAPL", side="BUY", order_type="MARKET", quantity=10)
        payload = broker._build_order_payload(order)

        assert payload["orderType"] == "MARKET"
        assert payload["orderStrategyType"] == "SINGLE"
        leg = payload["orderLegCollection"][0]
        assert leg["instruction"] == "BUY"
        assert leg["quantity"] == 10
        assert leg["instrument"]["symbol"] == "AAPL"
        assert leg["instrument"]["assetType"] == "EQUITY"
        assert "price" not in payload  # no limit price for MARKET

    def test_limit_sell_payload(self):
        broker = _broker()
        order = OrderRequest(symbol="MSFT", side="SELL", order_type="LIMIT", quantity=5, limit_price=300.0)
        payload = broker._build_order_payload(order)

        assert payload["orderType"] == "LIMIT"
        assert payload["price"] == "300.0"
        leg = payload["orderLegCollection"][0]
        assert leg["instruction"] == "SELL"

    def test_stop_limit_payload(self):
        broker = _broker()
        order = OrderRequest(
            symbol="SPY", side="SELL", order_type="STOP_LIMIT",
            quantity=3, limit_price=440.0, stop_price=445.0
        )
        payload = broker._build_order_payload(order)
        assert payload["orderType"] == "STOP_LIMIT"
        assert payload["price"] == "440.0"
        assert payload["stopPrice"] == "445.0"

    def test_symbol_is_uppercase(self):
        order = OrderRequest(symbol="aapl", side="BUY", order_type="MARKET", quantity=1)
        assert order.symbol == "AAPL"

    def test_time_in_force_in_payload(self):
        broker = _broker()
        order = OrderRequest(symbol="SPY", side="BUY", order_type="MARKET", quantity=1, time_in_force="GTC")
        payload = broker._build_order_payload(order)
        assert payload["duration"] == "GTC"
