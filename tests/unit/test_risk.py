"""Unit tests for the risk engine."""
import pytest
from unittest.mock import MagicMock, patch

from app.schemas.orders import OrderRequest
from app.services.risk.engine import RiskEngine
from app.config import Settings

_MARKET_HOURS_PATCH = "app.services.risk.engine.is_market_hours"


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        active_broker="paper",
        live_trading_enabled=False,
        live_trading_confirmed=False,
        max_position_size_usd=5000,
        max_daily_loss_usd=500,
        max_orders_per_day=10,
        order_cooldown_seconds=0,
        trading_start_time="00:00",
        trading_end_time="23:59",
        trading_timezone="America/New_York",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _order(symbol: str = "SPY", side: str = "BUY", qty: float = 1.0, price: float = 100.0) -> OrderRequest:
    return OrderRequest(symbol=symbol, side=side, order_type="MARKET", quantity=qty, limit_price=price)


class TestRiskEngine:
    def _engine(self, **settings_overrides) -> RiskEngine:
        s = _make_settings(**settings_overrides)
        engine = RiskEngine(settings=s)
        # Patch DB calls to avoid needing a real DB in unit tests
        engine.is_kill_switch_active = lambda: False
        engine._orders_today = lambda: 0
        engine._daily_realized_loss = lambda: 0.0
        engine._last_order_time = lambda sym: None
        engine._is_duplicate = lambda key: False
        return engine

    def test_paper_order_passes(self):
        engine = self._engine()
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order(), estimated_price=100.0)
        assert result.passed

    def test_kill_switch_blocks(self):
        engine = self._engine()
        engine.is_kill_switch_active = lambda: True
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order())
        assert not result.passed
        assert "Kill switch" in result.blocked_reason

    def test_live_trading_blocked_without_flag(self):
        engine = self._engine(active_broker="schwab", live_trading_enabled=False)
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order())
        assert not result.passed
        assert "LIVE_TRADING_ENABLED" in result.blocked_reason

    def test_max_position_size_blocks(self):
        engine = self._engine(max_position_size_usd=100)
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order(qty=10, price=200), estimated_price=200.0)
        assert not result.passed
        assert "exceeds max position size" in result.blocked_reason

    def test_daily_order_limit_blocks(self):
        engine = self._engine(max_orders_per_day=5)
        engine._orders_today = lambda: 5
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order())
        assert not result.passed
        assert "Daily order limit" in result.blocked_reason

    def test_duplicate_order_blocked(self):
        engine = self._engine()
        engine._is_duplicate = lambda key: True
        o = _order()
        o.idempotency_key = "test-key-123"
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(o)
        assert not result.passed
        assert "Duplicate" in result.blocked_reason

    def test_warning_near_position_limit(self):
        engine = self._engine(max_position_size_usd=100)
        # Order value = 85 (85% of 100 limit) → warning but passes
        with patch(_MARKET_HOURS_PATCH, return_value=True):
            result = engine.check(_order(qty=1, price=85), estimated_price=85.0)
        assert result.passed
        assert any("80%" in w for w in result.warnings)
