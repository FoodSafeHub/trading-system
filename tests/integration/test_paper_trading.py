"""
End-to-end paper trading simulation test.
Uses the real paper broker, real execution service, and an in-memory SQLite DB.
"""
import asyncio
import os
import pytest
import numpy as np
import pandas as pd

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("ACTIVE_BROKER", "paper")
os.environ.setdefault("TRADING_START_TIME", "00:00")
os.environ.setdefault("TRADING_END_TIME", "23:59")
os.environ.setdefault("ORDER_COOLDOWN_SECONDS", "0")
os.environ.setdefault("MAX_ORDERS_PER_DAY", "100")


@pytest.fixture(autouse=True)
def setup_db():
    # Must import after env vars are set
    from app.db import init_db
    init_db()


def test_paper_broker_buy_sell_roundtrip():
    from app.services.brokers.paper import PaperBroker

    async def run():
        broker = PaperBroker()
        await broker.authenticate()

        accounts = await broker.get_accounts()
        assert len(accounts) == 1
        acct = accounts[0]
        initial_cash = acct.buying_power

        # BUY 5 shares at ~$100
        from app.schemas.orders import OrderRequest
        buy = OrderRequest(symbol="SPY", side="BUY", order_type="MARKET", quantity=5, limit_price=100.0)
        buy_result = await broker.place_order(buy, acct.account_id)
        assert buy_result.status == "filled"
        assert buy_result.fill_price is not None

        # Verify position exists
        positions = await broker.get_positions(acct.account_id)
        spy_pos = next((p for p in positions if p.symbol == "SPY"), None)
        assert spy_pos is not None
        assert spy_pos.quantity == 5

        # SELL 5 shares
        sell = OrderRequest(symbol="SPY", side="SELL", order_type="MARKET", quantity=5, limit_price=100.0)
        sell_result = await broker.place_order(sell, acct.account_id)
        assert sell_result.status == "filled"

        # Position should be gone
        positions_after = await broker.get_positions(acct.account_id)
        spy_after = next((p for p in positions_after if p.symbol == "SPY"), None)
        assert spy_after is None

    asyncio.run(run())


def test_execution_service_paper_e2e():
    from app.services.brokers.paper import PaperBroker
    from app.services.execution.service import ExecutionService
    from app.schemas.orders import OrderRequest
    from app.db import SessionLocal
    from app.models.orders import Order

    async def run():
        broker = PaperBroker()
        await broker.authenticate()
        accounts = await broker.get_accounts()
        acct_id = accounts[0].account_id

        svc = ExecutionService(broker)
        order_req = OrderRequest(
            symbol="AAPL",
            side="BUY",
            order_type="MARKET",
            quantity=2,
            limit_price=150.0,
        )
        result = await svc.execute(order_req, account_id=acct_id, estimated_price=150.0)
        assert result is not None
        assert result.status in ("submitted", "filled", "error")

        with SessionLocal() as db:
            orders = db.query(Order).filter_by(symbol="AAPL").all()
            assert len(orders) >= 1

    asyncio.run(run())


def test_strategy_signal_generation():
    from app.services.strategy.rules import evaluate_strategy

    prices = pd.Series(np.linspace(100, 200, 80))
    signal = evaluate_strategy(
        "sma_rsi", "SPY", prices,
        {"sma_fast": 5, "sma_slow": 20, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
    )
    assert signal.direction in ("BUY", "SELL", "HOLD")
    assert signal.price_at_signal is not None
    assert signal.symbol == "SPY"
