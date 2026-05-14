from app.services.broker.base_broker import BaseBroker
from app.services.broker.models import Account, Order, Position
from app.services.broker.order_manager import OrderManager, get_order_manager
from app.services.broker.paper_broker import PaperBroker

__all__ = [
    "BaseBroker",
    "PaperBroker",
    "Order",
    "Position",
    "Account",
    "OrderManager",
    "get_order_manager",
]


def get_broker(broker_name: str = "paper", **kwargs) -> BaseBroker:
    """
    Factory function. Returns the configured broker.
    broker_name: "paper" | "alpaca"
    """
    if broker_name == "alpaca":
        from app.services.broker.alpaca_broker import AlpacaBroker
        return AlpacaBroker()
    return PaperBroker(**kwargs)
