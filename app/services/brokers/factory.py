from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)


def get_broker() -> BrokerBase:
    """Return the active broker adapter based on ACTIVE_BROKER config."""
    settings = get_settings()
    broker_name = settings.active_broker

    if broker_name == "paper":
        from app.services.brokers.paper import PaperBroker
        return PaperBroker()

    if broker_name == "schwab":
        from app.services.brokers.schwab import SchwabBroker
        return SchwabBroker()

    if broker_name == "webull":
        from app.services.brokers.webull import WebullBroker
        return WebullBroker()

    raise ValueError(f"Unknown broker: {broker_name!r}. Choose: paper | schwab | webull")
