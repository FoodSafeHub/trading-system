from __future__ import annotations

import logging
from typing import List

from app.config import get_settings
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)


def _build_one(name: str) -> BrokerBase:
    if name == "paper":
        from app.services.brokers.paper import PaperBroker
        return PaperBroker()
    if name == "schwab":
        from app.services.brokers.schwab import SchwabBroker
        return SchwabBroker()
    if name == "webull":
        from app.services.brokers.webull import WebullBroker
        return WebullBroker()
    if name == "zerodha":
        from app.services.brokers.zerodha import ZerodhaBroker
        return ZerodhaBroker()
    raise ValueError(f"Unknown broker: {name!r}. Choose: paper | schwab | webull | zerodha")


def _resolve_routing(routing: str, active_broker: str) -> List[str]:
    """Translate the trade_routing config into a concrete list of broker names."""
    if routing == "auto":
        return [active_broker]
    if routing == "both":
        return ["schwab", "webull"]
    return [routing]


def get_broker() -> BrokerBase:
    """Return the active broker adapter based on trade_routing / active_broker config.

    - trade_routing="auto" (default): use active_broker (backwards compatible).
    - trade_routing="schwab"|"webull"|"paper": use that single broker.
    - trade_routing="both": fan out every order to Schwab AND Webull via MultiBroker.
    """
    settings = get_settings()
    names = _resolve_routing(settings.trade_routing, settings.active_broker)
    if len(names) == 1:
        return _build_one(names[0])
    from app.services.brokers.multi import MultiBroker
    return MultiBroker([_build_one(n) for n in names])
