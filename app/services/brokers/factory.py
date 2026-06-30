from __future__ import annotations

import logging
from typing import List

from app.config import get_settings
from app.services.brokers.base import BrokerBase

logger = logging.getLogger(__name__)


# Cache of built broker adapters, keyed by broker name. Brokers hold their auth
# state (access token + expiry) in memory; previously get_broker() built a fresh
# instance per request, so that in-memory token was always empty and EVERY call
# re-loaded tokens from the DB (and re-ran the refresh check). Reusing one
# instance per broker lets the cached token survive across requests — the DB
# load happens once, and refresh only fires near real expiry. Account/position
# args are passed per-call, so the adapters carry no request-specific state that
# makes reuse unsafe.
_BROKER_CACHE: dict[str, BrokerBase] = {}


def _build_one(name: str) -> BrokerBase:
    cached = _BROKER_CACHE.get(name)
    if cached is not None:
        return cached

    if name == "paper":
        from app.services.brokers.paper import PaperBroker
        broker: BrokerBase = PaperBroker()
    elif name == "schwab":
        from app.services.brokers.schwab import SchwabBroker
        broker = SchwabBroker()
    elif name == "webull":
        from app.services.brokers.webull import WebullBroker
        broker = WebullBroker()
    elif name == "zerodha":
        from app.services.brokers.zerodha import ZerodhaBroker
        broker = ZerodhaBroker()
    else:
        raise ValueError(f"Unknown broker: {name!r}. Choose: paper | schwab | webull | zerodha")

    _BROKER_CACHE[name] = broker
    return broker


def reset_broker_cache() -> None:
    """Drop cached broker adapters — forces a rebuild (and DB token reload) on the
    next get_broker(). Call after re-running an OAuth flow so the new tokens are
    picked up instead of a stale cached instance."""
    _BROKER_CACHE.clear()


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
