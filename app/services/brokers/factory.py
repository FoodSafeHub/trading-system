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


def get_position_brokers() -> List[BrokerBase]:
    """Every distinct broker that could currently hold a position.

    The global broker (get_broker() → active_broker / trade_routing) only covers
    the default route. But per-assignment `broker` overrides route individual
    symbols elsewhere — India symbols to Zerodha, some US symbols explicitly to
    Webull — and those holdings live on a broker the global route never queries.
    Read-paths that answer "what do we actually hold / what stops are resting"
    (PnL page, trail reconcile) must enumerate ALL of them, or a position on a
    non-default broker is silently invisible (no trail armed, absent from PnL).

    Returns a de-duplicated list of concrete single-broker adapters. The global
    broker is expanded into its underlying legs (so a MultiBroker becomes its
    Schwab + Webull members) to keep list_orders/get_positions on real adapters
    that fan out correctly. Best-effort: a broker that fails to build is skipped
    with a warning rather than blanking the whole list.
    """
    out: List[BrokerBase] = []
    seen: set[str] = set()

    def _add(b: BrokerBase) -> None:
        name = getattr(b, "name", "")
        if name and name not in seen:
            seen.add(name)
            out.append(b)

    settings = get_settings()
    # Global route — expand MultiBroker into its legs so reads hit real adapters.
    for name in _resolve_routing(settings.trade_routing, settings.active_broker):
        try:
            _add(_build_one(name))
        except Exception as exc:
            logger.warning("[factory] get_position_brokers: could not build %r: %s", name, exc)

    # Per-assignment broker overrides (e.g. zerodha for India, explicit webull).
    for name in _assignment_broker_names():
        try:
            _add(_build_one(name))
        except Exception as exc:
            logger.warning("[factory] get_position_brokers: could not build %r: %s", name, exc)

    return out


def _assignment_broker_names() -> set[str]:
    """Distinct concrete broker names referenced by enabled assignments.

    An assignment's broker is "default" (follow the global route — already
    covered by get_broker) or a concrete name. India symbols left on "default"
    still route to Zerodha, so resolve those too.
    """
    names: set[str] = set()
    try:
        from app.db import SessionLocal
        from app.models.assignments import SymbolStrategyAssignment
        from app.services.markets import is_india_symbol

        with SessionLocal() as db:
            for a in db.query(SymbolStrategyAssignment).filter_by(enabled=True).all():
                broker = (a.broker or "default").lower()
                if broker != "default":
                    names.add(broker)
                elif is_india_symbol(a.symbol):
                    names.add("zerodha")
    except Exception as exc:
        logger.debug("[factory] _assignment_broker_names failed: %s", exc)
    return names
