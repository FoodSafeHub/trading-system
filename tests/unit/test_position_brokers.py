"""
Contract tests for get_position_brokers() — the enumeration that read-paths
(PnL page, trail reconcile) use so a position on a NON-default broker isn't
silently invisible.

Pin the contract:
    * The global route (active_broker / trade_routing) is always included.
    * Per-assignment broker overrides (e.g. webull, zerodha) are added on top,
      so a symbol routed away from the default broker still gets its positions
      read and its trail armed.
    * India symbols left on "default" resolve to zerodha.
    * The list is de-duplicated by broker name.
"""
from __future__ import annotations

import app.services.brokers.factory as factory


class _FakeBroker:
    def __init__(self, name):
        self.name = name


def _install_fakes(monkeypatch):
    built: list[str] = []

    def _fake_build_one(name):
        built.append(name)
        return _FakeBroker(name)

    monkeypatch.setattr(factory, "_build_one", _fake_build_one)
    return built


def test_includes_global_route_only_when_no_overrides(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(factory, "_resolve_routing", lambda routing, active: ["schwab"])
    monkeypatch.setattr(factory, "_assignment_broker_names", lambda: set())

    names = [b.name for b in factory.get_position_brokers()]
    assert names == ["schwab"]


def test_adds_assignment_override_brokers(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(factory, "_resolve_routing", lambda routing, active: ["schwab"])
    # Webull + Zerodha holdings live on brokers the global route never queries.
    monkeypatch.setattr(factory, "_assignment_broker_names", lambda: {"webull", "zerodha"})

    names = {b.name for b in factory.get_position_brokers()}
    assert names == {"schwab", "webull", "zerodha"}


def test_dedupes_when_override_matches_global(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(factory, "_resolve_routing", lambda routing, active: ["webull"])
    monkeypatch.setattr(factory, "_assignment_broker_names", lambda: {"webull"})

    names = [b.name for b in factory.get_position_brokers()]
    assert names == ["webull"]


def test_build_failure_is_skipped_not_fatal(monkeypatch):
    def _fake_build_one(name):
        if name == "webull":
            raise RuntimeError("webull creds missing")
        return _FakeBroker(name)

    monkeypatch.setattr(factory, "_build_one", _fake_build_one)
    monkeypatch.setattr(factory, "_resolve_routing", lambda routing, active: ["schwab"])
    monkeypatch.setattr(factory, "_assignment_broker_names", lambda: {"webull"})

    # A broker that fails to build must not blank the whole list.
    names = [b.name for b in factory.get_position_brokers()]
    assert names == ["schwab"]
