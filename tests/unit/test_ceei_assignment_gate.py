"""Assignment-level CEEI gate — end-to-end plumbing tests.

Pins the pieces that carry a per-assignment CEEI gate from the DB row to
evaluate_strategy:
  - DB migration adds the four ceei_* columns to a legacy table (idempotent).
  - Assignments API round-trips the fields, validates values, and the PATCH
    /ceei endpoint sets/clears them and warns on incompatible families.
  - ceei_overrides_from_assignment builds the exact params overlay (empty
    dict = no-op for unconfigured assignments).
  - apply_ceei_gate duck-types onto PerplexitySignal (the scheduler's
    perplexity path applies the gate post-signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.db as app_db
from app.api.routes.assignments import router as assignments_router
from app.db import Base, get_db
from app.models.assignments import SymbolStrategyAssignment
from app.services.strategy.perplexity.base import PerplexitySignal
from app.services.strategy.rules import (
    apply_ceei_gate,
    ceei_overrides_from_assignment,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared in-memory DB across connections
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def _override_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app = FastAPI()
    app.include_router(assignments_router)
    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as c:
        yield c


def _upsert(client, **extra):
    body = {"symbol": "NVDA", "system": "scanner",
            "strategy_name": "Momentum Breakout", **extra}
    return client.post("/assignments", json=body)


# ──────────────────────────────────────────────────────────────────────────────
# Migration
# ──────────────────────────────────────────────────────────────────────────────

def test_migration_adds_ceei_columns_to_legacy_table(monkeypatch, tmp_path):
    """A pre-CEEI table gains the four columns; running twice is a no-op."""
    legacy = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with legacy.connect() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE symbol_strategy_assignments (
                symbol VARCHAR(16) NOT NULL,
                system VARCHAR(32) NOT NULL,
                strategy_name VARCHAR(128) NOT NULL,
                enabled BOOLEAN,
                max_capital_usd FLOAT,
                max_shares FLOAT,
                broker VARCHAR(32) NOT NULL DEFAULT 'default',
                notes VARCHAR(256),
                tight_trail_pct FLOAT,
                approach_c_enabled INTEGER,
                assigned_at DATETIME,
                PRIMARY KEY (symbol, system, strategy_name)
            )
            """
        )
        conn.exec_driver_sql(
            "INSERT INTO symbol_strategy_assignments "
            "(symbol, system, strategy_name, enabled) VALUES ('AAPL','scanner','X',1)"
        )
        conn.commit()

    monkeypatch.setattr(app_db, "engine", legacy)
    app_db._migrate_add_assignments_ceei_columns()
    app_db._migrate_add_assignments_ceei_columns()  # idempotent

    with legacy.connect() as conn:
        cols = {r[1] for r in conn.exec_driver_sql(
            "PRAGMA table_info(symbol_strategy_assignments)").fetchall()}
        assert {"ceei_gate", "ceei_gate_enabled",
                "ceei_gate_threshold", "ceei_gate_lookback"} <= cols
        # Existing row survives with the gate fully NULL (inert).
        row = conn.exec_driver_sql(
            "SELECT ceei_gate, ceei_gate_enabled, ceei_gate_threshold, "
            "ceei_gate_lookback FROM symbol_strategy_assignments").fetchone()
        assert tuple(row) == (None, None, None, None)


# ──────────────────────────────────────────────────────────────────────────────
# API — create / update / read / validation
# ──────────────────────────────────────────────────────────────────────────────

class TestAssignmentsApi:
    def test_upsert_without_ceei_fields_defaults_null(self, client):
        r = _upsert(client)
        assert r.status_code == 200
        out = r.json()
        assert out["ceei_gate"] is None
        assert out["ceei_gate_enabled"] is None
        assert out["ceei_gate_threshold"] is None
        assert out["ceei_gate_lookback"] is None

    def test_upsert_roundtrips_ceei_fields(self, client):
        r = _upsert(client, ceei_gate="score", ceei_gate_enabled=True,
                    ceei_gate_threshold=55.0, ceei_gate_lookback=12)
        assert r.status_code == 200
        listed = client.get("/assignments").json()
        assert len(listed) == 1
        a = listed[0]
        assert a["ceei_gate"] == "score"
        assert a["ceei_gate_enabled"] is True
        assert a["ceei_gate_threshold"] == 55.0
        assert a["ceei_gate_lookback"] == 12

    def test_upsert_gate_none_normalizes_to_null(self, client):
        r = _upsert(client, ceei_gate="none")
        assert r.status_code == 200
        assert r.json()["ceei_gate"] is None

    def test_upsert_rejects_bad_gate_value(self, client):
        assert _upsert(client, ceei_gate="bogus").status_code == 400

    def test_upsert_rejects_bad_threshold_and_lookback(self, client):
        assert _upsert(client, ceei_gate="score",
                       ceei_gate_threshold=150.0).status_code == 400
        assert _upsert(client, ceei_gate="setup",
                       ceei_gate_lookback=0).status_code == 400

    def test_patch_ceei_sets_and_clears(self, client):
        _upsert(client)
        r = client.patch("/assignments/NVDA/ceei",
                         json={"ceei_gate": "trigger", "ceei_gate_enabled": True})
        assert r.status_code == 200
        assert r.json()["ceei_gate"] == "trigger"
        assert r.json()["warning"] is None
        # Clearing with "none" nulls every column.
        r = client.patch("/assignments/NVDA/ceei", json={"ceei_gate": "none"})
        assert r.status_code == 200
        out = r.json()
        assert out["ceei_gate"] is None
        assert out["ceei_gate_enabled"] is None
        assert out["ceei_gate_threshold"] is None
        assert out["ceei_gate_lookback"] is None

    def test_patch_ceei_warns_on_incompatible_family_but_saves(self, client):
        body = {"symbol": "KO", "system": "scanner",
                "strategy_name": "rsi2_mean_reversion"}
        assert client.post("/assignments", json=body).status_code == 200
        r = client.patch(
            "/assignments/KO/ceei",
            params={"system": "scanner", "strategy_name": "rsi2_mean_reversion"},
            json={"ceei_gate": "trigger"},
        )
        assert r.status_code == 200
        assert r.json()["ceei_gate"] == "trigger"   # saved anyway
        assert "REDUCES expectancy" in (r.json()["warning"] or "")

    def test_patch_ceei_validates(self, client):
        _upsert(client)
        assert client.patch("/assignments/NVDA/ceei",
                            json={"ceei_gate": "wat"}).status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# Params overlay (what the scheduler merges into evaluate_strategy)
# ──────────────────────────────────────────────────────────────────────────────

class TestOverridesBuilder:
    def test_unconfigured_assignment_yields_empty_overlay(self):
        for asgn in ({}, {"ceei_gate": None}, {"ceei_gate": "none"},
                     {"ceei_gate": "", "ceei_gate_threshold": 55.0}):
            assert ceei_overrides_from_assignment(asgn) == {}

    def test_full_overlay(self):
        ov = ceei_overrides_from_assignment({
            "ceei_gate": "score", "ceei_gate_enabled": False,
            "ceei_gate_threshold": 55, "ceei_gate_lookback": 12,
        })
        assert ov == {"ceei_gate": "score", "ceei_gate_enabled": False,
                      "ceei_gate_threshold": 55.0, "ceei_gate_lookback": 12}

    def test_partial_overlay_omits_unset_keys(self):
        ov = ceei_overrides_from_assignment({"ceei_gate": "trigger"})
        assert ov == {"ceei_gate": "trigger"}

    def test_merge_leaves_config_params_intact(self):
        base = {"rsi_period": 14, "sma_fast": 10}
        merged = {**base, **ceei_overrides_from_assignment({"ceei_gate": "setup"})}
        assert merged["rsi_period"] == 14 and merged["ceei_gate"] == "setup"
        assert "ceei_gate" not in base  # base dict untouched


# ──────────────────────────────────────────────────────────────────────────────
# Perplexity path — gate duck-types onto PerplexitySignal
# ──────────────────────────────────────────────────────────────────────────────

def test_apply_ceei_gate_on_perplexity_signal():
    np.random.seed(3)
    n = 300
    closes = 100 + np.cumsum(np.random.normal(0.0, 0.8, n))
    rng = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Open": closes - rng * 0.2, "High": closes + rng * 0.5,
                       "Low": closes - rng * 0.5, "Close": closes,
                       "Volume": np.full(n, 1_000_000.0)}, index=idx)
    sig = PerplexitySignal(symbol="TEST", strategy_name="Unified_Trend_Follow",
                           direction="BUY", entry_price=float(closes[-1]))
    out = apply_ceei_gate(sig, df["Close"], df,
                          ceei_overrides_from_assignment({"ceei_gate": "trigger"}))
    # A drift-free random walk almost never triggers → vetoed to HOLD.
    assert out.direction == "HOLD"
    assert out.indicators["ceei_gate_veto"] is True
    # SELL signals must pass through untouched.
    sell = PerplexitySignal(symbol="TEST", strategy_name="Unified_Trend_Follow",
                            direction="SELL")
    out2 = apply_ceei_gate(sell, df["Close"], df, {"ceei_gate": "trigger"})
    assert out2.direction == "SELL"


def test_ceei_gated_strategy_wrapper_delegates_and_vetoes():
    """CeeiGatedStrategy (used by the India-swing backtest route) must delegate
    attributes to the wrapped strategy and veto BUYs exactly like the live gate."""
    from app.services.strategy.perplexity.base import CeeiGatedStrategy, PerplexityStrategy

    np.random.seed(3)
    n = 300
    closes = 100 + np.cumsum(np.random.normal(0.0, 0.8, n))
    rng = np.abs(np.diff(closes, prepend=closes[0])) + closes * 0.005
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Open": closes - rng * 0.2, "High": closes + rng * 0.5,
                       "Low": closes - rng * 0.5, "Close": closes,
                       "Volume": np.full(n, 1_000_000.0)}, index=idx)

    class AlwaysBuy(PerplexityStrategy):
        name = "always_buy"
        config = {"max_hold_bars": 30}

        def run(self, symbol, d, **kwargs):
            return PerplexitySignal(symbol=symbol, strategy_name=self.name,
                                    direction="BUY", entry_price=float(d["Close"].iloc[-1]))

    wrapped = CeeiGatedStrategy(AlwaysBuy(), {"ceei_gate": "trigger"})
    # Attribute delegation (the backtest engine reads these off the strategy).
    assert wrapped.name == "always_buy"
    assert wrapped.config["max_hold_bars"] == 30
    assert wrapped.enabled is True
    # Drift-free random walk almost never triggers → BUY vetoed to HOLD; the
    # engine's extra kwargs must pass through without error.
    out = wrapped.run("TEST", df, regime=None, volatility_bucket="mid")
    assert out.direction == "HOLD"
    assert out.indicators["ceei_gate_veto"] is True


def test_model_columns_exist():
    cols = {c.name for c in SymbolStrategyAssignment.__table__.columns}
    assert {"ceei_gate", "ceei_gate_enabled",
            "ceei_gate_threshold", "ceei_gate_lookback"} <= cols
