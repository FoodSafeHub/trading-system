"""Analyst-ratings service + route tests (mocked yfinance).

Pins:
  - fetch_rating parses info / recommendations_summary / upgrades_downgrades
    into the flat dict the cache stores, incl. upside math.
  - A symbol with no coverage (NSE) degrades to a row with note, no exception.
  - upsert overwrites the existing row (one row per symbol).
  - The /ratings route serializes cached rows, holdings first.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.analyst_ratings import AnalystRating
from app.services.research import analyst_ratings as svc


class _FakeTicker:
    def __init__(self, info=None, summary=None, upgrades=None):
        self.info = info or {}
        self.recommendations_summary = summary
        self.upgrades_downgrades = upgrades


def _full_ticker():
    info = {
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 30,
        "targetMeanPrice": 120.0,
        "targetHighPrice": 150.0,
        "targetLowPrice": 90.0,
        "targetMedianPrice": 118.0,
        "currentPrice": 100.0,
    }
    summary = pd.DataFrame([
        {"period": "0m", "strongBuy": 10, "buy": 12, "hold": 6, "sell": 1, "strongSell": 1},
        {"period": "-1m", "strongBuy": 9, "buy": 11, "hold": 7, "sell": 2, "strongSell": 1},
    ])
    upgrades = pd.DataFrame(
        {
            "Firm": ["Morgan Stanley", "Goldman Sachs"],
            "Action": ["up", "main"],
            "FromGrade": ["Equal-Weight", ""],
            "ToGrade": ["Overweight", "Buy"],
        },
        index=pd.to_datetime(["2026-07-10", "2026-06-02"]),
    )
    return _FakeTicker(info, summary, upgrades)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[AnalystRating.__table__])
    TestSession = sessionmaker(bind=engine)
    with TestSession() as db:
        yield db


def test_fetch_rating_parses_everything():
    with patch("yfinance.Ticker", return_value=_full_ticker()):
        out = svc.fetch_rating("AAPL")
    assert out["symbol"] == "AAPL"
    assert out["recommendation_key"] == "buy"
    assert out["analyst_count"] == 30
    assert out["target_mean"] == 120.0
    assert out["upside_pct"] == pytest.approx(20.0)   # 120 vs 100
    assert (out["strong_buy"], out["buy"], out["hold"],
            out["sell"], out["strong_sell"]) == (10, 12, 6, 1, 1)
    ups = json.loads(out["upgrades_json"])
    assert ups[0]["firm"] == "Morgan Stanley" and ups[0]["date"] == "2026-07-10"
    assert "note" not in out or out["note"] is None or "unavailable" not in out["note"]


def test_fetch_rating_no_coverage_degrades_gracefully():
    with patch("yfinance.Ticker", return_value=_FakeTicker(info={})):
        out = svc.fetch_rating("AIIL")
    assert out["note"] == "no analyst coverage"
    assert out.get("recommendation_key") is None


def test_fetch_rating_never_raises_on_accessor_failure():
    class _Broken:
        @property
        def info(self):
            raise RuntimeError("rate limited")

        @property
        def recommendations_summary(self):
            raise RuntimeError("rate limited")

        @property
        def upgrades_downgrades(self):
            raise RuntimeError("rate limited")

    with patch("yfinance.Ticker", return_value=_Broken()):
        out = svc.fetch_rating("AAPL")
    assert out["note"] == "no analyst coverage"


def test_upsert_overwrites_single_row(db_session):
    svc.upsert_rating(db_session, {"symbol": "AAPL", "recommendation_key": "hold"},
                      is_holding=True, is_assigned=False)
    db_session.commit()
    svc.upsert_rating(db_session, {"symbol": "AAPL", "recommendation_key": "buy"},
                      is_holding=True, is_assigned=True)
    db_session.commit()
    rows = db_session.query(AnalystRating).all()
    assert len(rows) == 1
    assert rows[0].recommendation_key == "buy"
    assert rows[0].is_assigned is True


def test_refresh_all_isolates_per_symbol_failures(db_session):
    def _fetch(sym):
        if sym == "BAD":
            raise RuntimeError("boom")
        return {"symbol": sym, "recommendation_key": "buy"}

    with patch.object(svc, "rating_universe", return_value={
        "AAPL": {"is_holding": True, "is_assigned": False},
        "BAD": {"is_holding": False, "is_assigned": True},
    }), patch.object(svc, "fetch_rating", side_effect=_fetch):
        out = svc.refresh_all(db_session)
    assert out["refreshed"] == 1
    assert len(out["errors"]) == 1 and out["errors"][0].startswith("BAD")


def test_route_lists_holdings_first(db_session):
    from app.api.routes.analyst_ratings import list_ratings

    svc.upsert_rating(db_session, {"symbol": "ZZZ", "recommendation_key": "buy"},
                      is_holding=True, is_assigned=False)
    svc.upsert_rating(db_session, {"symbol": "AAA", "recommendation_key": "hold"},
                      is_holding=False, is_assigned=True)
    db_session.commit()

    out = list_ratings(db=db_session)
    assert [r.symbol for r in out] == ["ZZZ", "AAA"]   # holding sorts first
    assert out[0].is_holding is True
