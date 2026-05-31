"""
Tests for market_movers.py

Covers:
- normalize_mover_rows: valid rows, bad rows, unknown list_type handling
- MarketMoverRow.__post_init__: dollar_volume + rvol auto-computation
- merge_duplicate_symbols: memberships, rank tracking, priority resolution,
  strategy hints, directional bias, prominence score
- symbol_list_memberships: present/absent symbol lookup
- _resolve_bias: all bias combinations
- _compute_prominence: rank math
"""
from __future__ import annotations

import pytest

from app.services.strategy.daytrading.market_movers import (
    LIST_TYPE_DIRECTION,
    LIST_TYPE_STRATEGY_HINTS,
    MarketMoverRow,
    MarketMoverSnapshot,
    SymbolMoverContext,
    _compute_prominence,
    _resolve_bias,
    merge_duplicate_symbols,
    normalize_mover_rows,
    symbol_list_memberships,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _row(symbol="AAPL", list_type="top_gainers", rank=1,
         price=150.0, pct_change=3.5, volume=500_000.0, **kw) -> dict:
    return dict(
        symbol=symbol, list_type=list_type, rank=rank,
        price=price, pct_change=pct_change, volume=volume, **kw
    )


# ── normalize_mover_rows ──────────────────────────────────────────────────────

class TestNormalizeMoverRows:
    def test_basic_valid_row(self):
        rows = [_row()]
        result = normalize_mover_rows(rows)
        assert len(result) == 1
        r = result[0]
        assert r.symbol == "AAPL"
        assert r.list_type == "top_gainers"
        assert r.rank == 1
        assert r.price == pytest.approx(150.0)
        assert r.pct_change == pytest.approx(3.5)
        assert r.volume == pytest.approx(500_000.0)

    def test_symbol_uppercased(self):
        result = normalize_mover_rows([_row(symbol="aapl")])
        assert result[0].symbol == "AAPL"

    def test_dollar_volume_auto_computed(self):
        # price=100, volume=1000 → dollar_volume=100_000
        result = normalize_mover_rows([_row(price=100.0, volume=1000.0)])
        assert result[0].dollar_volume == pytest.approx(100_000.0)

    def test_dollar_volume_not_overwritten_when_provided(self):
        result = normalize_mover_rows([_row(dollar_volume=999.0)])
        assert result[0].dollar_volume == pytest.approx(999.0)

    def test_rvol_auto_computed(self):
        result = normalize_mover_rows([_row(volume=1_000_000.0, avg_volume=500_000.0)])
        assert result[0].rvol == pytest.approx(2.0)

    def test_rvol_not_overwritten_when_provided(self):
        result = normalize_mover_rows([_row(rvol=5.0, volume=0.0, avg_volume=0.0)])
        assert result[0].rvol == pytest.approx(5.0)

    def test_unknown_list_type_dropped_by_default(self):
        rows = [_row(list_type="made_up")]
        result = normalize_mover_rows(rows)
        assert len(result) == 0

    def test_unknown_list_type_kept_when_flag_false(self):
        rows = [_row(list_type="made_up")]
        result = normalize_mover_rows(rows, drop_unknown_list_types=False)
        assert len(result) == 1
        assert result[0].list_type == "most_active"   # falls back to most_active

    def test_empty_symbol_dropped(self):
        rows = [_row(symbol="")]
        result = normalize_mover_rows(rows)
        assert len(result) == 0

    def test_malformed_row_skipped(self):
        rows = [{"symbol": "AAPL", "list_type": "top_gainers",
                 "rank": "not_an_int", "price": "oops", "pct_change": 1.0, "volume": 0.0}]
        # Should not raise — just drop the malformed row
        result = normalize_mover_rows(rows)
        # rank=int("not_an_int") raises ValueError → row skipped
        assert len(result) == 0

    def test_multiple_valid_list_types(self):
        rows = [
            _row(list_type="top_gainers"),
            _row(list_type="top_losers"),
            _row(list_type="most_active"),
            _row(list_type="unusual_volume"),
            _row(list_type="gap_up"),
            _row(list_type="gap_down"),
        ]
        result = normalize_mover_rows(rows)
        assert len(result) == 6

    def test_source_and_timestamp_propagated(self):
        result = normalize_mover_rows(
            [_row()], source="webull", timestamp="2024-01-02T09:30:00+00:00"
        )
        assert result[0].source == "webull"
        assert result[0].timestamp == "2024-01-02T09:30:00+00:00"

    def test_row_level_source_overrides_global(self):
        rows = [_row(source="manual")]
        result = normalize_mover_rows(rows, source="webull")
        assert result[0].source == "manual"

    def test_has_news_and_tags(self):
        rows = [_row(has_news=True, news_tags=["earnings", "upgrade"])]
        result = normalize_mover_rows(rows)
        assert result[0].has_news is True
        assert "earnings" in result[0].news_tags

    def test_returns_empty_for_empty_input(self):
        assert normalize_mover_rows([]) == []


# ── merge_duplicate_symbols ───────────────────────────────────────────────────

class TestMergeDuplicateSymbols:
    def _snapshot(self, rows: list[dict]) -> MarketMoverSnapshot:
        normalized = normalize_mover_rows(rows)
        return MarketMoverSnapshot(rows=normalized)

    def test_single_symbol_single_list(self):
        snap = self._snapshot([_row(symbol="AAPL", list_type="top_gainers", rank=3)])
        ctx = merge_duplicate_symbols(snap)
        assert "AAPL" in ctx
        c = ctx["AAPL"]
        assert c.list_memberships == ["top_gainers"]
        assert c.best_rank_per_list == {"top_gainers": 3}

    def test_symbol_on_two_lists(self):
        snap = self._snapshot([
            _row(symbol="NVDA", list_type="top_gainers", rank=1, pct_change=5.0),
            _row(symbol="NVDA", list_type="unusual_volume", rank=5, pct_change=4.8),
        ])
        ctx = merge_duplicate_symbols(snap)
        c = ctx["NVDA"]
        assert "top_gainers" in c.list_memberships
        assert "unusual_volume" in c.list_memberships
        assert len(c.list_memberships) == 2

    def test_best_rank_per_list_tracks_minimum(self):
        # Same list, two entries with different ranks → keep the best (lowest)
        snap = self._snapshot([
            _row(symbol="TSLA", list_type="most_active", rank=10),
            _row(symbol="TSLA", list_type="most_active", rank=2),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["TSLA"].best_rank_per_list["most_active"] == 2

    def test_volume_takes_maximum(self):
        snap = self._snapshot([
            _row(symbol="AMD", list_type="top_gainers", rank=1, volume=1_000_000),
            _row(symbol="AMD", list_type="unusual_volume", rank=3, volume=3_000_000),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["AMD"].volume == pytest.approx(3_000_000.0)

    def test_spread_takes_maximum_worst_case(self):
        snap = self._snapshot([
            _row(symbol="HOOD", list_type="top_gainers", rank=5, spread_pct=0.10),
            _row(symbol="HOOD", list_type="most_active", rank=2, spread_pct=0.25),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["HOOD"].spread_pct == pytest.approx(0.25)

    def test_price_from_highest_priority_list(self):
        # top_gainers (priority 1) wins over most_active (priority 6)
        snap = self._snapshot([
            _row(symbol="MSFT", list_type="most_active", rank=1, price=400.0),
            _row(symbol="MSFT", list_type="top_gainers", rank=3, price=401.5),
        ])
        ctx = merge_duplicate_symbols(snap)
        # top_gainers row should be primary despite higher rank number
        assert ctx["MSFT"].price == pytest.approx(401.5)

    def test_has_news_union(self):
        snap = self._snapshot([
            _row(symbol="X", list_type="top_gainers", rank=1, has_news=False),
            _row(symbol="X", list_type="unusual_volume", rank=2, has_news=True,
                 news_tags=["earnings"]),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["X"].has_news is True
        assert "earnings" in ctx["X"].news_tags

    def test_strategy_hints_union(self):
        snap = self._snapshot([
            _row(symbol="SPY", list_type="top_gainers", rank=1),
            _row(symbol="SPY", list_type="gap_up", rank=5),
        ])
        ctx = merge_duplicate_symbols(snap)
        hints = ctx["SPY"].strategy_hints
        # top_gainers → ORBBreakout, VWAPMeanReversion, EMAMomentum
        # gap_up → ORBBreakout, OpeningGapFade, NRSqueezeBreakout
        assert "ORBBreakout" in hints
        assert "OpeningGapFade" in hints

    def test_directional_bias_long_only(self):
        snap = self._snapshot([
            _row(symbol="QQQ", list_type="top_gainers", rank=1),
            _row(symbol="QQQ", list_type="gap_up", rank=2),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["QQQ"].directional_bias == "long"

    def test_directional_bias_short_only(self):
        snap = self._snapshot([
            _row(symbol="XYZ", list_type="top_losers", rank=1),
            _row(symbol="XYZ", list_type="gap_down", rank=2),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["XYZ"].directional_bias == "short"

    def test_directional_bias_neutral_when_mixed(self):
        snap = self._snapshot([
            _row(symbol="WEIRD", list_type="top_gainers", rank=1),  # long
            _row(symbol="WEIRD", list_type="top_losers", rank=2),   # short
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["WEIRD"].directional_bias == "neutral"

    def test_directional_bias_neutral_for_most_active(self):
        snap = self._snapshot([_row(symbol="SPY", list_type="most_active", rank=1)])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["SPY"].directional_bias == "neutral"

    def test_mover_prominence_score_decreases_with_rank(self):
        snap1 = self._snapshot([_row(symbol="A", list_type="top_gainers", rank=1)])
        snap2 = self._snapshot([_row(symbol="A", list_type="top_gainers", rank=25)])
        ctx1 = merge_duplicate_symbols(snap1)
        ctx2 = merge_duplicate_symbols(snap2)
        assert ctx1["A"].mover_prominence_score > ctx2["A"].mover_prominence_score

    def test_multi_list_prominence_exceeds_single_list(self):
        snap_single = self._snapshot([_row(symbol="A", list_type="top_gainers", rank=5)])
        snap_multi  = self._snapshot([
            _row(symbol="A", list_type="top_gainers", rank=5),
            _row(symbol="A", list_type="gap_up", rank=5),
        ])
        c1 = merge_duplicate_symbols(snap_single)["A"]
        c2 = merge_duplicate_symbols(snap_multi)["A"]
        assert c2.mover_prominence_score > c1.mover_prominence_score

    def test_empty_snapshot_returns_empty_dict(self):
        snap = MarketMoverSnapshot(rows=[])
        assert merge_duplicate_symbols(snap) == {}

    def test_sources_deduplicated(self):
        snap = self._snapshot([
            _row(symbol="A", list_type="top_gainers", rank=1, source="webull"),
            _row(symbol="A", list_type="most_active", rank=3, source="webull"),
        ])
        ctx = merge_duplicate_symbols(snap)
        assert ctx["A"].sources == ["webull"]   # deduplicated


# ── symbol_list_memberships ───────────────────────────────────────────────────

class TestSymbolListMemberships:
    def _contexts(self):
        snap = MarketMoverSnapshot(rows=normalize_mover_rows([
            _row(symbol="AAPL", list_type="top_gainers", rank=1),
        ]))
        return merge_duplicate_symbols(snap)

    def test_present_symbol_returns_context(self):
        ctx = self._contexts()
        result = symbol_list_memberships("AAPL", ctx)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_case_insensitive_lookup(self):
        ctx = self._contexts()
        result = symbol_list_memberships("aapl", ctx)
        assert result is not None

    def test_absent_symbol_returns_none(self):
        ctx = self._contexts()
        assert symbol_list_memberships("MSFT", ctx) is None


# ── _resolve_bias ─────────────────────────────────────────────────────────────

class TestResolveBias:
    def test_gainers_only_is_long(self):
        assert _resolve_bias(["top_gainers"]) == "long"

    def test_losers_only_is_short(self):
        assert _resolve_bias(["top_losers"]) == "short"

    def test_gap_up_is_long(self):
        assert _resolve_bias(["gap_up"]) == "long"

    def test_gap_down_is_short(self):
        assert _resolve_bias(["gap_down"]) == "short"

    def test_most_active_neutral(self):
        assert _resolve_bias(["most_active"]) == "neutral"

    def test_unusual_volume_neutral(self):
        assert _resolve_bias(["unusual_volume"]) == "neutral"

    def test_long_and_short_is_neutral(self):
        assert _resolve_bias(["top_gainers", "top_losers"]) == "neutral"

    def test_gap_up_and_most_active_is_long(self):
        assert _resolve_bias(["gap_up", "most_active"]) == "long"

    def test_empty_list_is_neutral(self):
        assert _resolve_bias([]) == "neutral"


# ── _compute_prominence ───────────────────────────────────────────────────────

class TestComputeProminence:
    def test_rank1_single_list_near_max(self):
        score = _compute_prominence({"top_gainers": 1}, total_rows_in_snapshot=10)
        assert score == pytest.approx(1.0)   # 1.0 × 1.0 bonus = 1.0, capped

    def test_rank51_returns_zero(self):
        score = _compute_prominence({"top_gainers": 51}, total_rows_in_snapshot=100)
        assert score == pytest.approx(0.0)

    def test_rank25_is_half(self):
        score = _compute_prominence({"top_gainers": 25}, total_rows_in_snapshot=50)
        # rank_factor = 1 - 24/50 = 0.52; bonus = 1.0 (single list); result = 0.52
        assert 0.45 < score < 0.60

    def test_two_lists_get_bonus(self):
        score_one = _compute_prominence({"top_gainers": 5}, 20)
        score_two = _compute_prominence({"top_gainers": 5, "gap_up": 5}, 40)
        assert score_two > score_one   # multi-list bonus applied

    def test_empty_ranks_returns_zero(self):
        assert _compute_prominence({}, 100) == pytest.approx(0.0)

    def test_capped_at_one(self):
        # Three lists all at rank 1 → should not exceed 1.0
        score = _compute_prominence(
            {"top_gainers": 1, "gap_up": 1, "unusual_volume": 1},
            total_rows_in_snapshot=30
        )
        assert score <= 1.0


# ── LIST_TYPE constants sanity ────────────────────────────────────────────────

class TestConstants:
    def test_all_list_types_have_direction(self):
        for lt in ("top_gainers", "top_losers", "most_active", "unusual_volume", "gap_up", "gap_down"):
            assert lt in LIST_TYPE_DIRECTION

    def test_all_list_types_have_strategy_hints_entry(self):
        for lt in ("top_gainers", "top_losers", "most_active", "unusual_volume", "gap_up", "gap_down"):
            assert lt in LIST_TYPE_STRATEGY_HINTS

    def test_strategy_hints_are_lists(self):
        for lt, hints in LIST_TYPE_STRATEGY_HINTS.items():
            assert isinstance(hints, list), f"{lt} hints should be a list"
