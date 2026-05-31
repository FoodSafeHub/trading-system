"""
market_movers.py — Normalized market-mover layer for the day-trading engine.

Ingests raw mover rows from any source (Webull, broker API, manual feed, etc.)
and produces a clean, deduplicated snapshot that the candidate ranker can consume.

Design constraints
------------------
* This module is PURE DATA — it never triggers entries. The mover snapshot
  only extends the universe that the existing scanner/strategy engine considers.
* Symbols appearing on mover lists get a scoring boost in candidate_ranker.py,
  but still require a valid DayTradeSignal from a strategy before any trade fires.
* All timestamps are stored as UTC-aware datetime objects or ISO-8601 strings.
* No network calls here — all data comes in from the caller.

Public API
----------
    from app.services.strategy.daytrading.market_movers import (
        MarketMoverRow, MarketMoverSnapshot, SymbolMoverContext,
        normalize_mover_rows, merge_duplicate_symbols, symbol_list_memberships,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

ListType = Literal[
    "most_active",
    "top_gainers",
    "top_losers",
    "unusual_volume",
    "gap_up",
    "gap_down",
]

# All recognised list-type strings — used for validation in normalize_mover_rows
_VALID_LIST_TYPES: frozenset[str] = frozenset(
    {"most_active", "top_gainers", "top_losers", "unusual_volume", "gap_up", "gap_down"}
)

# Strategy buckets each list type naturally points toward
LIST_TYPE_STRATEGY_HINTS: dict[str, list[str]] = {
    "top_gainers":    ["ORBBreakout", "VWAPMeanReversion", "EMAMomentum"],
    "top_losers":     ["OpeningGapFade", "VWAPMeanReversion", "EMAMomentum"],
    "most_active":    [],                          # neutral — still requires strategy signal
    "gap_up":         ["ORBBreakout", "OpeningGapFade", "NRSqueezeBreakout"],
    "gap_down":       ["OpeningGapFade", "VWAPMeanReversion"],
    "unusual_volume": ["NRSqueezeBreakout", "ORBBreakout"],
}

# Directional bias implied by each list type
# "long", "short", or "neutral" — influences candidate_ranker direction scoring
LIST_TYPE_DIRECTION: dict[str, str] = {
    "top_gainers":    "long",
    "top_losers":     "short",
    "gap_up":         "long",
    "gap_down":       "short",
    "most_active":    "neutral",
    "unusual_volume": "neutral",
}


# ── Core dataclasses ──────────────────────────────────────────────────────────

@dataclass
class MarketMoverRow:
    """
    One entry from one mover list, as delivered by the data source.

    All monetary fields are in the symbol's native currency (USD for US equities).
    Fields that the data source does not provide should be left at their
    default values (0.0 / None / "") — the ranker treats them as unavailable.
    """
    symbol: str
    list_type: ListType             # which mover list this row came from
    rank: int                       # 1-based rank within the list (1 = top)
    price: float                    # last trade price at time of snapshot
    pct_change: float               # % change from prior close (signed, e.g. +3.5 or -2.1)
    volume: float                   # cumulative volume at snapshot time (shares)

    # ── Optional enrichment ───────────────────────────────────────────────────
    dollar_volume: float = 0.0      # price × volume; computed if not provided
    avg_volume: float = 0.0         # 30-day average daily volume (0 = unknown)
    rvol: float = 0.0               # relative volume = volume / avg_volume (0 = unknown)
    market_cap: float = 0.0         # market cap in dollars (0 = unknown)
    float_shares: float = 0.0       # shares float (0 = unknown)
    spread_pct: float = 0.0         # bid-ask spread as % of price (0 = unknown)

    # ── Catalyst / news flags ─────────────────────────────────────────────────
    has_news: bool = False           # any news flag from the data source
    news_tags: list[str] = field(default_factory=list)  # e.g. ["earnings", "upgrade"]

    # ── Source provenance ─────────────────────────────────────────────────────
    source: str = ""                 # e.g. "webull", "manual", "alpaca"
    timestamp: str = ""              # ISO-8601 UTC string of when this row was fetched

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        if self.dollar_volume == 0.0 and self.price > 0 and self.volume > 0:
            self.dollar_volume = self.price * self.volume
        if self.rvol == 0.0 and self.avg_volume > 0 and self.volume > 0:
            self.rvol = self.volume / self.avg_volume

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":       self.symbol,
            "list_type":    self.list_type,
            "rank":         self.rank,
            "price":        round(self.price, 4),
            "pct_change":   round(self.pct_change, 3),
            "volume":       int(self.volume),
            "dollar_volume": round(self.dollar_volume, 2),
            "avg_volume":   int(self.avg_volume),
            "rvol":         round(self.rvol, 3),
            "market_cap":   round(self.market_cap, 0),
            "float_shares": int(self.float_shares),
            "spread_pct":   round(self.spread_pct, 4),
            "has_news":     self.has_news,
            "news_tags":    list(self.news_tags),
            "source":       self.source,
            "timestamp":    self.timestamp,
        }


@dataclass
class MarketMoverSnapshot:
    """
    The full set of mover rows from one point in time.

    Create once per scan cycle, then pass to merge_duplicate_symbols() and
    symbol_list_memberships() to get per-symbol context objects.
    """
    rows: list[MarketMoverRow] = field(default_factory=list)
    fetched_at: str = ""            # ISO-8601 UTC of snapshot creation
    source: str = ""                # primary source label

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()

    # ── Convenience accessors ─────────────────────────────────────────────────

    def rows_for(self, list_type: str) -> list[MarketMoverRow]:
        """Return all rows matching a specific list_type, in rank order."""
        return sorted(
            [r for r in self.rows if r.list_type == list_type],
            key=lambda r: r.rank,
        )

    def symbols(self) -> list[str]:
        """Deduplicated symbol list, ordered by first appearance."""
        seen: dict[str, None] = {}
        for r in self.rows:
            seen[r.symbol] = None
        return list(seen.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at,
            "source":     self.source,
            "row_count":  len(self.rows),
            "rows":       [r.to_dict() for r in self.rows],
        }


@dataclass
class SymbolMoverContext:
    """
    Consolidated view of one symbol across ALL mover lists in a snapshot.

    Produced by merge_duplicate_symbols() — one object per unique symbol.
    This is what candidate_ranker.py consumes.
    """
    symbol: str

    # ── List memberships ──────────────────────────────────────────────────────
    list_memberships: list[str] = field(default_factory=list)   # e.g. ["top_gainers","gap_up"]
    best_rank_per_list: dict[str, int] = field(default_factory=dict)  # {list_type: rank}

    # ── Best values across all appearances ───────────────────────────────────
    price: float = 0.0
    pct_change: float = 0.0         # from the highest-priority list (gainers/losers/gap first)
    volume: float = 0.0             # max volume seen
    dollar_volume: float = 0.0
    rvol: float = 0.0               # max RVOL seen
    spread_pct: float = 0.0         # max spread (worst case) — used for hard filter

    # ── Catalyst ──────────────────────────────────────────────────────────────
    has_news: bool = False
    news_tags: list[str] = field(default_factory=list)

    # ── Strategy hints implied by list membership ─────────────────────────────
    strategy_hints: list[str] = field(default_factory=list)
    directional_bias: str = "neutral"  # "long" | "short" | "neutral"

    # ── Mover score (0–1): how prominently this symbol appears in the movers ──
    # Populated by symbol_list_memberships(); used as input to the ranker.
    mover_prominence_score: float = 0.0

    # ── Provenance ────────────────────────────────────────────────────────────
    sources: list[str] = field(default_factory=list)
    snapshot_time: str = ""

    @property
    def is_mover(self) -> bool:
        return len(self.list_memberships) > 0

    @property
    def list_count(self) -> int:
        return len(self.list_memberships)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":                self.symbol,
            "list_memberships":      list(self.list_memberships),
            "best_rank_per_list":    dict(self.best_rank_per_list),
            "price":                 round(self.price, 4),
            "pct_change":            round(self.pct_change, 3),
            "volume":                int(self.volume),
            "dollar_volume":         round(self.dollar_volume, 2),
            "rvol":                  round(self.rvol, 3),
            "spread_pct":            round(self.spread_pct, 4),
            "has_news":              self.has_news,
            "news_tags":             list(self.news_tags),
            "strategy_hints":        list(self.strategy_hints),
            "directional_bias":      self.directional_bias,
            "mover_prominence_score": round(self.mover_prominence_score, 4),
            "sources":               list(self.sources),
            "snapshot_time":         self.snapshot_time,
        }


# ── Public functions ──────────────────────────────────────────────────────────

def normalize_mover_rows(
    raw_rows: list[dict[str, Any]],
    *,
    source: str = "",
    timestamp: str = "",
    drop_unknown_list_types: bool = True,
) -> list[MarketMoverRow]:
    """
    Convert a list of raw dicts (from any data source) into MarketMoverRow objects.

    Each dict should contain at minimum:
        symbol, list_type, rank, price, pct_change, volume

    Unknown keys are silently ignored; missing optional keys use defaults.
    Rows with unknown list_type values are dropped if drop_unknown_list_types=True,
    or kept with list_type cast to "most_active" if False.

    Parameters
    ----------
    raw_rows
        List of dicts from the mover data source.
    source
        Label for the data origin (e.g. "webull", "alpaca").
    timestamp
        ISO-8601 UTC string; defaults to now() if empty.
    drop_unknown_list_types
        Drop rows whose list_type is not in _VALID_LIST_TYPES.

    Returns
    -------
    list[MarketMoverRow]
        Cleaned, validated rows. May be shorter than raw_rows if rows were dropped.
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    out: list[MarketMoverRow] = []
    dropped = 0

    for i, raw in enumerate(raw_rows):
        try:
            sym = str(raw.get("symbol", "")).upper().strip()
            if not sym:
                dropped += 1
                continue

            raw_lt = str(raw.get("list_type", "")).lower().strip()
            if raw_lt not in _VALID_LIST_TYPES:
                if drop_unknown_list_types:
                    logger.debug("normalize_mover_rows: dropping row %d — unknown list_type %r", i, raw_lt)
                    dropped += 1
                    continue
                raw_lt = "most_active"

            row = MarketMoverRow(
                symbol=sym,
                list_type=raw_lt,                                       # type: ignore[arg-type]
                rank=int(raw.get("rank", i + 1)),
                price=float(raw.get("price", 0.0)),
                pct_change=float(raw.get("pct_change", 0.0)),
                volume=float(raw.get("volume", 0.0)),
                dollar_volume=float(raw.get("dollar_volume", 0.0)),
                avg_volume=float(raw.get("avg_volume", 0.0)),
                rvol=float(raw.get("rvol", 0.0)),
                market_cap=float(raw.get("market_cap", 0.0)),
                float_shares=float(raw.get("float_shares", 0.0)),
                spread_pct=float(raw.get("spread_pct", 0.0)),
                has_news=bool(raw.get("has_news", False)),
                news_tags=list(raw.get("news_tags", [])),
                source=raw.get("source", source),
                timestamp=raw.get("timestamp", timestamp),
            )
            out.append(row)
        except Exception as exc:
            logger.warning("normalize_mover_rows: skipping malformed row %d: %s", i, exc)
            dropped += 1

    if dropped:
        logger.info("normalize_mover_rows: %d rows dropped out of %d", dropped, len(raw_rows))

    return out


def merge_duplicate_symbols(
    snapshot: MarketMoverSnapshot,
) -> dict[str, SymbolMoverContext]:
    """
    Collapse all rows in the snapshot into one SymbolMoverContext per symbol.

    Rules for merging:
    - list_memberships: union of all list_types this symbol appeared on.
    - best_rank_per_list: lowest (best) rank seen for each list_type.
    - price: from the highest-priority list (gainers > losers > gap > active > unusual).
    - pct_change: from the highest-priority list for the symbol.
    - volume / dollar_volume / rvol: maximum across all rows.
    - spread_pct: maximum (worst-case, used for hard filters).
    - has_news / news_tags: union.
    - strategy_hints: union from LIST_TYPE_STRATEGY_HINTS.
    - directional_bias: resolved from list memberships (see _resolve_bias).
    - mover_prominence_score: computed from rank + list count (see _prominence).

    Returns
    -------
    dict[str, SymbolMoverContext]
        Keyed by symbol (uppercase).
    """
    # List priority for resolving price / pct_change when a symbol appears on multiple lists
    _PRIORITY: dict[str, int] = {
        "top_gainers":    1,
        "top_losers":     2,
        "gap_up":         3,
        "gap_down":       4,
        "unusual_volume": 5,
        "most_active":    6,
    }

    # Group rows by symbol
    by_symbol: dict[str, list[MarketMoverRow]] = {}
    for row in snapshot.rows:
        by_symbol.setdefault(row.symbol, []).append(row)

    result: dict[str, SymbolMoverContext] = {}

    for sym, rows in by_symbol.items():
        memberships: list[str] = []
        ranks: dict[str, int] = {}
        all_sources: list[str] = []
        all_news_tags: list[str] = []
        has_news = False
        max_volume = 0.0
        max_dollar_vol = 0.0
        max_rvol = 0.0
        max_spread = 0.0

        # Sort rows by priority so we pick price/pct_change from the best source
        rows_sorted = sorted(rows, key=lambda r: _PRIORITY.get(r.list_type, 99))

        for row in rows_sorted:
            lt = row.list_type
            if lt not in memberships:
                memberships.append(lt)
            # Best rank per list (lower = better)
            if lt not in ranks or row.rank < ranks[lt]:
                ranks[lt] = row.rank
            if row.source and row.source not in all_sources:
                all_sources.append(row.source)
            if row.has_news:
                has_news = True
            for tag in row.news_tags:
                if tag not in all_news_tags:
                    all_news_tags.append(tag)
            max_volume     = max(max_volume, row.volume)
            max_dollar_vol = max(max_dollar_vol, row.dollar_volume)
            max_rvol       = max(max_rvol, row.rvol)
            max_spread     = max(max_spread, row.spread_pct)

        # Price + pct_change from the highest-priority row
        primary = rows_sorted[0]

        # Collect strategy hints (union, deduped, ordered)
        hints_seen: dict[str, None] = {}
        for lt in memberships:
            for h in LIST_TYPE_STRATEGY_HINTS.get(lt, []):
                hints_seen[h] = None

        bias = _resolve_bias(memberships)
        prominence = _compute_prominence(ranks, len(snapshot.rows))

        result[sym] = SymbolMoverContext(
            symbol=sym,
            list_memberships=memberships,
            best_rank_per_list=ranks,
            price=primary.price,
            pct_change=primary.pct_change,
            volume=max_volume,
            dollar_volume=max_dollar_vol,
            rvol=max_rvol,
            spread_pct=max_spread,
            has_news=has_news,
            news_tags=all_news_tags,
            strategy_hints=list(hints_seen.keys()),
            directional_bias=bias,
            mover_prominence_score=round(prominence, 4),
            sources=all_sources,
            snapshot_time=snapshot.fetched_at,
        )

    return result


def symbol_list_memberships(
    symbol: str,
    contexts: dict[str, SymbolMoverContext],
) -> SymbolMoverContext | None:
    """
    Return the SymbolMoverContext for a single symbol, or None if it is
    not present in any mover list.

    Convenience wrapper around the dict returned by merge_duplicate_symbols().
    """
    return contexts.get(symbol.upper().strip())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_bias(memberships: list[str]) -> str:
    """
    Compute directional bias from the union of list memberships.

    Rules (applied in priority order):
    1. If BOTH long-biased and short-biased lists are present → "neutral"
       (e.g., symbol is on both top_gainers and top_losers due to intraday reversal).
    2. If only long-biased lists present → "long".
    3. If only short-biased lists present → "short".
    4. Otherwise → "neutral".
    """
    biases = {LIST_TYPE_DIRECTION.get(lt, "neutral") for lt in memberships}
    if "long" in biases and "short" in biases:
        return "neutral"
    if "long" in biases:
        return "long"
    if "short" in biases:
        return "short"
    return "neutral"


def _compute_prominence(
    best_rank_per_list: dict[str, int],
    total_rows_in_snapshot: int,
) -> float:
    """
    Mover prominence score in [0, 1].

    Formula:
        prominence = mean(list_scores) × list_count_bonus

    where:
        list_score for each list = max(0, 1 - (rank - 1) / 50)
            → rank 1  = 1.0
            → rank 25 = 0.50
            → rank 50 = 0.0
            → rank 51+ = 0.0

        list_count_bonus:
            1 list  = 1.0
            2 lists = 1.15
            3+ lists = 1.30

    Capped at 1.0.
    """
    if not best_rank_per_list:
        return 0.0

    list_scores = [max(0.0, 1.0 - (rank - 1) / 50.0) for rank in best_rank_per_list.values()]
    mean_score = sum(list_scores) / len(list_scores)

    n = len(best_rank_per_list)
    bonus = 1.0 if n == 1 else (1.15 if n == 2 else 1.30)

    return min(1.0, mean_score * bonus)
