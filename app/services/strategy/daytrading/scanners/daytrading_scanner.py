"""
DayTradingScanner — pre-market routine that finds today's best intraday candidates.

Mimics what a human discretionary trader does each morning:
  1. Load a universe of liquid names.
  2. Pull overnight / pre-market data.
  3. Hard-filter the trash (thin, dead, wildly volatile).
  4. Score survivors on liquidity, volatility, pre-market activity, and gap size.
  5. Tag each name with the strategy bucket it fits (ORB, VWAP, gap, momentum).
  6. Hand the ranked watchlist to the brain for final regime-aware adjustment.

Data sources (in priority order):
  1. BarCache (existing in-process cache, if warm) — zero-latency.
  2. yfinance daily bars for ATR / 30-day avg volume.
  3. yfinance 1d with pre-market session for gap/pre-market volume.
     NOTE: yfinance pre-market data is sparse for most symbols; when
     unavailable the scanner approximates gap_pct from (open - prior_close)
     and marks premarket_volume as the first 30-min intraday proxy.

All thresholds live in DayTradingScannerConfig so they can be overridden
without touching this file.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from app.services.strategy.daytrading.market_open import ET, now_et
from app.services.strategy.daytrading.scanners.universe import (
    load_float_cache,
    load_universe,
    load_india_universe,
    save_float_cache,
)

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DayTradingScannerConfig:
    """All hard-filter and scoring thresholds in one place."""

    # ── Hard filters ──────────────────────────────────────────────────────────
    min_avg_volume: float = 1_000_000   # shares/day — drop thinly traded names
    min_price: float = 5.0              # skip penny stocks
    max_price: float | None = None      # None = no cap
    min_float: float | None = None      # min shares float (None = no minimum)
    max_float: float | None = None      # max shares float (None = no cap) — set this
                                        # low to find low-float runners; high to focus
                                        # on liquid large-caps
    min_atr_pct: float = 1.0            # too dead to trade intraday
    max_atr_pct: float = 8.0            # too wild (blow-up risk)

    # ── Market selection ──────────────────────────────────────────────────────
    # "us"     → NASDAQ/NYSE universe, yfinance + TwelveData for data
    # "india"  → Nifty 200 universe, Upstox for all data
    # "both"   → runs US then India and merges results
    market: str = "us"

    # ── Scan execution ────────────────────────────────────────────────────────
    max_workers: int = 32               # parallel workers for Phase 2 deep fetch
    chunk_size: int = 200               # symbols per Phase 2 chunk
    universe_max_symbols: int | None = None  # cap universe size for development; None = all

    # ── Pre-market activity ───────────────────────────────────────────────────
    # rel_vol = premarket_volume / avg_daily_volume_30d
    min_premarket_rel_vol: float = 0.01   # 1% — very low bar, just exclude zeros

    # ── Gap thresholds (abs(gap_pct)) ─────────────────────────────────────────
    gap_flat_pct: float = 0.5       # below this → "flat" (no gap)
    gap_small_pct: float = 1.5      # 0.5–1.5% → small gap
    gap_medium_pct: float = 3.5     # 1.5–3.5% → medium gap
    gap_large_pct: float = 7.0      # 3.5–7%  → large gap; >7% → extreme

    # ── Scoring weights (must sum to 1.0) ─────────────────────────────────────
    weight_volume: float = 0.35
    weight_volatility: float = 0.30
    weight_rel_vol: float = 0.25
    weight_gap: float = 0.10

    # ── Catalyst boost ────────────────────────────────────────────────────────
    catalyst_boost: float = 0.08      # added to score when catalyst is detected

    # ── Ideal ATR% range for scoring ──────────────────────────────────────────
    ideal_atr_low: float = 1.5
    ideal_atr_high: float = 5.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SymbolScanMetrics:
    """Raw measurements collected for each symbol before scoring."""
    symbol: str
    last_price: float
    avg_daily_volume_30d: float
    atr_14: float
    atr_pct: float                          # atr_14 / last_price * 100
    premarket_volume: float                 # 0.0 if unavailable
    premarket_rel_vol: float                # premarket_volume / avg_daily_volume_30d
    premarket_gap_pct: float                # (open - prior_close) / prior_close * 100
    gap_direction: Literal["up", "down", "flat"]
    gap_size: Literal["none", "small", "medium", "large", "extreme"]
    has_catalyst: bool
    catalyst_tags: list[str]                # ["earnings", "news", ...]
    # How the catalyst was inferred. "heuristic" = volume-anomaly / best-effort
    # yfinance earnings calendar (NOT a confirmed news event). "none" when no
    # catalyst was flagged. Downstream code/UI must not treat "heuristic" as a
    # confirmed catalyst.
    catalyst_source: str = "none"           # "none" | "heuristic"
    # Raw snapshot for debugging
    prior_close: float = 0.0
    today_open: float = 0.0
    data_quality: str = "ok"               # "ok" | "partial" | "no_data"

    # Shares float — populated lazily from the daily float cache. 0.0 means
    # the float is unknown (yfinance returned nothing for this symbol today).
    shares_float: float = 0.0

    # Spread stub — replace when L1 data is available
    spread_pct: float | None = None         # None = unavailable

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": round(self.last_price, 2),
            "avg_daily_volume_30d": int(self.avg_daily_volume_30d),
            "atr_14": round(self.atr_14, 4),
            "atr_pct": round(self.atr_pct, 2),
            "premarket_volume": int(self.premarket_volume),
            "premarket_rel_vol": round(self.premarket_rel_vol, 4),
            "premarket_gap_pct": round(self.premarket_gap_pct, 3),
            "gap_direction": self.gap_direction,
            "gap_size": self.gap_size,
            "has_catalyst": self.has_catalyst,
            "catalyst_tags": self.catalyst_tags,
            "catalyst_source": self.catalyst_source,
            "prior_close": round(self.prior_close, 4),
            "today_open": round(self.today_open, 4),
            "data_quality": self.data_quality,
            "shares_float": int(self.shares_float),
        }


@dataclass
class SymbolScanResult:
    """Scored, tagged output for one symbol."""
    symbol: str
    score: float                            # 0.0–1.0 (higher = better candidate)
    adjusted_score: float                   # after regime / market-state adjustment
    tags: list[str]                         # human-readable strategy hints
    recommended_strategy_bucket: str        # "ORB" | "VWAP" | "gap" | "momentum"
    metrics: SymbolScanMetrics
    rejection_reason: str = ""              # empty string = passed all filters

    # ── Native-signal pre-check fields (populated only for top-K) ─────────────
    # Empty / None when the pre-check did not run on this symbol (deep in the
    # tail) or when no audited strategy accepted at the current bar.
    native_signal_active: bool = False
    active_native_strategies: list[str] = field(default_factory=list)
    best_native_strategy: str | None = None
    best_native_side: str | None = None            # "BUY" | "SELL" | None
    best_native_confidence: float | None = None
    native_precheck_ran: bool = False              # True if we executed the pre-check

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "tags": self.tags,
            "recommended_strategy_bucket": self.recommended_strategy_bucket,
            "metrics": self.metrics.to_dict(),
            "rejection_reason": self.rejection_reason,
            "native_signal_active": self.native_signal_active,
            "active_native_strategies": list(self.active_native_strategies),
            "best_native_strategy": self.best_native_strategy,
            "best_native_side": self.best_native_side,
            "best_native_confidence": (
                round(self.best_native_confidence, 3)
                if self.best_native_confidence is not None else None
            ),
            "native_precheck_ran": self.native_precheck_ran,
        }


# ── Scanner ───────────────────────────────────────────────────────────────────

class DayTradingScanner:
    """
    Pre-market scanner that ranks today's intraday candidates.

    Typical usage:
        scanner = DayTradingScanner(config=DayTradingScannerConfig())
        watchlist = scanner.get_intraday_watchlist(max_symbols=20)

    Brain-aware usage:
        scanner = DayTradingScanner(config=..., brain=DayTradingBrain())
        watchlist = scanner.get_intraday_watchlist(max_symbols=20)
        # Brain adjusts scores based on current market state (TREND_UP etc.)
    """

    def __init__(
        self,
        config: DayTradingScannerConfig | None = None,
        universe: list[str] | None = None,
        brain=None,          # DayTradingBrain — optional, avoids circular import
        cache=None,          # BarCache — reuse if already warm
    ):
        self.config = config or DayTradingScannerConfig()
        if universe is None:
            market = self.config.market.lower()
            if market == "india":
                uni = load_india_universe()
            elif market == "both":
                uni = load_universe() + load_india_universe()
            else:
                uni = load_universe()
            if self.config.universe_max_symbols:
                uni = uni[: self.config.universe_max_symbols]
            self._universe = uni
        else:
            self._universe = [s.upper() for s in universe]
        self._brain = brain
        self._cache = cache
        self._float_cache: dict[str, float] = load_float_cache()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_universe(self) -> list[str]:
        """Return the symbol universe this scanner will evaluate."""
        return list(self._universe)

    @classmethod
    def invalidate_bulk_cache(cls) -> None:
        """Force the next scan to re-download all data.
        Call this when the user changes filter presets mid-session."""
        cls._daily_cache            = {}
        cls._daily_cache_date       = None
        cls._india_daily_cache      = {}
        cls._india_daily_cache_date = None
        cls._pm_vol_cache           = {}
        cls._pm_vol_cache_date      = None
        logger.info("Scanner caches invalidated — next scan will re-download everything")

    def fetch_metrics(self, symbol: str) -> SymbolScanMetrics:
        """
        Pull daily + pre-market data for one symbol and return raw metrics.

        Data flow:
          1. Try BarCache for 1d bars (if cache was passed in and is warm).
          2. Fall back to yfinance 60-day daily download.
          3. Attempt pre-market gap from yfinance premarket=True (often sparse).
          4. Approximate pre-market volume from first-30-min intraday if needed.
        """
        try:
            return self._fetch_metrics_impl(symbol)
        except Exception as e:
            logger.warning("fetch_metrics failed for %s: %s", symbol, e)
            return SymbolScanMetrics(
                symbol=symbol,
                last_price=0.0,
                avg_daily_volume_30d=0.0,
                atr_14=0.0,
                atr_pct=0.0,
                premarket_volume=0.0,
                premarket_rel_vol=0.0,
                premarket_gap_pct=0.0,
                gap_direction="flat",
                gap_size="none",
                has_catalyst=False,
                catalyst_tags=[],
                data_quality="no_data",
            )

    def score_symbol(self, metrics: SymbolScanMetrics) -> SymbolScanResult:
        """
        Apply hard filters, then score and tag a symbol.

        Hard filters return a SymbolScanResult with score=0 and rejection_reason set.
        Survivors get a numeric score and strategy tags.
        """
        cfg = self.config

        # ── Hard filter: data quality ─────────────────────────────────────────
        if metrics.data_quality == "no_data" or metrics.last_price <= 0:
            return self._reject(metrics, "no_data")

        # ── Hard filter: price ────────────────────────────────────────────────
        if metrics.last_price < cfg.min_price:
            return self._reject(metrics, f"price {metrics.last_price:.2f} < min {cfg.min_price}")
        if cfg.max_price and metrics.last_price > cfg.max_price:
            return self._reject(metrics, f"price {metrics.last_price:.2f} > max {cfg.max_price}")

        # ── Hard filter: liquidity ────────────────────────────────────────────
        if metrics.avg_daily_volume_30d < cfg.min_avg_volume:
            return self._reject(
                metrics,
                f"avg_vol {metrics.avg_daily_volume_30d/1e6:.1f}M < min {cfg.min_avg_volume/1e6:.1f}M",
            )

        # ── Hard filter: float ────────────────────────────────────────────────
        # shares_float == 0 means yfinance returned nothing (common for ETFs
        # and thinly-covered small-caps). Skip the float check rather than
        # reject — the liquidity and ATR filters already guard against junk.
        # Only apply bounds when float data is actually available (> 0).
        if (cfg.min_float or cfg.max_float) and metrics.shares_float > 0:
            if cfg.min_float and metrics.shares_float < cfg.min_float:
                return self._reject(
                    metrics,
                    f"float {metrics.shares_float/1e6:.1f}M < min {cfg.min_float/1e6:.1f}M",
                )
            if cfg.max_float and metrics.shares_float > cfg.max_float:
                return self._reject(
                    metrics,
                    f"float {metrics.shares_float/1e6:.1f}M > max {cfg.max_float/1e6:.1f}M",
                )

        # ── Hard filter: ATR ──────────────────────────────────────────────────
        if metrics.atr_pct < cfg.min_atr_pct:
            return self._reject(metrics, f"atr_pct {metrics.atr_pct:.2f}% too low (dead)")
        if metrics.atr_pct > cfg.max_atr_pct:
            return self._reject(metrics, f"atr_pct {metrics.atr_pct:.2f}% too wild")

        # ── Score: liquidity ──────────────────────────────────────────────────
        # log10 scale so mega-cap volume doesn't dominate; capped at 1.0
        score_vol = min(math.log10(metrics.avg_daily_volume_30d / 1e6 + 1) / math.log10(101), 1.0)

        # ── Score: volatility (Goldilocks zone) ───────────────────────────────
        lo = cfg.ideal_atr_low
        hi = cfg.ideal_atr_high
        a = metrics.atr_pct
        if a < cfg.min_atr_pct:
            score_vol_atr = 0.0
        elif a <= lo:
            score_vol_atr = (a - cfg.min_atr_pct) / max(lo - cfg.min_atr_pct, 0.01)
        elif a <= hi:
            score_vol_atr = 1.0
        elif a <= cfg.max_atr_pct:
            score_vol_atr = 1.0 - (a - hi) / max(cfg.max_atr_pct - hi, 0.01)
        else:
            score_vol_atr = 0.0
        score_vol_atr = max(0.0, min(1.0, score_vol_atr))

        # ── Score: pre-market relative volume ─────────────────────────────────
        rv = metrics.premarket_rel_vol
        if rv < 0.02:
            score_rv = rv / 0.02 * 0.3         # 0–2% → up to 0.3
        elif rv < 0.10:
            score_rv = 0.3 + (rv - 0.02) / 0.08 * 0.5    # 2–10% → 0.3–0.8
        else:
            score_rv = min(0.8 + (rv - 0.10) / 0.10 * 0.2, 1.0)  # 10%+ → 0.8–1.0

        # ── Score: gap ────────────────────────────────────────────────────────
        abs_gap = abs(metrics.premarket_gap_pct)
        gs = metrics.gap_size
        _gap_scores = {"none": 0.0, "small": 0.25, "medium": 0.65, "large": 0.85, "extreme": 0.5}
        score_gap = _gap_scores.get(gs, 0.0)

        # ── Combined score ────────────────────────────────────────────────────
        score = (
            cfg.weight_volume     * score_vol
            + cfg.weight_volatility * score_vol_atr
            + cfg.weight_rel_vol  * score_rv
            + cfg.weight_gap      * score_gap
        )

        # ── Catalyst boost ────────────────────────────────────────────────────
        if metrics.has_catalyst:
            score = min(score + cfg.catalyst_boost, 1.0)

        # ── Tags ──────────────────────────────────────────────────────────────
        tags: list[str] = []

        # Gap tags
        if gs == "none":
            tags.append("VWAP")                 # no gap → VWAP reversion candidate
        elif gs in ("small", "medium"):
            if rv >= 0.05:
                tags.append("gap_and_go")
            else:
                tags.append("gap_fade_candidate")
        elif gs in ("large", "extreme"):
            if rv >= 0.10:
                tags.append("gap_and_go")
            tags.append("gap_fade_candidate")   # large gap can fade hard too

        # ORB tag: wants decent ATR and some pre-market activity
        if metrics.atr_pct >= 1.5 and rv >= 0.03:
            tags.append("ORB")

        # Momentum tag: strong gap + strong rel vol
        if abs_gap >= cfg.gap_small_pct and rv >= 0.08:
            tags.append("momentum")

        # Pre-market heat tag
        if rv >= 0.10:
            tags.append("hot_rel_vol")

        # Catalyst tags
        if metrics.has_catalyst:
            tags.extend(["catalyst", "higher_risk"])

        # Low-float proxy: very high rel vol on a normal-volatility name → possible low float
        if rv >= 0.20 and metrics.atr_pct < 4.0:
            tags.append("low_float_proxy")

        # De-duplicate while preserving order
        seen: set[str] = set()
        tags = [t for t in tags if not (t in seen or seen.add(t))]

        # ── Recommended strategy bucket ───────────────────────────────────────
        bucket = _pick_bucket(tags, metrics, self.config)

        return SymbolScanResult(
            symbol=metrics.symbol,
            score=round(score, 4),
            adjusted_score=round(score, 4),
            tags=tags,
            recommended_strategy_bucket=bucket,
            metrics=metrics,
        )

    # ── Daily OHLCV cache (US) ── symbol → DataFrame, refreshed daily ───────
    _daily_cache: dict[str, pd.DataFrame] = {}
    _daily_cache_date: date | None = None

    # ── Daily OHLCV cache (India / Upstox) ───────────────────────────────────
    _india_daily_cache: dict[str, pd.DataFrame] = {}
    _india_daily_cache_date: date | None = None

    # ── Pre-market / opening-activity volume cache ────────────────────────────
    _pm_vol_cache: dict[str, float] = {}
    _pm_vol_cache_date: date | None = None

    def scan(self, max_symbols: int = 50) -> list[SymbolScanResult]:
        """
        Three-phase scan for 7,000+ symbol universes.

        Phase 1 — bulk OHLCV download + pre-filter (full universe, batched):
          Downloads 35d daily OHLCV for every symbol in batches of 100 using
          yfinance multi-ticker mode (~50× faster than per-symbol calls).
          Computes price, 30d avg volume, ATR-14, and gap from this data.
          Eliminates ~90% of the universe immediately. Full OHLCV stored in
          _daily_cache so Phase 2 never re-downloads daily bars.

        Phase 2a — batch pre-market volume (survivors only, batched):
          Downloads 1m intraday bars with prepost=True for all survivors in
          batches of 50, extracting 04:00–09:30 volume. One batch call covers
          50 symbols vs the old 50 individual calls.

        Phase 2b — build metrics from cached data (no network calls):
          Constructs SymbolScanMetrics entirely from Phase 1 + 2a cache.
          No per-symbol HTTP requests needed.

        Phase 3 — float, score, rank.
        """
        universe = self.load_universe()
        cfg = self.config
        market = cfg.market.lower()
        logger.info("Scanner starting: %d symbols, market=%s", len(universe), market)

        from app.services.markets import is_india_symbol as _is_india
        india_syms = [s for s in universe if _is_india(s)]
        us_syms    = [s for s in universe if not _is_india(s)]

        # ── Phase 1: bulk OHLCV download + pre-filter ─────────────────────────
        survivors: list[str] = []
        daily_data: dict[str, pd.DataFrame] = {}

        if us_syms:
            us_survivors, us_daily = self._bulk_download_and_filter(us_syms)
            survivors.extend(us_survivors)
            daily_data.update(us_daily)
            logger.info("Phase 1 US: %d/%d passed", len(us_survivors), len(us_syms))

        if india_syms:
            in_survivors, in_daily = self._india_download_and_filter(india_syms)
            survivors.extend(in_survivors)
            daily_data.update(in_daily)
            logger.info("Phase 1 India: %d/%d passed", len(in_survivors), len(india_syms))

        logger.info("Phase 1 complete: %d/%d total survivors", len(survivors), len(universe))
        if not survivors:
            logger.warning("Phase 1 eliminated all symbols — check filters")
            return []

        # ── Phase 2a: opening-activity volume for survivors ───────────────────
        # US: pre-market 04:00–09:30 ET via yfinance prepost batches
        # India: first-30-min session volume 09:15–09:45 IST via Upstox batches
        us_surv    = [s for s in survivors if not _is_india(s)]
        india_surv = [s for s in survivors if _is_india(s)]

        pm_vols: dict[str, float] = {}
        if us_surv:
            pm_vols.update(self._batch_premarket_volume(us_surv))
        if india_surv:
            pm_vols.update(self._india_opening_volume(india_surv))

        logger.info("Phase 2a complete: activity volume for %d symbols", len(pm_vols))

        # ── Phase 2b: build metrics from cache — no HTTP calls ────────────────
        metrics_list: list[SymbolScanMetrics] = []
        for sym in survivors:
            try:
                m = self._metrics_from_cache(sym, daily_data[sym], pm_vols.get(sym, 0.0))
                metrics_list.append(m)
            except Exception as e:
                logger.debug("metrics_from_cache failed for %s: %s", sym, e)

        logger.info("Phase 2b complete: %d metrics built from cache", len(metrics_list))

        # ── Phase 3a: float (only when filters active) ────────────────────────
        if cfg.min_float or cfg.max_float:
            self._attach_floats(metrics_list)

        # ── Phase 3b: score and rank ───────────────────────────────────────────
        results = [self.score_symbol(m) for m in metrics_list]
        passed  = [r for r in results if not r.rejection_reason]
        logger.info(
            "Scan complete: %d passed all filters out of %d survivors (%d universe)",
            len(passed), len(metrics_list), len(universe),
        )
        passed.sort(key=lambda r: r.score, reverse=True)
        return passed[:max_symbols]

    # ── Phase 1: bulk download ────────────────────────────────────────────────

    def _bulk_download_and_filter(
        self, universe: list[str]
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """
        Download 35d daily OHLCV for the full universe in batches of 100.
        Returns (survivors, daily_df_map) where daily_df_map covers survivors only.

        Cached in-process for the calendar day — subsequent scans (e.g. UI
        refresh with different top-N) reuse the cache and skip all downloads.
        """
        today = date.today()
        cfg   = self.config

        if DayTradingScanner._daily_cache_date == today and DayTradingScanner._daily_cache:
            logger.info("Phase 1: reusing in-process OHLCV cache (%d symbols)", len(DayTradingScanner._daily_cache))
            daily_data = DayTradingScanner._daily_cache
            survivors  = self._apply_prefilters(daily_data, cfg)
            return survivors, {s: daily_data[s] for s in survivors}

        BATCH_SIZE    = 100
        BATCH_WORKERS = 8

        batches = [universe[i: i + BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]
        logger.info("Phase 1: %d batches × %d symbols, %d workers", len(batches), BATCH_SIZE, BATCH_WORKERS)

        all_daily: dict[str, pd.DataFrame] = {}

        def _fetch_ohlcv_batch(syms: list[str]) -> dict[str, pd.DataFrame]:
            try:
                df = yf.download(
                    " ".join(syms),
                    period="35d",
                    interval="1d",
                    progress=False,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                )
                if df.empty:
                    return {}
                out: dict[str, pd.DataFrame] = {}
                multi = isinstance(df.columns, pd.MultiIndex)
                for sym in syms:
                    try:
                        sym_df = df[sym].dropna(how="all") if multi else df
                        if sym_df is None or len(sym_df) < 10:
                            continue
                        # Normalise column names
                        sym_df = sym_df.copy()
                        sym_df.index = pd.to_datetime(sym_df.index)
                        out[sym] = sym_df
                    except Exception:
                        continue
                return out
            except Exception as e:
                logger.debug("OHLCV batch failed: %s", e)
                return {}

        with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as pool:
            futs = {pool.submit(_fetch_ohlcv_batch, b): b for b in batches}
            for fut in as_completed(futs):
                try:
                    all_daily.update(fut.result())
                except Exception as e:
                    logger.debug("Batch future error: %s", e)

        logger.info("Phase 1 download: OHLCV for %d/%d symbols", len(all_daily), len(universe))

        DayTradingScanner._daily_cache      = all_daily
        DayTradingScanner._daily_cache_date = today

        survivors = self._apply_prefilters(all_daily, cfg)
        return survivors, {s: all_daily[s] for s in survivors}

    @staticmethod
    def _apply_prefilters(
        daily_map: dict[str, pd.DataFrame],
        cfg: "DayTradingScannerConfig",
    ) -> list[str]:
        """
        Cheap pre-filters computed entirely from cached OHLCV — no network calls.
        Eliminates price-out-of-range, low-volume, and ATR-out-of-range symbols.
        """
        survivors: list[str] = []
        for sym, df in daily_map.items():
            try:
                if df is None or len(df) < 10:
                    continue
                price = float(df["Close"].iloc[-1])
                if price <= 0 or price < cfg.min_price:
                    continue
                if cfg.max_price and price > cfg.max_price:
                    continue

                avg_vol = float(df["Volume"].tail(30).mean())
                # Relaxed volume floor (50%) — score_symbol re-applies the strict
                # threshold; this just prunes the obvious non-starters.
                if avg_vol < cfg.min_avg_volume * 0.5:
                    continue

                # ATR pre-filter — eliminates the comatose and the wildly volatile
                # before Phase 2 even starts.
                atr = _compute_daily_atr(df, period=14)
                if atr > 0:
                    atr_pct = atr / price * 100
                    if atr_pct < cfg.min_atr_pct * 0.5:   # too dead
                        continue
                    if atr_pct > cfg.max_atr_pct * 1.5:   # way too wild
                        continue

                survivors.append(sym)
            except Exception:
                continue
        return survivors

    # ── Phase 2a: batch pre-market volume ────────────────────────────────────

    def _batch_premarket_volume(self, symbols: list[str]) -> dict[str, float]:
        """
        Fetch 1m intraday bars for all survivors in batches of 50.
        Extracts 04:00–09:30 ET volume from each symbol's bars.

        A single multi-ticker yf.download call for 50 symbols is ~50× faster
        than 50 individual calls because Yahoo batches the HTTP request.
        """
        today = date.today()
        if DayTradingScanner._pm_vol_cache_date == today and DayTradingScanner._pm_vol_cache:
            # Return cached values; compute only the symbols not yet cached.
            cached = {s: DayTradingScanner._pm_vol_cache[s]
                      for s in symbols if s in DayTradingScanner._pm_vol_cache}
            missing = [s for s in symbols if s not in cached]
            if not missing:
                logger.info("Phase 2a: all %d pm-vol values from cache", len(symbols))
                return cached
            logger.info("Phase 2a: %d from cache, %d to fetch", len(cached), len(missing))
            fresh = self._fetch_pm_vol_batched(missing)
            DayTradingScanner._pm_vol_cache.update(fresh)
            cached.update(fresh)
            return cached

        result = self._fetch_pm_vol_batched(symbols)
        DayTradingScanner._pm_vol_cache      = dict(result)
        DayTradingScanner._pm_vol_cache_date = today
        return result

    @staticmethod
    def _fetch_pm_vol_batched(symbols: list[str]) -> dict[str, float]:
        """Download 1m bars for a list of symbols in batches of 50 and extract pre-market volume."""
        from app.services.strategy.daytrading.market_open import _normalise_yf

        BATCH_SIZE    = 50
        BATCH_WORKERS = 6   # kept moderate — 1m prepost data is heavier than daily

        batches = [symbols[i: i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        result: dict[str, float] = {}

        def _pm_batch(syms: list[str]) -> dict[str, float]:
            try:
                df = yf.download(
                    " ".join(syms),
                    period="1d",
                    interval="1m",
                    prepost=True,
                    progress=False,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                )
                if df.empty:
                    return {}
                out: dict[str, float] = {}
                multi = isinstance(df.columns, pd.MultiIndex)
                for sym in syms:
                    try:
                        sym_df = df[sym].dropna(how="all") if multi else df
                        if sym_df is None or sym_df.empty:
                            continue
                        sym_df = _normalise_yf(sym_df)
                        pm_mask = (sym_df.index.time >= time(4, 0)) & (sym_df.index.time < time(9, 30))
                        pm_vol = float(sym_df.loc[pm_mask, "Volume"].sum()) if pm_mask.any() else 0.0
                        out[sym] = pm_vol
                    except Exception:
                        continue
                return out
            except Exception as e:
                logger.debug("PM-vol batch failed: %s", e)
                return {}

        with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as pool:
            futs = {pool.submit(_pm_batch, b): b for b in batches}
            for fut in as_completed(futs):
                try:
                    result.update(fut.result())
                except Exception:
                    pass

        return result

    # ── India Phase 1: Upstox daily OHLCV download + pre-filter ─────────────

    def _india_download_and_filter(
        self, symbols: list[str]
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """
        Fetch 60d daily OHLCV for NSE symbols via Upstox in parallel.
        Upstox has no multi-ticker batch endpoint, so we use a thread pool.
        184 Nifty-200 symbols × ~0.3s/call ÷ 16 workers ≈ 4s total.
        """
        today = date.today()
        if (DayTradingScanner._india_daily_cache_date == today
                and DayTradingScanner._india_daily_cache):
            cached = DayTradingScanner._india_daily_cache
            logger.info("Phase 1 India: reusing cache (%d symbols)", len(cached))
            survivors = self._apply_prefilters(cached, self.config)
            return survivors, {s: cached[s] for s in survivors}

        from app.services.marketdata import upstox_data

        def _fetch_one(sym: str) -> tuple[str, pd.DataFrame | None]:
            try:
                df = upstox_data.fetch_bars(sym, interval="1d", period="60d")
                if df is not None and not df.empty and len(df) >= 10:
                    return sym, df
            except Exception as e:
                logger.debug("Upstox daily fetch failed for %s: %s", sym, e)
            return sym, None

        workers = min(16, max(1, int(self.config.max_workers)))
        all_daily: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one, s): s for s in symbols}
            for fut in as_completed(futs):
                sym, df = fut.result()
                if df is not None:
                    all_daily[sym] = df

        logger.info("India OHLCV: fetched %d/%d symbols via Upstox", len(all_daily), len(symbols))
        DayTradingScanner._india_daily_cache      = all_daily
        DayTradingScanner._india_daily_cache_date = today

        survivors = self._apply_prefilters(all_daily, self.config)
        return survivors, {s: all_daily[s] for s in survivors}

    # ── India Phase 2a: opening-activity volume (first 30 min) ───────────────

    def _india_opening_volume(self, symbols: list[str]) -> dict[str, float]:
        """
        Fetch NSE opening-activity volume via Upstox 5m bars.

        NSE has no pre-market session. The first 30 minutes (09:15–09:45 IST)
        is the equivalent signal: high volume in this window = catalyst/news
        just as pre-market volume does for US stocks.

        Uses the pm-vol cache so repeated scans are free.
        """
        today = date.today()
        if DayTradingScanner._pm_vol_cache_date == today:
            cached = {s: DayTradingScanner._pm_vol_cache[s]
                      for s in symbols if s in DayTradingScanner._pm_vol_cache}
            missing = [s for s in symbols if s not in cached]
            if not missing:
                return cached
            fresh = self._fetch_india_opening_vol(missing)
            DayTradingScanner._pm_vol_cache.update(fresh)
            cached.update(fresh)
            return cached

        result = self._fetch_india_opening_vol(symbols)
        DayTradingScanner._pm_vol_cache.update(result)
        DayTradingScanner._pm_vol_cache_date = today
        return result

    @staticmethod
    def _fetch_india_opening_vol(symbols: list[str]) -> dict[str, float]:
        """Fetch first-30-min volume for NSE symbols via Upstox 5m bars in parallel."""
        from app.services.marketdata import upstox_data
        from datetime import time as _time
        import pytz

        IST = pytz.timezone("Asia/Kolkata")
        OPEN_END = _time(9, 45)  # first 30 min of NSE session

        def _fetch_one(sym: str) -> tuple[str, float]:
            try:
                df = upstox_data.fetch_bars(sym, interval="5m", period="2d")
                if df is None or df.empty:
                    return sym, 0.0
                today_date = date.today()
                today_bars = df[df.index.date == today_date]
                opening = today_bars[today_bars.index.time <= OPEN_END]
                return sym, float(opening["Volume"].sum()) if not opening.empty else 0.0
            except Exception:
                return sym, 0.0

        workers = min(16, 4)  # Upstox rate-limits at higher concurrency
        result: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one, s): s for s in symbols}
            for fut in as_completed(futs):
                sym, vol = fut.result()
                result[sym] = vol
        return result

    # ── Phase 2b: build metrics from cached OHLCV ────────────────────────────

    def _metrics_from_cache(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        pm_vol: float,
    ) -> SymbolScanMetrics:
        """
        Build a complete SymbolScanMetrics from already-downloaded data.
        Zero network calls — everything comes from Phase 1 + 2a caches.
        """
        cfg = self.config
        last_price  = float(daily_df["Close"].iloc[-1])
        avg_vol_30d = float(daily_df["Volume"].tail(30).mean())
        atr_14      = _compute_daily_atr(daily_df, period=14)
        atr_pct     = (atr_14 / last_price * 100) if last_price > 0 else 0.0

        prior_close = float(daily_df["Close"].iloc[-2]) if len(daily_df) >= 2 else last_price
        today_open  = float(daily_df["Open"].iloc[-1])
        today_bar   = _is_today(daily_df)
        open_price  = today_open if today_bar else last_price
        if not today_bar:
            prior_close = last_price

        gap_pct  = (open_price - prior_close) / prior_close * 100 if prior_close > 0 else 0.0
        abs_gap  = abs(gap_pct)
        if abs_gap < cfg.gap_flat_pct:
            gap_size, gap_dir = "none", "flat"
        elif abs_gap < cfg.gap_small_pct:
            gap_size, gap_dir = "small", "up" if gap_pct > 0 else "down"
        elif abs_gap < cfg.gap_medium_pct:
            gap_size, gap_dir = "medium", "up" if gap_pct > 0 else "down"
        elif abs_gap < cfg.gap_large_pct:
            gap_size, gap_dir = "large", "up" if gap_pct > 0 else "down"
        else:
            gap_size, gap_dir = "extreme", "up" if gap_pct > 0 else "down"

        rel_vol        = pm_vol / avg_vol_30d if avg_vol_30d > 0 else 0.0
        catalyst_tags  = _detect_catalyst(symbol, daily_df)

        return SymbolScanMetrics(
            symbol=symbol,
            last_price=last_price,
            avg_daily_volume_30d=avg_vol_30d,
            atr_14=round(atr_14, 4),
            atr_pct=round(atr_pct, 3),
            premarket_volume=pm_vol,
            premarket_rel_vol=round(rel_vol, 4),
            premarket_gap_pct=round(gap_pct, 3),
            gap_direction=gap_dir,
            gap_size=gap_size,
            has_catalyst=len(catalyst_tags) > 0,
            catalyst_tags=catalyst_tags,
            catalyst_source="heuristic" if catalyst_tags else "none",
            prior_close=round(prior_close, 4),
            today_open=round(open_price, 4),
            data_quality="ok",
        )

    def _attach_floats(self, metrics_list: list[SymbolScanMetrics]) -> None:
        """Populate ``shares_float`` on every metric using the daily cache.

        Only symbols that already cleared the cheap price + liquidity gates
        get a yfinance lookup — this keeps the float fetch capped at a few
        hundred calls per day even on a 5,800-symbol universe.

        The cache is keyed by symbol and refreshed once per calendar day.
        """
        cfg = self.config
        # First pass: stamp every metric with whatever the cache already has.
        for m in metrics_list:
            cached = self._float_cache.get(m.symbol)
            if cached is not None:
                m.shares_float = float(cached)

        # Decide which symbols are worth fetching float for — only the ones
        # that survive the cheap price/volume gates AND don't already have
        # a fresh cached value.
        needs_fetch: list[str] = []
        for m in metrics_list:
            if m.shares_float > 0:
                continue
            if m.data_quality == "no_data":
                continue
            if m.last_price < cfg.min_price:
                continue
            if cfg.max_price and m.last_price > cfg.max_price:
                continue
            if m.avg_daily_volume_30d < cfg.min_avg_volume:
                continue
            needs_fetch.append(m.symbol)

        if not needs_fetch:
            return

        logger.info("Fetching float for %d symbols", len(needs_fetch))
        new_floats: dict[str, float] = {}
        # Keep float fetch workers low — yfinance's fast_info reuses a shared
        # crumb token, and at >4 concurrent requests we hit 401 "Invalid Crumb"
        # rate-limit responses that wipe out a chunk of the fetch.
        float_workers = min(4, max(1, int(cfg.max_workers)))
        with ThreadPoolExecutor(max_workers=float_workers) as pool:
            futures = {pool.submit(_fetch_float, sym): sym for sym in needs_fetch}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    val = fut.result()
                    if val > 0:
                        new_floats[sym] = val
                except Exception as e:
                    logger.debug("float fetch failed for %s: %s", sym, e)

        # Merge + persist
        self._float_cache.update(new_floats)
        save_float_cache(self._float_cache)

        # Stamp the newly fetched floats onto the metrics objects.
        for m in metrics_list:
            if m.shares_float == 0:
                val = self._float_cache.get(m.symbol)
                if val:
                    m.shares_float = float(val)

    def get_intraday_watchlist(
        self,
        max_symbols: int = 20,
        market_state: str | None = None,
        *,
        run_native_precheck: bool = True,
        precheck_top_k: int = 10,
        mover_snapshot=None,   # MarketMoverSnapshot | None — from market_movers.py
    ) -> list[SymbolScanResult]:
        """
        Brain-aware wrapper around scan().

        If a brain was passed at construction time:
          - Queries the brain for the current market state (TREND_UP, TREND_DOWN,
            CHOPPY, HIGH_VOL, NEWS_RISK).
          - Applies a per-state multiplier to each symbol's score based on how
            well its strategy bucket fits the current environment.
          - NEWS_RISK: caps the watchlist at 5 names regardless of max_symbols.

        If no brain is available, returns the raw scan() output.

        Parameters
        ----------
        run_native_precheck : when True (default), runs the audited 4-strategy
            generate_signals() check on the top ``precheck_top_k`` candidates
            and promotes any with native_signal_active=True to the front of
            the watchlist. The check is bounded (10 yfinance fetches by
            default), so the cost is fixed regardless of universe size.
        precheck_top_k : how many post-regime top candidates the pre-check
            runs on. Default 10. Set to 0 to disable equivalent to
            run_native_precheck=False.
        mover_snapshot : optional MarketMoverSnapshot (from market_movers.py).
            When provided, the CandidateRanker blends mover prominence (which
            lists the symbol is on, its rank on each list) into adjusted_score
            so pre-market movers surface above equally-scored non-movers.
        """
        candidates = self.scan(max_symbols=max_symbols * 3)  # over-fetch, then trim
        if not candidates:
            return []

        # Resolve market state
        state = market_state
        if state is None and self._brain is not None:
            state = self._resolve_market_state()

        if state is None:
            # No brain / no state — apply mover blend then pre-check on raw top-K.
            if mover_snapshot is not None:
                try:
                    from app.services.strategy.daytrading.candidate_ranker import (
                        CandidateRanker,
                    )
                    from app.services.strategy.daytrading.market_movers import (
                        merge_duplicate_symbols,
                    )
                    mover_contexts = merge_duplicate_symbols(mover_snapshot)
                    ranker = CandidateRanker()
                    ranked = ranker.rank(
                        symbols=[r.symbol for r in candidates],
                        mover_contexts=mover_contexts,
                        market_state="UNKNOWN",
                    )
                    ranked_map = {rc.symbol: rc for rc in ranked}
                    for r in candidates:
                        rc = ranked_map.get(r.symbol)
                        if rc is not None:
                            r.adjusted_score = round(
                                0.80 * r.adjusted_score + 0.20 * rc.total_score, 4
                            )
                    candidates.sort(key=lambda r: r.adjusted_score, reverse=True)
                except Exception as _e:
                    logger.warning("Mover ranker blend failed: %s", _e)
            watchlist = candidates[:max_symbols]
            if run_native_precheck and precheck_top_k > 0:
                self._apply_native_precheck(
                    watchlist, market_state=None, top_k=precheck_top_k,
                )
                watchlist.sort(key=_precheck_sort_key, reverse=True)
            return watchlist

        # Apply regime adjustment multipliers
        multipliers = _regime_strategy_multipliers(state)
        for r in candidates:
            bucket_mult = multipliers.get(r.recommended_strategy_bucket, 0.8)
            # Also penalise very high ATR names in choppy/high-vol markets
            if state in ("CHOPPY", "HIGH_VOL") and r.metrics.atr_pct > 5.0:
                bucket_mult *= 0.7
            r.adjusted_score = round(r.score * bucket_mult, 4)

        candidates.sort(key=lambda r: r.adjusted_score, reverse=True)

        # ── Mover-ranker blend ────────────────────────────────────────────────
        # If a mover snapshot was supplied, run CandidateRanker to compute a
        # mover-prominence boost and blend it into adjusted_score (weight 20%).
        # This brings pre-market movers to the surface without overriding the
        # signal-quality scores the scan() step calculated.
        if mover_snapshot is not None:
            try:
                from app.services.strategy.daytrading.candidate_ranker import (
                    CandidateRanker,
                )
                from app.services.strategy.daytrading.market_movers import (
                    merge_duplicate_symbols,
                )
                mover_contexts = merge_duplicate_symbols(mover_snapshot)
                ranker = CandidateRanker()
                scanner_results_map = {
                    r.symbol: {
                        "score": r.score,
                        "adjusted_score": r.adjusted_score,
                        "tags": r.tags,
                        "recommended_strategy_bucket": r.recommended_strategy_bucket,
                        "metrics": r.metrics.to_dict(),
                    }
                    for r in candidates
                }
                ranked = ranker.rank(
                    symbols=[r.symbol for r in candidates],
                    mover_contexts=mover_contexts,
                    market_state=state or "UNKNOWN",
                    scanner_results=scanner_results_map,
                )
                ranked_map = {rc.symbol: rc for rc in ranked}
                for r in candidates:
                    rc = ranked_map.get(r.symbol)
                    if rc is not None:
                        # Blend: 80% scanner adjusted_score + 20% ranker total_score.
                        r.adjusted_score = round(
                            0.80 * r.adjusted_score + 0.20 * rc.total_score, 4
                        )
                candidates.sort(key=lambda r: r.adjusted_score, reverse=True)
            except Exception as _e:
                logger.warning("Mover ranker blend failed, using scanner scores: %s", _e)

        cap = 5 if state == "NEWS_RISK" else max_symbols
        watchlist = candidates[:cap]

        # ── Native-signal pre-check on the top-K ──────────────────────────────
        # Promotes "signaling right now" candidates above "looks good today but
        # nothing is firing." Sorted score is preserved as the tiebreaker.
        if run_native_precheck and precheck_top_k > 0:
            self._apply_native_precheck(
                watchlist, market_state=state, top_k=precheck_top_k,
            )
            watchlist.sort(key=_precheck_sort_key, reverse=True)

        logger.info(
            f"Watchlist ready: {len(watchlist)} symbols (market_state={state})"
        )
        return watchlist

    def scan_diagnostics(self, sample_rejections: int = 12) -> dict[str, Any]:
        """Explain *why* the universe collapsed: per-phase survivor counts and a
        histogram of rejection reasons.

        Runs the full scan() (reusing the in-process daily / pm-vol caches, so on
        a warm cache this is nearly free) and then re-scores every survivor to
        tally how many were dropped by each hard filter. Read-only — does not arm
        anything. Surfaced by the ``/daytrading/scanner/diagnostics`` endpoint.
        """
        universe = self.load_universe()
        # Build metrics for survivors using the same pipeline scan() uses.
        # scan() already populates the caches; calling it warms them and gives us
        # the passing set. We then independently score the survivors to bucket
        # rejections (score_symbol stamps rejection_reason for each).
        passed = self.scan(max_symbols=10_000)  # effectively "all that pass"
        passed_syms = {r.symbol for r in passed}

        # Rebuild metrics for the cached survivors to classify rejections.
        from app.services.markets import is_india_symbol as _is_india
        daily = DayTradingScanner._daily_cache
        pm = DayTradingScanner._pm_vol_cache
        rejection_hist: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        survivors_scored = 0

        for sym, df in (daily or {}).items():
            try:
                m = self._metrics_from_cache(sym, df, pm.get(sym, 0.0))
            except Exception:
                continue
            survivors_scored += 1
            result = self.score_symbol(m)
            if result.rejection_reason:
                bucket = _bucket_rejection(result.rejection_reason)
                rejection_hist[bucket] = rejection_hist.get(bucket, 0) + 1
                if len(examples.setdefault(bucket, [])) < 3:
                    examples[bucket].append(f"{sym}: {result.rejection_reason}")

        return {
            "market": self.config.market,
            "universe_size": len(universe),
            "phase1_survivors_scored": survivors_scored,
            "passed_all_filters": len(passed_syms),
            "rejection_histogram": _sorted_rejection_hist(rejection_hist),
            "rejection_examples": {k: examples[k] for k in list(examples)[:sample_rejections]},
            "config_snapshot": {
                "min_price": self.config.min_price,
                "max_price": self.config.max_price,
                "min_avg_volume": self.config.min_avg_volume,
                "min_atr_pct": self.config.min_atr_pct,
                "max_atr_pct": self.config.max_atr_pct,
                "min_float": self.config.min_float,
                "max_float": self.config.max_float,
            },
        }

    def _apply_native_precheck(
        self,
        watchlist: list["SymbolScanResult"],
        market_state: str | None,
        top_k: int,
    ) -> None:
        """Run native_precheck on the top-K of `watchlist` and stamp results in place."""
        # Lazy import: keeps the scanner module's import surface unchanged and
        # avoids paying for autotrader/strategy imports when precheck is off.
        from app.services.strategy.daytrading.scanners.native_precheck import (
            run_native_precheck,
        )

        top = watchlist[:top_k]
        if not top:
            return
        results = run_native_precheck(
            symbols=[r.symbol for r in top],
            market_state=market_state,
        )
        for r in top:
            check = results.get(r.symbol)
            r.native_precheck_ran = True
            if check is None:
                continue
            r.native_signal_active = check.native_signal_active
            r.active_native_strategies = list(check.active_native_strategies)
            r.best_native_strategy = check.best_native_strategy
            r.best_native_side = check.best_native_side
            r.best_native_confidence = check.best_native_confidence

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_market_state(self) -> str | None:
        """Ask the brain for today's market state using SPY 5m data."""
        try:
            from app.services.strategy.daytrading.runner import fetch_intraday
            df_spy = fetch_intraday("SPY", interval="5m", period="2d")
            if df_spy.empty:
                return None
            status = self._brain.evaluate_market_state(df_spy, df_spy)
            return status.market_state
        except Exception as e:
            logger.warning("Could not resolve market state from brain: %s", e)
            return None

    def _fetch_metrics_impl(self, symbol: str) -> SymbolScanMetrics:
        """Core data-fetching logic. Documented inline."""

        # ── Step 1: Try BarCache for 1d bars ─────────────────────────────────
        daily_df = self._get_daily_bars(symbol)

        if daily_df is None or len(daily_df) < 5:
            return SymbolScanMetrics(
                symbol=symbol,
                last_price=0.0, avg_daily_volume_30d=0.0,
                atr_14=0.0, atr_pct=0.0,
                premarket_volume=0.0, premarket_rel_vol=0.0,
                premarket_gap_pct=0.0, gap_direction="flat", gap_size="none",
                has_catalyst=False, catalyst_tags=[], data_quality="no_data",
            )

        # ── Step 2: Basic price + volume metrics ──────────────────────────────
        last_price = float(daily_df["Close"].iloc[-1])
        avg_vol_30d = float(daily_df["Volume"].tail(30).mean())

        # ── Step 3: ATR-14 on daily bars ──────────────────────────────────────
        atr_14 = _compute_daily_atr(daily_df, period=14)
        atr_pct = (atr_14 / last_price * 100) if last_price > 0 else 0.0

        # ── Step 4: Gap + pre-market data ─────────────────────────────────────
        # yfinance does not reliably provide pre-market OHLCV for all symbols.
        # Strategy:
        #   a. Pull daily 'Open' for today. If today has an Open, compute gap
        #      as (today_open - prior_close) / prior_close.
        #   b. Attempt to pull pre-market volume from a 5m intraday download
        #      filtered to 04:00–09:30 ET. If empty, use first two 5m bars of
        #      regular session as a proxy (they carry residual pre-market demand).
        prior_close = float(daily_df["Close"].iloc[-2]) if len(daily_df) >= 2 else last_price
        today_open = float(daily_df["Open"].iloc[-1])
        today_is_new_bar = _is_today(daily_df)

        # If today's row is present in daily data, use its Open; else last_price
        if today_is_new_bar:
            open_price = today_open
        else:
            open_price = last_price  # approximation when market hasn't opened yet
            prior_close = last_price  # last close IS the prior close

        gap_pct = (open_price - prior_close) / prior_close * 100 if prior_close > 0 else 0.0

        # Pre-market volume via 5m intraday bars
        pm_vol = self._get_premarket_volume(symbol, avg_vol_30d)

        # ── Step 5: Classify gap ──────────────────────────────────────────────
        abs_gap = abs(gap_pct)
        cfg = self.config
        if abs_gap < cfg.gap_flat_pct:
            gap_size = "none"
            gap_dir = "flat"
        elif abs_gap < cfg.gap_small_pct:
            gap_size = "small"
            gap_dir = "up" if gap_pct > 0 else "down"
        elif abs_gap < cfg.gap_medium_pct:
            gap_size = "medium"
            gap_dir = "up" if gap_pct > 0 else "down"
        elif abs_gap < cfg.gap_large_pct:
            gap_size = "large"
            gap_dir = "up" if gap_pct > 0 else "down"
        else:
            gap_size = "extreme"
            gap_dir = "up" if gap_pct > 0 else "down"

        # ── Step 6: Catalyst detection (stub — extend with news API) ──────────
        # Currently we flag potential catalyst if earnings are nearby.
        # To add real news: replace _detect_catalyst with an API call.
        catalyst_tags = _detect_catalyst(symbol, daily_df)
        has_catalyst = len(catalyst_tags) > 0

        rel_vol = pm_vol / avg_vol_30d if avg_vol_30d > 0 else 0.0

        return SymbolScanMetrics(
            symbol=symbol,
            last_price=last_price,
            avg_daily_volume_30d=avg_vol_30d,
            atr_14=round(atr_14, 4),
            atr_pct=round(atr_pct, 3),
            premarket_volume=pm_vol,
            premarket_rel_vol=round(rel_vol, 4),
            premarket_gap_pct=round(gap_pct, 3),
            gap_direction=gap_dir,
            gap_size=gap_size,
            has_catalyst=has_catalyst,
            catalyst_tags=catalyst_tags,
            catalyst_source="heuristic" if has_catalyst else "none",
            prior_close=round(prior_close, 4),
            today_open=round(open_price, 4),
            data_quality="ok",
        )

    def _get_daily_bars(self, symbol: str) -> pd.DataFrame | None:
        """Return daily OHLCV. Checks process-level daily cache first (free), then network."""
        from app.services.strategy.daytrading.market_open import _td_fetch

        # 1. In-process daily cache built by Phase 1 — zero cost
        if (DayTradingScanner._daily_cache_date == date.today()
                and symbol in DayTradingScanner._daily_cache):
            return DayTradingScanner._daily_cache[symbol]

        # 2. BarCache (warm in-process bar store)
        if self._cache is not None:
            cached = self._cache.get_bars(symbol, "1d")
            if cached is not None and len(cached) >= 10:
                return cached

        # 3. Twelve Data
        df = _td_fetch(symbol, "1d", "60d")
        if not df.empty:
            return df

        # 4. yfinance fallback
        try:
            df = yf.download(symbol, period="60d", interval="1d", progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.debug("daily fetch failed for %s: %s", symbol, e)
            return None

    def _get_premarket_volume(self, symbol: str, avg_daily_vol: float) -> float:
        """Return pre-market volume (04:00–09:30 ET). Checks pm-vol cache first."""
        # Phase 2a batch cache — free if scan() already ran
        if (DayTradingScanner._pm_vol_cache_date == date.today()
                and symbol in DayTradingScanner._pm_vol_cache):
            return DayTradingScanner._pm_vol_cache[symbol]

        from app.services.strategy.daytrading.market_open import _td_fetch, _normalise_yf
        try:
            df = _td_fetch(symbol, "1m", "1d")
            if not df.empty:
                pm_mask = (df.index.time >= time(4, 0)) & (df.index.time < time(9, 30))
                if pm_mask.any():
                    return float(df.loc[pm_mask, "Volume"].sum())

            df = yf.download(symbol, period="1d", interval="1m", prepost=True, progress=False)
            if not df.empty:
                df = _normalise_yf(df)
                pm_mask = (df.index.time >= time(4, 0)) & (df.index.time < time(9, 30))
                if pm_mask.any():
                    return float(df.loc[pm_mask, "Volume"].sum())

            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _reject(metrics: SymbolScanMetrics, reason: str) -> SymbolScanResult:
        return SymbolScanResult(
            symbol=metrics.symbol,
            score=0.0,
            adjusted_score=0.0,
            tags=[],
            recommended_strategy_bucket="none",
            metrics=metrics,
            rejection_reason=reason,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _fetch_float(symbol: str) -> float:
    """Return the shares float (or shares outstanding) for ``symbol``.

    Strategy:
      1. Try yfinance ``fast_info.shares`` first — light, fast, reliable, and
         survives the rate-limit / crumb-401 storms that ``.info`` runs into.
         This returns shares outstanding, which is a close upper bound on
         float for most stocks (float = shares out minus restricted shares;
         the gap is small for established names).
      2. Fall back to ``.info["floatShares"]`` for the exact figure when
         ``fast_info`` is unavailable — slower, more likely to be rate-limited.
      3. Return 0.0 on total failure (caller treats 0 as "unknown" and will
         reject the symbol if a float filter is active).
    """
    try:
        t = yf.Ticker(symbol)
        try:
            shares = getattr(t.fast_info, "shares", None)
            if shares and shares > 0:
                return float(shares)
        except Exception:
            pass
        # Fallback to the heavier .info endpoint
        info = t.info or {}
        val = info.get("floatShares") or info.get("sharesOutstanding")
        return float(val) if val else 0.0
    except Exception:
        return 0.0


def _bucket_rejection(reason: str) -> str:
    """Collapse a free-text rejection reason into a stable category for the
    diagnostics histogram. Mirrors the strings score_symbol() emits."""
    r = reason.lower()
    if "no_data" in r:
        return "no_data"
    if "price" in r and "< min" in r:
        return "price_below_min"
    if "price" in r and "> max" in r:
        return "price_above_max"
    if "avg_vol" in r:
        return "liquidity_below_min"
    if "float" in r and "< min" in r:
        return "float_below_min"
    if "float" in r and "> max" in r:
        return "float_above_max"
    if "too low" in r or "dead" in r:
        return "atr_too_low"
    if "too wild" in r:
        return "atr_too_high"
    return "other"


def _sorted_rejection_hist(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Return the rejection histogram as a sorted list for deterministic JSON."""
    return [
        {"reason": k, "count": v}
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _compute_daily_atr(df: pd.DataFrame, period: int = 14) -> float:
    """True-range ATR on daily bars. No external dependency needed."""
    if len(df) < period + 1:
        return 0.0
    try:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else 0.0
    except Exception:
        return 0.0


def _is_today(df: pd.DataFrame) -> bool:
    """True if the last row of the dataframe corresponds to today's date."""
    if df.empty:
        return False
    last_idx = pd.to_datetime(df.index[-1])
    last_date = last_idx.date() if hasattr(last_idx, "date") else last_idx.to_pydatetime().date()
    return last_date == date.today()


def _fallback_premarket_vol(symbol: str) -> float:
    """Return 0.0 — no reliable pre-market volume source available.

    Previous versions returned the first two regular-session 5m bars as a
    proxy, but that is NOT pre-market volume and inflated rel_vol scores.
    Callers treat 0.0 as "unavailable" and score it neutrally.
    """
    return 0.0


def _detect_catalyst(symbol: str, daily_df: pd.DataFrame) -> list[str]:
    """
    Lightweight catalyst detection using volume anomaly as a proxy.

    A real implementation would call an earnings calendar API
    (e.g., yfinance Ticker.calendar, Alpha Vantage, Polygon.io).

    Current approach:
      - If yesterday's volume was >2× the 20-day average → tag "vol_anomaly"
        (likely driven by news, earnings, or an analyst action).
      - Returns empty list if insufficient data.

    TODO: Replace with a real earnings-calendar call, e.g.:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal and "Earnings Date" in cal: ...
    """
    tags: list[str] = []
    try:
        if len(daily_df) >= 21:
            vol_20d_avg = float(daily_df["Volume"].iloc[-21:-1].mean())
            last_vol = float(daily_df["Volume"].iloc[-1])
            if vol_20d_avg > 0 and last_vol > vol_20d_avg * 2.0:
                tags.append("vol_anomaly")

        # Attempt yfinance earnings calendar (often empty for ETFs / non-US ADRs).
        # Suppress yfinance HTTP errors — they are expected for ETFs.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                cal = ticker.calendar
            # yfinance returns a dict like {"Earnings Date": [...]}
            if cal and isinstance(cal, dict):
                earnings_dates = cal.get("Earnings Date", [])
                if earnings_dates:
                    today = date.today()
                    for ed in earnings_dates:
                        try:
                            ed_date = pd.Timestamp(ed).date()
                            delta = (ed_date - today).days
                            if -1 <= delta <= 1:    # earnings within ±1 day
                                tags.append("earnings_nearby")
                                break
                        except Exception:
                            pass
        except Exception:
            pass  # calendar not available — silently skip
    except Exception:
        pass
    return tags


def _pick_bucket(
    tags: list[str],
    metrics: SymbolScanMetrics,
    config: DayTradingScannerConfig,
) -> str:
    """
    Choose one primary strategy bucket from the tags and metrics.

    Priority:
      1. gap_and_go   → "gap"
      2. ORB + high ATR → "ORB"
      3. momentum     → "momentum"
      4. VWAP or gap_fade → "VWAP"
      5. default      → "ORB"
    """
    tag_set = set(tags)
    if "gap_and_go" in tag_set:
        return "gap"
    if "ORB" in tag_set and metrics.atr_pct >= config.ideal_atr_low:
        return "ORB"
    if "momentum" in tag_set:
        return "momentum"
    if "VWAP" in tag_set or "gap_fade_candidate" in tag_set:
        return "VWAP"
    return "ORB"


def _precheck_sort_key(r: "SymbolScanResult") -> tuple[int, float, float]:
    """Sort key that promotes native_signal_active=True ahead of inactive picks.

    Within each tier, ties are broken by best_native_confidence (when present)
    and finally by adjusted_score. Used with reverse=True so higher tiers and
    higher numerics come first.
    """
    tier = 1 if r.native_signal_active else 0
    conf = r.best_native_confidence if r.best_native_confidence is not None else 0.0
    return (tier, conf, r.adjusted_score)


def _regime_strategy_multipliers(state: str) -> dict[str, float]:
    """
    Per-market-state score multipliers by strategy bucket.

    A multiplier < 1.0 downweights a bucket that doesn't fit the current
    environment. A multiplier > 1.0 upweights a good fit.
    """
    _MULTIPLIERS: dict[str, dict[str, float]] = {
        # Strong directional day — ORB, momentum, gap-and-go all shine
        "TREND_UP": {
            "ORB": 1.2,
            "gap": 1.2,
            "momentum": 1.1,
            "VWAP": 0.8,
        },
        # Weak/down day — fades and mean-reversion > breakouts
        "TREND_DOWN": {
            "ORB": 0.7,
            "gap": 0.9,     # gap fades work on down days
            "momentum": 0.9,
            "VWAP": 1.1,
        },
        # Range-bound — VWAP and gap-fades dominate, ORB breaks fail
        "CHOPPY": {
            "ORB": 0.6,
            "gap": 0.85,
            "momentum": 0.7,
            "VWAP": 1.2,
        },
        # Elevated vol — keep to VWAP, avoid wide-stop strategies
        "HIGH_VOL": {
            "ORB": 0.75,
            "gap": 0.75,
            "momentum": 0.85,
            "VWAP": 1.1,
        },
        # News catalyst risk — cut list to safest names only
        "NEWS_RISK": {
            "ORB": 0.5,
            "gap": 0.6,
            "momentum": 0.6,
            "VWAP": 0.9,
        },
    }
    return _MULTIPLIERS.get(state, {})
