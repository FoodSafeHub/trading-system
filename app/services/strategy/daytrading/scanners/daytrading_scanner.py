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
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from app.services.strategy.daytrading.market_open import ET, now_et

logger = logging.getLogger(__name__)

# ── Default symbol universe ───────────────────────────────────────────────────
# Covers the most liquid large-caps plus ETFs that day traders watch daily.
# The user can override this at construction time.
_DEFAULT_UNIVERSE: list[str] = [
    # Mega-cap tech / growth
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    # Financials + other large caps
    "JPM", "BAC", "GS", "MS",
    # ETFs
    "SPY", "QQQ", "IWM", "XLK", "XLF",
    # High-beta / momentum favourites
    "PLTR", "COIN", "MSTR", "HOOD", "SOFI",
]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DayTradingScannerConfig:
    """All hard-filter and scoring thresholds in one place."""

    # ── Hard filters ──────────────────────────────────────────────────────────
    min_avg_volume: float = 1_000_000   # shares/day — drop thinly traded names
    min_price: float = 5.0              # skip penny stocks
    max_price: float | None = None      # None = no cap
    min_atr_pct: float = 1.0            # too dead to trade intraday
    max_atr_pct: float = 8.0            # too wild (blow-up risk)

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
    # Raw snapshot for debugging
    prior_close: float = 0.0
    today_open: float = 0.0
    data_quality: str = "ok"               # "ok" | "partial" | "no_data"

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
            "prior_close": round(self.prior_close, 4),
            "today_open": round(self.today_open, 4),
            "data_quality": self.data_quality,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "tags": self.tags,
            "recommended_strategy_bucket": self.recommended_strategy_bucket,
            "metrics": self.metrics.to_dict(),
            "rejection_reason": self.rejection_reason,
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
        self._universe = [s.upper() for s in (universe or _DEFAULT_UNIVERSE)]
        self._brain = brain
        self._cache = cache  # BarCache instance, or None → fetch direct

    # ── Public API ────────────────────────────────────────────────────────────

    def load_universe(self) -> list[str]:
        """Return the symbol universe this scanner will evaluate."""
        return list(self._universe)

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

    def scan(self, max_symbols: int = 50) -> list[SymbolScanResult]:
        """
        Run the full pre-market scan:
          1. Load universe.
          2. Fetch metrics per symbol (parallel-friendly, but sequential here).
          3. Score each symbol.
          4. Return top-N by score (rejected symbols excluded).
        """
        results: list[SymbolScanResult] = []
        universe = self.load_universe()
        logger.info("Scanner starting: %d symbols in universe", len(universe))

        for symbol in universe:
            metrics = self.fetch_metrics(symbol)
            result = self.score_symbol(metrics)
            results.append(result)

        # Split passed vs rejected for logging
        passed = [r for r in results if not r.rejection_reason]
        rejected = [r for r in results if r.rejection_reason]
        logger.info(
            "Scan complete: %d passed filters, %d rejected",
            len(passed), len(rejected),
        )
        for r in rejected:
            logger.debug("  SKIP %s — %s", r.symbol, r.rejection_reason)

        passed.sort(key=lambda r: r.score, reverse=True)
        return passed[:max_symbols]

    def get_intraday_watchlist(
        self,
        max_symbols: int = 20,
        market_state: str | None = None,
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
        """
        candidates = self.scan(max_symbols=max_symbols * 3)  # over-fetch, then trim
        if not candidates:
            return []

        # Resolve market state
        state = market_state
        if state is None and self._brain is not None:
            state = self._resolve_market_state()

        if state is None:
            # No brain / no state — return raw top-N
            return candidates[:max_symbols]

        # Apply regime adjustment multipliers
        multipliers = _regime_strategy_multipliers(state)
        for r in candidates:
            bucket_mult = multipliers.get(r.recommended_strategy_bucket, 0.8)
            # Also penalise very high ATR names in choppy/high-vol markets
            if state in ("CHOPPY", "HIGH_VOL") and r.metrics.atr_pct > 5.0:
                bucket_mult *= 0.7
            r.adjusted_score = round(r.score * bucket_mult, 4)

        candidates.sort(key=lambda r: r.adjusted_score, reverse=True)

        cap = 5 if state == "NEWS_RISK" else max_symbols
        watchlist = candidates[:cap]

        logger.info(
            "Watchlist ready: %d symbols (market_state=%s)", len(watchlist), state
        )
        return watchlist

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
            prior_close=round(prior_close, 4),
            today_open=round(open_price, 4),
            data_quality="ok",
        )

    def _get_daily_bars(self, symbol: str) -> pd.DataFrame | None:
        """Return 60-day daily OHLCV. Tries BarCache first, then yfinance."""
        # Try cache first (BarCache stores "1d" timeframe)
        if self._cache is not None:
            cached = self._cache.get_bars(symbol, "1d")
            if cached is not None and len(cached) >= 10:
                return cached

        # Fall back to yfinance
        try:
            df = yf.download(symbol, period="60d", interval="1d", progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.debug("yfinance daily fetch failed for %s: %s", symbol, e)
            return None

    def _get_premarket_volume(self, symbol: str, avg_daily_vol: float) -> float:
        """
        Attempt to get pre-market volume (04:00–09:30 ET).

        yfinance 1m data covers pre-market when 'prepost=True' is set, but
        coverage is unreliable. Falls back to first two 5m intraday bars
        (9:30–9:40 ET) as a rough proxy for overnight demand.
        """
        try:
            df = yf.download(
                symbol, period="1d", interval="1m",
                prepost=True, progress=False,
            )
            if df.empty:
                return _fallback_premarket_vol(symbol)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            idx = pd.to_datetime(df.index)
            # Normalise timezone
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC").tz_convert(ET)
            else:
                idx = idx.tz_convert(ET)
            df.index = idx

            # Pre-market window: 04:00–09:29 ET
            pm_mask = (df.index.time >= time(4, 0)) & (df.index.time < time(9, 30))
            pm_df = df[pm_mask]
            if not pm_df.empty:
                return float(pm_df["Volume"].sum())

            # Fallback: use 5m intraday (regular session) first two bars as proxy
            return _fallback_premarket_vol(symbol)

        except Exception:
            return _fallback_premarket_vol(symbol)

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
    """
    Pre-market volume approximation using first two 5m intraday bars
    (9:30–9:40 ET). These bars carry higher-than-normal volume when there
    was genuine pre-market interest. Returns 0.0 on failure.
    """
    try:
        df = yf.download(symbol, period="1d", interval="5m", progress=False)
        if df.empty:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.to_datetime(df.index)
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC").tz_convert(ET)
        else:
            idx = idx.tz_convert(ET)
        df.index = idx
        # First two 5m bars of regular session
        reg_mask = df.index.time >= time(9, 30)
        reg_df = df[reg_mask]
        if len(reg_df) >= 2:
            return float(reg_df["Volume"].iloc[:2].sum())
        elif not reg_df.empty:
            return float(reg_df["Volume"].iloc[0])
        return 0.0
    except Exception:
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
