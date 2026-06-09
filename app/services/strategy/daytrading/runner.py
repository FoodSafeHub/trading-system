from __future__ import annotations

import time as _time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

import logging

from app.services.strategy.daytrading.brain import DayTradingBrain
from app.services.strategy.daytrading.brain.config_adjuster import ConfigAdjuster
from app.services.strategy.daytrading.brain.strategy_selector import StrategySelector
from app.services.strategy.daytrading.brain.symbol_profiles import SymbolAnalyzer
from app.services.strategy.daytrading.brain.trade_explainer import TradeExplainer
from app.services.strategy.daytrading.backtest_exit_simulator import (
    SimulatedExit, aggregate_pnl, simulate_exit,
)
from app.services.strategy.daytrading.execution.fill_simulator import FillConfig, FillSimulator
from app.services.strategy.daytrading.market_open import (
    ET,
    apply_choppy_penalty,
    compute_vwap,
    get_spy_regime,
    is_market_open,
    market_session,
    market_status,
    regime_allows_strategy,
)
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.risk_templates import (
    get_symbol_bucket, position_size as risk_position_size,
)
from app.services.strategy.daytrading.pipeline_diagnostics import (
    PipelineDiagnostics, _categorise_rejection,
)
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP
from app.services.strategy.daytrading.brain.symbol_policy import allows_live, allows_scan

from datetime import time as _time

logger = logging.getLogger(__name__)

_FIRST_HOUR_END = _time(10, 30)   # signals/bars at or before this are "first hour"

# Process-level provider counters. Tracks how many fetch_intraday() calls each
# provider has served since the API process started. Read by the dashboard
# provider-health badge via /daytrading/data-source-status. Cleared on restart.
_PROVIDER_COUNTERS: dict[str, int] = {
    "twelvedata": 0,
    "webull": 0,
    "yfinance": 0,
    "upstox": 0,
    "empty": 0,
}


def get_provider_stats() -> dict[str, Any]:
    """Snapshot of provider usage since process start. Returns total + per-provider counts."""
    total = sum(_PROVIDER_COUNTERS.values())
    served = total - _PROVIDER_COUNTERS["empty"]
    # Last-used = provider with most recent successful serve. We approximate via
    # "primary" = the most-used non-empty provider in this session.
    primary = max(
        (p for p in ("twelvedata", "webull", "yfinance")),
        key=lambda p: _PROVIDER_COUNTERS[p],
    ) if served > 0 else None
    return {
        "counters": dict(_PROVIDER_COUNTERS),
        "total_calls": total,
        "served_calls": served,
        "empty_calls": _PROVIDER_COUNTERS["empty"],
        "primary_provider": primary,
    }

# Module-level brain singleton
_brain = DayTradingBrain()


# Twelve Data interval map: yfinance-style -> Twelve Data format
_TD_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "1d": "1day",
}

# period string -> approximate outputsize (number of bars to request)
_TD_OUTPUTSIZE = {
    "1d":  390,    # 1 trading day of 1m bars
    "2d":  780,
    "5d":  390,    # 5d of 5m bars ≈ 390 bars
    "60d": 780,    # 60d of 15m bars ≈ 780 bars
    "730d": 1000,
}


def _fetch_twelvedata(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch intraday bars from Twelve Data REST API. Returns empty DataFrame on failure."""
    from app.services.strategy.daytrading.data_providers import td_breaker
    if td_breaker.is_tripped():
        return pd.DataFrame()
    try:
        from app.config import get_settings
        api_key = get_settings().twelve_data_api_key
        if not api_key:
            return pd.DataFrame()

        td_interval = _TD_INTERVAL_MAP.get(interval)
        if not td_interval:
            return pd.DataFrame()

        outputsize = _TD_OUTPUTSIZE.get(period, 500)

        import requests
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": outputsize,
            "timezone": "America/New_York",
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "error" or "values" not in data:
            msg = data.get("message", "no values") or "no values"
            td_breaker.note_response_text(msg)
            logger.warning("[twelvedata] %s %s: %s", symbol, interval, msg)
            return pd.DataFrame()

        records = data["values"]
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Localize to ET
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(ET)
        else:
            df.index = df.index.tz_convert(ET)

        logger.info("[twelvedata] %s %s %s -> %d bars", symbol, interval, period, len(df))
        return df

    except Exception as e:
        logger.warning("[twelvedata] fetch failed for %s %s: %s", symbol, interval, e)
        return pd.DataFrame()


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and ensure ET timezone."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        idx = idx.tz_convert(ET)
    df.index = idx
    return df


def fetch_intraday(
    symbol: str,
    interval: str = "5m",
    period: str = "5d",
    diag: PipelineDiagnostics | None = None,
) -> pd.DataFrame:
    """Download intraday bars.

    India (NSE) symbols route to Upstox intraday, which is the only provider here
    that serves NSE bars (Twelve Data/Webull/yfinance are US-oriented and return
    empty for NSE intraday). US symbols use the chain: Twelve Data -> Webull ->
    yfinance. India bars stay in IST; US bars are normalised to ET.
    """
    df = pd.DataFrame()
    source = "yfinance"

    # 0) India (NSE) — Upstox intraday. Short-circuits the US provider chain.
    from app.services.markets import is_india_symbol
    if is_india_symbol(symbol):
        from app.services.marketdata import upstox_data
        ind = upstox_data.fetch_bars(symbol, interval=interval, period=period)
        if not ind.empty:
            ind.attrs["source"] = "upstox"
            _PROVIDER_COUNTERS["upstox"] = _PROVIDER_COUNTERS.get("upstox", 0) + 1
            logger.info("[upstox-intraday] %s %s %s -> %d bars",
                        symbol, interval, period, len(ind))
            return ind
        if diag is not None:
            diag.data_warning = (
                f"No Upstox intraday data for {symbol} {interval} {period}. "
                f"Check Upstox login (token expires daily ~03:30 IST) and that "
                f"the NSE symbol resolves."
            )
        logger.warning("fetch_intraday: 0 Upstox bars for %s %s %s", symbol, interval, period)
        _PROVIDER_COUNTERS["empty"] += 1
        return ind  # empty — do NOT fall through to US providers for an NSE name

    # 1) Twelve Data — primary for intraday intervals
    if interval in _TD_INTERVAL_MAP and interval != "1d":
        df = _fetch_twelvedata(symbol, interval, period)
        if not df.empty:
            source = "twelvedata"

    # 2) Webull — first fallback (covers TD rate-limit, auth error, region gate)
    if df.empty:
        from app.services.strategy.daytrading.data_providers.webull_md import fetch_webull
        wb = fetch_webull(symbol, interval, period)
        if not wb.empty:
            logger.info("[fallback] TD -> Webull for %s %s %s (%d bars)",
                        symbol, interval, period, len(wb))
            df = wb
            source = "webull"

    # 3) yfinance — second fallback
    if df.empty:
        logger.info("[fallback] Webull -> yfinance for %s %s %s",
                    symbol, interval, period)
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            if diag is not None:
                diag.data_warning = (
                    f"No data for {symbol} {interval} {period} "
                    f"(tried Twelve Data + Webull + yfinance)"
                )
            logger.warning("fetch_intraday: 0 bars for %s %s %s", symbol, interval, period)
            _PROVIDER_COUNTERS["empty"] += 1
            return df
        df = _normalise_df(df)
        source = "yfinance"

    # Attach provider attribution so downstream callers can read df.attrs["source"]
    df.attrs["source"] = source
    _PROVIDER_COUNTERS[source] = _PROVIDER_COUNTERS.get(source, 0) + 1

    if diag is not None and interval == "5m":
        mh = df.between_time("09:30", "16:00")
        diag.bars_loaded_5m = len(df)
        diag.bars_in_market_hours = len(mh)
        diag.earliest_bar = str(df.index[0]) if not df.empty else ""
        diag.latest_bar = str(df.index[-1]) if not df.empty else ""
        diag.timezone = str(df.index.tzinfo)
        logger.info(
            "fetch_intraday [%s]: %s %s %s -> %d bars (%d mkt-hrs) [%s .. %s]",
            source, symbol, interval, period, len(df), len(mh),
            diag.earliest_bar[:16], diag.latest_bar[:16],
        )
        if len(df) < 10:
            diag.data_warning = (
                f"Only {len(df)} bars loaded for {symbol} {period} — "
                f"source={source}"
            )
    elif diag is not None and interval == "15m":
        diag.bars_loaded_15m = len(df)

    return df


def profile_symbol(symbol: str, period: str = "60d") -> dict[str, Any]:
    """
    Download intraday data and compute a full SymbolProfile.
    Returns a dict ready for JSON serialisation.
    """
    df_5m = fetch_intraday(symbol, interval="5m", period=period)
    if df_5m.empty:
        return {"error": f"No data available for {symbol}"}
    profile = SymbolAnalyzer.analyze(df_5m, symbol)
    return profile.to_dict()


def run_signals(
    symbol: str,
    enabled_strategies: list[str] | None = None,
    custom_configs: dict[str, dict] | None = None,
    apply_brain: bool = True,
    today_trades: list[dict] | None = None,
    initial_capital: float = 100_000.0,
    open_positions: int = 0,
) -> dict[str, Any]:
    """
    Fetch latest intraday data, detect regime, run all enabled strategies.
    If apply_brain=True, signals are filtered through the brain pipeline.
    Returns a dict with regime info, brain status, signal dicts, and diagnostics.
    """
    diag = PipelineDiagnostics(symbol=symbol, period="5d")

    status = market_status()
    df_5m  = fetch_intraday(symbol, interval="5m",  period="5d",  diag=diag)
    df_15m = fetch_intraday(symbol, interval="15m", period="60d", diag=diag)

    spy_df = fetch_intraday("SPY", interval="5m", period="2d") if symbol != "SPY" else df_5m
    regime, spy_vs_vwap, gap_pct, gap_type = get_spy_regime(spy_df)

    # ── Deployment policy gate ────────────────────────────────────────────────
    live_ok, live_reason = allows_live(symbol, regime)
    if not live_ok:
        diag.data_warning = f"Policy blocked: {live_reason}"
        diag.finalise()
        return {
            "signals": [],
            "regime": regime,
            "policy_blocked": True,
            "policy_reason": live_reason,
            "diagnostics": diag.to_dict(),
            "market_status": status,
        }

    raw_signals: list[dict] = []
    strategies_run: list[str] = []
    strategies_skipped: list[str] = []

    # Build symbol profile once for auto-config (mirrors backtest path so live
    # and backtest use the same tuned parameters when no custom_config is given).
    _live_profile = SymbolAnalyzer.analyze(df_5m, symbol) if not df_5m.empty else None

    for strategy in ALL_STRATEGIES:
        if enabled_strategies and strategy.name not in enabled_strategies:
            continue
        if not regime_allows_strategy(regime, strategy.name):
            strategies_skipped.append(strategy.name)
            logger.debug(
                "run_signals: %s SKIPPED strategy=%s (regime=%s not allowed)",
                symbol, strategy.name, regime,
            )
            continue

        strategies_run.append(strategy.name)
        # Prefer explicit custom_config; fall back to auto-tuned config so live
        # and backtest both run on symbol-profile-adjusted parameters.
        cfg = (custom_configs or {}).get(strategy.name)
        if cfg is None and _live_profile is not None:
            cfg = ConfigAdjuster.adjust(strategy.name, _live_profile).adjusted or None
        before = len(raw_signals)
        try:
            sigs = strategy.generate_signals(df_5m, df_15m, symbol, cfg, regime)
        except Exception as e:
            logger.warning("run_signals: strategy %s raised %s for %s", strategy.name, e, symbol)
            sigs = []

        for sig in sigs:
            sig.confidence = apply_choppy_penalty(sig.confidence, regime)
            raw_signals.append(asdict(sig))

        generated = len(raw_signals) - before
        logger.debug(
            "run_signals: %s strategy=%s regime=%s -> %d raw signals",
            symbol, strategy.name, regime, generated,
        )

    diag.strategies_run = strategies_run
    diag.strategies_skipped_by_regime = strategies_skipped
    diag.raw_signals_generated = len(raw_signals)
    diag.raw_buy_signals  = sum(1 for s in raw_signals if s.get("direction") == "BUY")
    diag.raw_sell_signals = sum(1 for s in raw_signals if s.get("direction") in ("SELL", "SELL_SHORT"))
    diag.raw_hold_signals = sum(1 for s in raw_signals if s.get("direction") == "HOLD")

    logger.info(
        "run_signals: %s regime=%s raw=%d (buy=%d sell=%d) strategies=%s skipped=%s",
        symbol, regime,
        diag.raw_signals_generated, diag.raw_buy_signals, diag.raw_sell_signals,
        strategies_run, strategies_skipped,
    )

    brain_status_dict: dict = {}
    accepted_signals: list[dict] = []
    rejected_signals: list[dict] = []

    if apply_brain and raw_signals:
        brain_status = _brain.build_status(
            df_5m, spy_df if symbol != "SPY" else None,
            today_trades=today_trades,
            initial_capital=initial_capital,
            open_positions=open_positions,
        )
        brain_status_dict = {
            "market_state": brain_status.market_state,
            "state_confidence": brain_status.state_confidence,
            "state_reasons": brain_status.state_reasons,
            "enabled_strategies": brain_status.enabled_strategies,
            "disabled_strategies": brain_status.disabled_strategies,
            "kill_switch": brain_status.kill_switch,
            "kill_switch_reason": brain_status.kill_switch_reason,
            "trades_today": brain_status.trades_today,
            "losses_in_a_row": brain_status.losses_in_a_row,
            "daily_pnl_pct": brain_status.daily_pnl_pct,
            "size_multiplier": brain_status.size_multiplier,
            "routing_summary": brain_status.routing_summary,
        }

        if brain_status.kill_switch:
            logger.warning(
                "run_signals: %s KILL SWITCH active — %s",
                symbol, brain_status.kill_switch_reason,
            )

        decisions = _brain.filter_signals(
            raw_signals,
            account_state={
                "today_trades": today_trades or [],
                "initial_capital": initial_capital,
                "open_positions": open_positions,
            },
        )

        rejection_sample: list[str] = []
        for dec in decisions:
            sig_out = dict(dec.signal or {})
            sig_out["brain_accepted"] = dec.accepted
            sig_out["brain_reason"] = dec.rejection_reason or dec.explanation
            sig_out["brain_size_multiplier"] = dec.size_multiplier
            sig_out["brain_market_state"] = dec.market_state
            if dec.accepted:
                accepted_signals.append(sig_out)
            else:
                rejected_signals.append(sig_out)
                reason_str = dec.rejection_reason or dec.explanation or ""
                cat = _categorise_rejection(reason_str)
                if cat == "regime":      diag.rejected_by_regime      += 1
                elif cat == "volume":    diag.rejected_by_volume       += 1
                elif cat == "rr":        diag.rejected_by_rr           += 1
                elif cat == "time":      diag.rejected_by_time         += 1
                elif cat == "extension": diag.rejected_by_extension    += 1
                elif cat == "kill_switch": diag.rejected_by_kill_switch += 1
                else:                    diag.rejected_other           += 1
                if len(rejection_sample) < 5:
                    rejection_sample.append(f"[{cat}] {reason_str[:80]}")

        diag.rejected_by_brain_total = len(rejected_signals)
        diag.accepted_signals = len(accepted_signals)
        diag.rejection_reasons = rejection_sample

        # ── First-hour window instrumentation ──────────────────────────────
        fh_bars_today = df_5m.between_time("09:30", "10:30")
        diag.first_hour_bars_loaded = len(fh_bars_today)
        fh_rej_counts: dict[str, int] = {}
        for sig_d in raw_signals:
            try:
                st = pd.Timestamp(sig_d.get("signal_time", ""))
                if st.tzinfo is None:
                    st = st.tz_localize(ET)
                if st.time() <= _FIRST_HOUR_END:
                    diag.first_hour_raw_signals += 1
            except Exception:
                pass
        for dec in decisions:
            try:
                st = pd.Timestamp((dec.signal or {}).get("signal_time", ""))
                if st.tzinfo is None:
                    st = st.tz_localize(ET)
                if st.time() <= _FIRST_HOUR_END:
                    if not dec.accepted:
                        diag.first_hour_brain_rejections += 1
                        cat = _categorise_rejection(dec.rejection_reason or "")
                        fh_rej_counts[cat] = fh_rej_counts.get(cat, 0) + 1
            except Exception:
                pass
        diag.first_hour_rejection_counts = fh_rej_counts
        if fh_rej_counts:
            diag.first_hour_top_rejection_reason = max(fh_rej_counts, key=fh_rej_counts.get)

        logger.info(
            "run_signals: %s brain accepted=%d rejected=%d "
            "(regime=%d vol=%d rr=%d time=%d ext=%d kill=%d other=%d)",
            symbol,
            diag.accepted_signals, diag.rejected_by_brain_total,
            diag.rejected_by_regime, diag.rejected_by_volume, diag.rejected_by_rr,
            diag.rejected_by_time, diag.rejected_by_extension,
            diag.rejected_by_kill_switch, diag.rejected_other,
        )

    elif apply_brain and not raw_signals:
        # Brain not even invoked — no raw signals to filter
        logger.info("run_signals: %s brain skipped (0 raw signals)", symbol)
        diag.accepted_signals = 0
    else:
        # apply_brain=False: pass everything through
        accepted_signals = raw_signals
        diag.accepted_signals = len(raw_signals)

    diag.finalise()

    return {
        "symbol": symbol,
        "regime": regime,
        "spy_vs_vwap_pct": spy_vs_vwap,
        "market_status": status,
        "signal_count": len(accepted_signals),
        "signals": accepted_signals,
        "rejected_signals": rejected_signals,
        "raw_signal_count": len(raw_signals),
        "brain": brain_status_dict,
        "diagnostics": diag.to_dict(),
    }


def run_backtest(
    symbol: str,
    strategy_name: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.95,
    interval: str = "5m",
    custom_config: dict | None = None,
    use_auto_config: bool = True,
    commission_per_share: float = 0.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """
    Bar-by-bar simulation of a single strategy.
    Returns trade log, performance metrics, symbol profile, and explanation
    if no trades fired.
    """
    strategy = STRATEGY_MAP.get(strategy_name)
    if strategy is None:
        return {"error": f"Unknown strategy: {strategy_name}"}

    _mkt_tz = market_session(symbol).tz   # IST for India, ET for US — for trade-time display

    diag = PipelineDiagnostics(symbol=symbol, period=period)

    df_5m  = fetch_intraday(symbol, interval="5m",  period=period,  diag=diag)
    df_15m = fetch_intraday(symbol, interval="15m", period="730d" if interval == "15m" else period, diag=diag)

    if df_5m.empty:
        diag.data_warning = f"No 5m data returned for {symbol} {period}"
        diag.finalise()
        return {"error": f"Insufficient intraday data for {symbol} {period}", "diagnostics": diag.to_dict()}

    # ── Price sanity guard ────────────────────────────────────────────────────
    # Strategies are calibrated for stocks above $2. Sub-$2 penny stocks produce
    # nonsensical signals (ATR < 1 cent, stop distances rounding to zero, vol
    # spikes on trivial dollar amounts). Reject early with a clear message.
    _last_price = float(df_5m["Close"].iloc[-1])
    _MIN_BACKTEST_PRICE = 2.0
    if _last_price < _MIN_BACKTEST_PRICE:
        return {
            "error": (
                f"{symbol} last price ${_last_price:.4f} is below the ${_MIN_BACKTEST_PRICE:.2f} "
                "minimum. The intraday strategies are not calibrated for penny stocks — "
                "stops and targets round to noise at sub-$2 price levels. "
                "Use a stock priced above $2 with at least 500K avg daily volume."
            ),
            "symbol": symbol,
            "strategy": strategy_name,
            "trades": [],
            "metrics": {"total_trades": 0},
            "analysis": {},
        }

    # ── Symbol profile + auto-config ─────────────────────────────────────────
    profile = SymbolAnalyzer.analyze(df_5m, symbol)
    adjustment = ConfigAdjuster.adjust(strategy_name, profile)
    effective_config = adjustment.adjusted if (use_auto_config and not custom_config) else (custom_config or {})

    # ── Strategy selection check ──────────────────────────────────────────────
    dates_all = sorted(set(df_5m.index.date))
    diag.trading_days_found = len(dates_all)
    regime_counts: dict[str, int] = {"BULL_OPEN": 0, "BEAR_OPEN": 0, "CHOPPY": 0}

    fill_sim = FillSimulator(FillConfig(
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
    ))
    trades: list[dict] = []
    equity = initial_capital
    total_commission = 0.0
    total_slippage = 0.0

    logger.info(
        "run_backtest: %s strategy=%s period=%s initial_days=%d bars_5m=%d",
        symbol, strategy_name, period, len(dates_all), len(df_5m),
    )

    for date in dates_all:
        day_5m = df_5m[df_5m.index.date == date]
        day_15m = df_15m[df_15m.index.date == date] if not df_15m.empty else pd.DataFrame()

        if len(day_5m) < 4:
            diag.trading_days_skipped_short += 1
            logger.debug("run_backtest: %s %s only %d bars — skipping", symbol, date, len(day_5m))
            continue

        # simplified regime: compare open vs prior close
        prev_days = df_5m[df_5m.index.date < date]
        if prev_days.empty:
            regime = "CHOPPY"
        else:
            prior_close = float(prev_days["Close"].iloc[-1])
            open_p = float(day_5m["Open"].iloc[0])
            current = float(day_5m["Close"].iloc[-1])
            vwap_series = compute_vwap(day_5m)
            vwap_val = float(vwap_series.iloc[-1])
            if open_p > prior_close and current > vwap_val:
                regime = "BULL_OPEN"
            elif open_p < prior_close and current < vwap_val:
                regime = "BEAR_OPEN"
            else:
                regime = "CHOPPY"

        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if not regime_allows_strategy(regime, strategy_name):
            diag.days_skipped_by_regime += 1
            logger.debug(
                "run_backtest: %s %s strategy=%s not allowed in regime=%s",
                symbol, date, strategy_name, regime,
            )
            continue

        try:
            # Pass full history up to (and including) the current date so
            # strategies can warm up multi-day indicators (BB, RSI, EMA, ATR)
            # without lookahead.  Strategies must slice to today internally.
            hist_5m = df_5m[df_5m.index.date <= date]
            signals = strategy.generate_signals(hist_5m, day_15m, symbol, effective_config, regime)
        except Exception as e:
            logger.warning("run_backtest: strategy %s raised %s on %s %s", strategy_name, e, symbol, date)
            continue

        day_raw = [s for s in signals if s.direction != "HOLD"]
        diag.raw_signals_generated += len(day_raw)
        diag.raw_buy_signals  += sum(1 for s in day_raw if s.direction == "BUY")
        diag.raw_sell_signals += sum(1 for s in day_raw if s.direction in ("SELL", "SELL_SHORT"))
        diag.raw_hold_signals += sum(1 for s in signals if s.direction == "HOLD")

        # First-hour bar count (9:30–10:30 ET)
        fh_bars = day_5m.between_time("09:30", "10:30")
        diag.first_hour_bars_loaded += len(fh_bars)
        # First-hour raw signals
        for s in day_raw:
            try:
                st = pd.Timestamp(s.signal_time)
                if st.tzinfo is None:
                    from app.services.strategy.daytrading.market_open import ET as _ET
                    st = st.tz_localize(_ET)
                if st.time() <= _FIRST_HOUR_END:
                    diag.first_hour_raw_signals += 1
            except Exception:
                pass

        if day_raw:
            logger.debug(
                "run_backtest: %s %s regime=%s -> %d raw signals",
                symbol, date, regime, len(day_raw),
            )

        for sig in signals:
            if sig.direction == "HOLD":
                continue

            entry = sig.entry_price
            stop = sig.stop_price
            target = sig.target_price

            sig_time = pd.Timestamp(sig.signal_time)
            future_bars = day_5m[day_5m.index > sig_time]

            if future_bars.empty:
                diag.trades_skipped_no_future_bars += 1
                logger.debug(
                    "run_backtest: %s %s no future bars after signal at %s — skipping",
                    symbol, strategy_name, sig_time,
                )
                continue

            # ── Position size: bucket-aware risk-based sizing ──────────────
            # Stop distance determines size so a 0.4–0.6%-of-equity loss limit
            # is honored regardless of the strategy's stop width. `position_pct`
            # remains as a notional cap so a tiny stop can't oversize.
            market = "NSE" if _is_india_for_sizing(symbol) else "US"
            bucket = get_symbol_bucket(symbol, market=market)
            position_size = _risk_based_size(
                equity=equity, bucket=bucket,
                entry=entry, stop=stop, notional_cap_pct=position_pct,
            )
            if position_size <= 0:
                diag.trades_skipped_no_future_bars += 1   # reused: invalid trade
                logger.debug(
                    "run_backtest: %s %s zero size (entry=%.4f stop=%.4f bucket=%s) — skipping",
                    symbol, strategy_name, entry, stop, bucket,
                )
                continue

            diag.trades_opened += 1
            # First-hour trade tracking
            try:
                st_check = pd.Timestamp(sig.signal_time)
                if st_check.tzinfo is None:
                    from app.services.strategy.daytrading.market_open import ET as _ET2
                    st_check = st_check.tz_localize(_ET2)
                if st_check.time() <= _FIRST_HOUR_END:
                    diag.first_hour_executed_trades += 1
            except Exception:
                pass

            # ── Exit simulation: honor ExitPlan if the strategy emitted one ─
            sim = simulate_exit(
                direction=sig.direction,
                entry_price=entry, initial_stop=stop, initial_target=target,
                qty=float(position_size), signal_time=sig_time,
                future_bars=future_bars, exit_plan=sig.exit_plan, symbol=symbol,
                max_hold_bars_default=strategy.default_config.get("max_hold_bars", 60),
            )
            agg = aggregate_pnl(sig.direction, entry, sim)
            exit_price = agg["exit_price"] if agg["qty"] > 0 else entry
            exit_time  = sim.legs[-1].time if sim.legs else sig_time
            outcome    = sim.primary_outcome
            hold_bars  = sim.hold_bars

            diag.trades_closed += 1

            # Fill-simulator: apply commission + slippage to both legs.
            # Costs are modelled on the full qty for a single round-trip (one
            # entry leg + one exit leg with the qty-weighted exit price). This
            # is consistent with how the metrics aggregator treats one signal
            # as one trade; multi-tranche leg-by-leg fill costs are a follow-up.
            vol_pct = profile.volatility_pct if profile else 1.0
            entry_fill = fill_sim.fill_entry(sig.direction, entry, position_size, vol_pct)
            exit_fill  = fill_sim.fill_exit(sig.direction, exit_price, position_size, vol_pct)
            fill_summary = fill_sim.summarize_trade(
                symbol, sig.direction, position_size, entry_fill, exit_fill
            )

            pnl = fill_summary.net_pnl
            gross_pnl = fill_summary.gross_pnl
            pnl_pct = pnl / (entry * position_size) * 100 if entry * position_size > 0 else 0.0
            total_commission += fill_summary.total_commission
            total_slippage   += fill_summary.total_slippage
            equity += pnl

            trades.append({
                "date": str(date),
                "symbol": symbol,
                "strategy": strategy_name,
                "direction": sig.direction,
                "entry_price": round(entry_fill.fill_price, 4),
                "exit_price": round(exit_fill.fill_price, 4),
                "stop_price": round(stop, 4),
                "target_price": round(target, 4),
                "entry_time": _ts_str(sig_time, _mkt_tz),
                "exit_time": _ts_str(exit_time, _mkt_tz),
                "hold_bars": hold_bars,
                "pnl": round(pnl, 2),
                "gross_pnl": round(gross_pnl, 2),
                "commission": round(fill_summary.total_commission, 4),
                "slippage": round(fill_summary.total_slippage, 4),
                "pnl_pct": round(pnl_pct, 4),
                "outcome": outcome,
                "regime": regime,
                "confidence": sig.confidence,
                # ExitPlan / sizing diagnostics
                "bucket": bucket,
                "position_size": float(position_size),
                "final_stop": round(sim.final_stop, 4),
                "breakeven_hit": sim.breakeven_hit,
                "trail_activated": sim.trail_activated,
                "exit_legs": [
                    {"qty": leg.qty, "price": round(leg.price, 4),
                     "reason": leg.reason, "time": _ts_str(leg.time, _mkt_tz)}
                    for leg in sim.legs
                ],
            })

    diag.regime_distribution = regime_counts
    diag.strategies_run = [strategy_name]
    diag.accepted_signals = diag.raw_signals_generated   # backtest has no brain filter
    logger.info(
        "run_backtest: %s %s -> days=%d skip_short=%d skip_regime=%d "
        "raw=%d trades_opened=%d trades_closed=%d skip_no_bars=%d",
        symbol, strategy_name,
        diag.trading_days_found, diag.trading_days_skipped_short, diag.days_skipped_by_regime,
        diag.raw_signals_generated, diag.trades_opened, diag.trades_closed,
        diag.trades_skipped_no_future_bars,
    )

    diag.finalise()
    result = _compute_metrics(trades, initial_capital, equity, symbol, strategy_name,
                              total_commission=total_commission, total_slippage=total_slippage)
    result["diagnostics"] = diag.to_dict()

    # ── Attach profile + config adjustment to every result ────────────────────
    result["symbol_profile"] = profile.to_dict()
    result["config_adjustment"] = {
        "changes": adjustment.changes,
        "reason_summary": adjustment.reason_summary,
        "has_changes": adjustment.has_changes,
        "adjusted_config": adjustment.adjusted,
    }

    # ── If 0 trades, produce a full explanation ───────────────────────────────
    if not trades:
        dominant_regime = max(regime_counts, key=regime_counts.get)
        # Use BULL_OPEN for symbol-fit scoring so regime doesn't zero-out
        # strategies that are valid for the symbol but happen to be off
        # during this period's dominant regime. Regime mismatch is explained
        # separately inside TradeExplainer._diagnose.
        selection = StrategySelector.select(symbol, "BULL_OPEN", profile)
        explanation = TradeExplainer.explain_empty_backtest(
            symbol=symbol,
            strategy=strategy_name,
            period=period,
            profile=profile,
            selection=selection,
            regime_counts=regime_counts,
        )
        result["explanation"] = explanation.to_dict()
        result["recommended_alternative"] = selection.primary
        result["selection"] = {
            "primary": selection.primary,
            "secondary": selection.secondary,
            "enabled": selection.enabled,
            "disabled_reasons": selection.disabled_reasons,
            "symbol_fit_scores": selection.symbol_fit_scores,
            "recommendation_text": selection.recommendation_text,
        }

    return result


def _ts_str(ts, tz=ET) -> str:
    """Return a plain tz-naive ISO string (no offset) in the given market tz.

    Defaults to US ET; callers pass the symbol's session tz (IST for India) so
    recorded trade times match the market the symbol trades on.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(tz).tz_localize(None)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _is_india_for_sizing(symbol: str) -> bool:
    """Used by backtest sizing to pick US vs NSE risk bucket. Safe-defaults to US."""
    try:
        from app.services.markets import is_india_symbol
        return bool(is_india_symbol(symbol))
    except Exception:
        return False


def _risk_based_size(
    equity: float, bucket: str, entry: float, stop: float,
    notional_cap_pct: float = 0.95,
) -> int:
    """Risk-budgeted share count, capped by notional exposure.

    Delegates to risk_templates.position_size (the canonical bucket-aware
    helper) for the risk math, then applies `notional_cap_pct` of equity as a
    hard ceiling so an unusually tight stop cannot oversize.

    Returns 0 (skip trade) when stop distance or equity is non-positive, when
    the entry price is non-positive, or when the bucketed share count rounds
    to 0 even after at-least-1-share floor.
    """
    if equity <= 0 or entry <= 0 or abs(entry - stop) <= 0:
        return 0
    shares_by_risk = risk_position_size(
        account_value=equity, bucket=bucket, entry=entry, stop=stop,
    )
    if shares_by_risk <= 0:
        return 0
    max_shares_by_notional = int((equity * notional_cap_pct) / entry)
    if max_shares_by_notional <= 0:
        return 0
    return min(shares_by_risk, max_shares_by_notional)


def _compute_metrics(
    trades: list[dict], initial_capital: float, final_equity: float,
    symbol: str, strategy_name: str,
    total_commission: float = 0.0, total_slippage: float = 0.0,
) -> dict[str, Any]:
    if not trades:
        return {
            "symbol": symbol, "strategy": strategy_name,
            "trades": [], "metrics": {"total_trades": 0},
            "analysis": _empty_analysis(),
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    total_pnl = df["pnl"].sum()
    win_rate = len(wins) / len(df) * 100
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = [initial_capital]
    running = initial_capital
    for p in df["pnl"]:
        running += p
        equity_curve.append(running)

    peak = initial_capital
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    returns = df["pnl_pct"] / 100
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0.0

    df["entry_hour"] = pd.to_datetime(df["entry_time"]).dt.hour
    df["entry_dow"] = pd.to_datetime(df["entry_time"]).dt.day_name()
    best_hour = df.groupby("entry_hour")["pnl"].mean().idxmax() if not df.empty else None
    best_dow = df.groupby("entry_dow")["pnl"].mean().idxmax() if not df.empty else None

    # max consecutive losses
    results = (df["pnl"] > 0).tolist()
    max_cl = cur_cl = 0
    for r in results:
        cur_cl = 0 if r else cur_cl + 1
        max_cl = max(max_cl, cur_cl)

    analysis = _build_analysis(df, wins, losses, equity_curve, strategy_name)

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "trades": trades,
        "equity_curve": equity_curve,
        "analysis": analysis,
        "metrics": {
            "total_trades": len(df),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((final_equity - initial_capital) / initial_capital * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_hold_bars": round(df["hold_bars"].mean(), 1),
            "avg_pnl_per_trade": round(df["pnl"].mean(), 2),
            "avg_pnl_pct": round(df["pnl_pct"].mean(), 4),
            "max_consecutive_losses": max_cl,
            "best_hour": int(best_hour) if best_hour is not None else None,
            "best_day_of_week": best_dow,
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "gross_pnl": round(df["gross_pnl"].sum() if "gross_pnl" in df.columns else total_pnl, 2),
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "net_pnl": round(total_pnl, 2),
        },
    }


def _empty_analysis() -> dict:
    return {
        "regime_breakdown": {},
        "hourly_performance": {},
        "dow_performance": {},
        "outcome_breakdown": {},
        "win_loss_streaks": {"max_wins": 0, "max_losses": 0},
        "avg_mae": 0.0,
        "avg_mfe": 0.0,
        "expectancy": 0.0,
        "diagnosis": [],
        "strengths": [],
        "weaknesses": [],
    }


def _build_analysis(
    df: pd.DataFrame,
    wins: pd.DataFrame,
    losses: pd.DataFrame,
    equity_curve: list,
    strategy_name: str,
) -> dict:
    """
    Deep trade analysis — surfaces patterns a trader can act on.
    Breaks down performance by regime, time of day, day of week, and outcome type.
    Adds diagnosis: what the strategy is doing right and wrong.
    """
    analysis: dict = {}

    # ── Regime breakdown ─────────────────────────────────────────────────────
    regime_stats: dict = {}
    for regime, grp in df.groupby("regime"):
        w = (grp["pnl"] > 0).sum()
        regime_stats[regime] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl": round(grp["pnl_pct"].mean(), 4),
            "total_pnl": round(grp["pnl"].sum(), 2),
        }
    analysis["regime_breakdown"] = regime_stats

    # ── Hourly performance ────────────────────────────────────────────────────
    hourly: dict = {}
    for hour, grp in df.groupby("entry_hour"):
        w = (grp["pnl"] > 0).sum()
        hourly[int(hour)] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["hourly_performance"] = hourly

    # ── Day of week ───────────────────────────────────────────────────────────
    dow: dict = {}
    for day, grp in df.groupby("entry_dow"):
        w = (grp["pnl"] > 0).sum()
        dow[str(day)] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["dow_performance"] = dow

    # ── Outcome breakdown ─────────────────────────────────────────────────────
    outcome_stats: dict = {}
    for outcome, grp in df.groupby("outcome"):
        w = (grp["pnl"] > 0).sum()
        outcome_stats[str(outcome)] = {
            "count": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["outcome_breakdown"] = outcome_stats

    # ── Win/Loss streaks ──────────────────────────────────────────────────────
    results_seq = (df["pnl"] > 0).tolist()
    max_wins = max_losses = cur_w = cur_l = 0
    for r in results_seq:
        if r:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_wins = max(max_wins, cur_w)
        max_losses = max(max_losses, cur_l)
    analysis["win_loss_streaks"] = {"max_wins": max_wins, "max_losses": max_losses}

    # ── Expectancy ────────────────────────────────────────────────────────────
    wr = len(wins) / len(df) if len(df) > 0 else 0
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
    expectancy = round(wr * avg_win - (1 - wr) * avg_loss, 4)
    analysis["expectancy"] = expectancy

    # ── Avg win / avg loss sizes ──────────────────────────────────────────────
    analysis["avg_win_pct"] = round(avg_win, 4)
    analysis["avg_loss_pct"] = round(avg_loss, 4)
    analysis["payoff_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    # ── Hold time analysis ────────────────────────────────────────────────────
    analysis["avg_win_hold_bars"] = round(wins["hold_bars"].mean(), 1) if len(wins) > 0 else 0
    analysis["avg_loss_hold_bars"] = round(losses["hold_bars"].mean(), 1) if len(losses) > 0 else 0

    # ── Natural language diagnosis ────────────────────────────────────────────
    diagnosis: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []

    win_rate = wr * 100

    # Win rate assessment
    if win_rate >= 60:
        strengths.append(f"Strong win rate of {win_rate:.1f}% — strategy is selective and accurate.")
    elif win_rate >= 50:
        strengths.append(f"Positive win rate {win_rate:.1f}% — profitable with good R:R.")
    elif win_rate >= 45:
        weaknesses.append(f"Win rate {win_rate:.1f}% is marginal — needs better filters or larger R:R targets.")
    else:
        weaknesses.append(f"Win rate {win_rate:.1f}% is below breakeven threshold — strategy needs rethinking.")

    # Payoff ratio
    pr = analysis["payoff_ratio"]
    if pr >= 2.0:
        strengths.append(f"Excellent payoff ratio {pr:.1f}:1 — winners are much larger than losers.")
    elif pr >= 1.5:
        strengths.append(f"Good payoff ratio {pr:.1f}:1 — risk/reward is working.")
    elif pr >= 1.0:
        diagnosis.append(f"Payoff ratio {pr:.1f}:1 is acceptable but targets could be extended.")
    else:
        weaknesses.append(f"Payoff ratio {pr:.1f}:1 means losers are bigger than winners — tighten stops or widen targets.")

    # Expectancy
    if expectancy > 0.3:
        strengths.append(f"Strong positive expectancy of +{expectancy:.3f}% per trade.")
    elif expectancy > 0:
        diagnosis.append(f"Positive expectancy +{expectancy:.3f}% — marginal edge, look for higher-confidence setups.")
    else:
        weaknesses.append(f"Negative expectancy {expectancy:.3f}% — strategy loses money on average per trade.")

    # Hold time: are wins being cut too early?
    if analysis["avg_win_hold_bars"] < analysis["avg_loss_hold_bars"]:
        weaknesses.append(
            f"Wins held {analysis['avg_win_hold_bars']:.0f} bars vs losses {analysis['avg_loss_hold_bars']:.0f} bars — "
            "cutting winners too early or letting losers run. Consider trailing stops."
        )
    else:
        strengths.append(
            f"Wins held longer ({analysis['avg_win_hold_bars']:.0f} bars) than losses ({analysis['avg_loss_hold_bars']:.0f} bars) — good trade management."
        )

    # Regime performance
    best_regime = max(regime_stats.items(), key=lambda x: x[1]["win_rate"]) if regime_stats else None
    worst_regime = min(regime_stats.items(), key=lambda x: x[1]["win_rate"]) if regime_stats else None
    if best_regime:
        diagnosis.append(f"Best regime: {best_regime[0]} ({best_regime[1]['win_rate']:.0f}% win rate, {best_regime[1]['trades']} trades).")
    if worst_regime and worst_regime[0] != (best_regime[0] if best_regime else None):
        if worst_regime[1]["win_rate"] < 40:
            weaknesses.append(f"Weak in {worst_regime[0]} regime ({worst_regime[1]['win_rate']:.0f}% win rate) — consider disabling in this regime.")

    # Outcome: EOD exits
    eod = outcome_stats.get("EOD_EXIT", {})
    if eod.get("count", 0) > len(df) * 0.3:
        pct = eod["count"] / len(df) * 100
        if eod.get("avg_pnl_pct", 0) < 0:
            weaknesses.append(f"{pct:.0f}% of trades hit end-of-day exit with avg loss — signals are being taken too late in the session.")
        else:
            diagnosis.append(f"{pct:.0f}% of trades resolved at end of day — these are time-forced exits, not technical exits.")

    # Max consecutive losses
    if max_losses >= 5:
        weaknesses.append(f"Max {max_losses} consecutive losses — consider a daily loss limit of 3 stopped trades.")
    elif max_losses <= 2:
        strengths.append(f"Max consecutive losses is only {max_losses} — low drawdown risk.")

    analysis["diagnosis"] = diagnosis
    analysis["strengths"] = strengths
    analysis["weaknesses"] = weaknesses

    return analysis


def run_backtest_all(
    symbol: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
) -> list[dict[str, Any]]:
    scan_ok, scan_reason = allows_scan(symbol)
    if not scan_ok:
        logger.info("run_backtest_all: %s excluded by policy — %s", symbol, scan_reason)
        return []
    results = []
    for strategy in ALL_STRATEGIES:
        result = run_backtest(symbol, strategy.name, period, initial_capital)
        if "metrics" in result:
            m = result["metrics"]
            results.append({
                "strategy": strategy.name,
                "trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "profit_factor": m.get("profit_factor", 0),
                "total_pnl": m.get("total_pnl", 0),
                "avg_pnl_pct": m.get("avg_pnl_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "best_hour": m.get("best_hour"),
                "best_day_of_week": m.get("best_day_of_week"),
            })
    results.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)
    return results


def run_backtest_with_brain(
    symbol: str,
    strategy_name: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.95,
) -> dict[str, Any]:
    """
    Run the same bar-by-bar backtest but filter every signal through the brain.
    Returns both raw and brain-filtered metrics side by side for comparison.
    """
    raw = run_backtest(symbol, strategy_name, period, initial_capital, position_pct)

    strategy = STRATEGY_MAP.get(strategy_name)
    if strategy is None:
        return raw

    _mkt_tz = market_session(symbol).tz   # IST for India, ET for US — for trade-time display

    df_5m = fetch_intraday(symbol, interval="5m", period=period)
    df_15m = fetch_intraday(symbol, interval="15m", period="730d")
    spy_df = fetch_intraday("SPY", interval="5m", period=period) if symbol != "SPY" else df_5m

    if df_5m.empty:
        return raw

    local_brain = DayTradingBrain()
    filtered_trades: list[dict] = []
    equity = initial_capital
    brain_total_commission = 0.0
    brain_total_slippage = 0.0
    fill_sim = FillSimulator(FillConfig(slippage_bps=2.0, commission_per_share=0.0))
    profile = SymbolAnalyzer.analyze(fetch_intraday(symbol, "5m", "60d"), symbol)
    dates = sorted(set(df_5m.index.date))

    # Diagnostics for the brain-filtered pass
    brain_diag = PipelineDiagnostics(symbol=symbol, period=period)
    brain_diag.bars_loaded_5m = len(df_5m)
    brain_diag.bars_loaded_15m = len(df_15m)
    brain_diag.trading_days_found = len(dates)
    brain_regime_counts: dict[str, int] = {"BULL_OPEN": 0, "BEAR_OPEN": 0, "CHOPPY": 0, "HIGH_VOL": 0, "NEWS_RISK": 0}

    for date in dates:
        day_5m = df_5m[df_5m.index.date == date]
        day_15m = df_15m[df_15m.index.date == date] if not df_15m.empty else pd.DataFrame()
        day_spy = spy_df[spy_df.index.date == date] if not spy_df.empty else pd.DataFrame()

        if len(day_5m) < 4:
            brain_diag.trading_days_skipped_short += 1
            continue

        # ── Full brain classifier (replaces simplified open/close heuristic) ──
        # Pass the full history up-to-and-including this date so the classifier's
        # rolling 20-bar ATR average has proper cross-day context. _today_bars()
        # inside classify_market_state will extract only today's bars for signals.
        hist_5m = df_5m[df_5m.index.date <= date]
        hist_spy = spy_df[spy_df.index.date <= date] if not spy_df.empty else pd.DataFrame()
        try:
            from app.services.strategy.daytrading.brain.market_state import (
                classify_market_state, TREND_UP, TREND_DOWN, HIGH_VOL, NEWS_RISK,
            )
            from app.services.strategy.daytrading.brain.strategy_router import route_strategies
            ms = classify_market_state(hist_5m, hist_spy if not hist_spy.empty else None)
            routing = route_strategies(ms)
            local_brain._last_market_state = ms
            local_brain._last_routing = routing

            # Map brain state -> legacy regime string for strategy.generate_signals compat
            if ms.state == TREND_UP:
                regime = "BULL_OPEN"
            elif ms.state == TREND_DOWN:
                regime = "BEAR_OPEN"
            elif ms.state in (HIGH_VOL, NEWS_RISK):
                # High-vol / news: skip new entries (too risky)
                brain_regime_counts[ms.state] = brain_regime_counts.get(ms.state, 0) + 1
                brain_diag.days_skipped_by_regime += 1
                continue
            else:
                regime = "CHOPPY"

            # Skip if brain confidence is too low to trade
            if ms.confidence < 0.25:
                regime = "CHOPPY"

            # brain_day_size_mult is intentionally 1.0 here — the routing size multiplier
            # (0.75 CHOPPY, 0.5 HIGH_VOL, etc.) is already baked into decisions[0].size_multiplier
            # by the router+governor pipeline. Applying a second table would double-penalise.
            brain_day_size_mult = 1.0

        except Exception:
            ms = None
            regime = "CHOPPY"
            brain_day_size_mult = 0.7

        brain_regime_counts[regime] = brain_regime_counts.get(regime, 0) + 1

        if not regime_allows_strategy(regime, strategy_name):
            brain_diag.days_skipped_by_regime += 1
            continue

        try:
            signals = strategy.generate_signals(day_5m, day_15m, symbol, None, regime)
        except Exception:
            continue

        day_raw = [s for s in signals if s.direction != "HOLD"]
        brain_diag.raw_signals_generated += len(day_raw)
        brain_diag.raw_buy_signals  += sum(1 for s in day_raw if s.direction == "BUY")
        brain_diag.raw_sell_signals += sum(1 for s in day_raw if s.direction in ("SELL", "SELL_SHORT"))
        brain_diag.raw_hold_signals += sum(1 for s in signals if s.direction == "HOLD")

        for sig in signals:
            if sig.direction == "HOLD":
                continue

            sig_dict = asdict(sig)
            today_trades_only = [t for t in filtered_trades if t.get("date") == str(date)]
            # For consecutive-loss kill switch: include recent prior trades so it
            # persists across day boundaries (governor resets to today-only otherwise).
            recent_prior = [t for t in filtered_trades if t.get("date") != str(date)][-6:]
            governor_trades = recent_prior + today_trades_only
            decisions = local_brain.filter_signals(
                [sig_dict],
                account_state={
                    "today_trades": governor_trades,
                    "initial_capital": initial_capital,
                    "open_positions": 0,
                },
                current_bar_time=pd.Timestamp(sig.signal_time) if sig.signal_time else None,
            )
            if not decisions or not decisions[0].accepted:
                # Track brain rejections
                if decisions:
                    reason_str = decisions[0].rejection_reason or decisions[0].explanation or ""
                    cat = _categorise_rejection(reason_str)
                    if cat == "regime":          brain_diag.rejected_by_regime      += 1
                    elif cat == "volume":        brain_diag.rejected_by_volume       += 1
                    elif cat == "rr":            brain_diag.rejected_by_rr           += 1
                    elif cat == "time":          brain_diag.rejected_by_time         += 1
                    elif cat == "extension":     brain_diag.rejected_by_extension    += 1
                    elif cat == "kill_switch":   brain_diag.rejected_by_kill_switch  += 1
                    else:                        brain_diag.rejected_other           += 1
                    brain_diag.rejected_by_brain_total += 1
                    if len(brain_diag.rejection_reasons) < 5:
                        brain_diag.rejection_reasons.append(f"[{cat}] {reason_str[:80]}")
                continue

            size_mult = decisions[0].size_multiplier * brain_day_size_mult

            entry = sig.entry_price
            stop = sig.stop_price
            target = sig.target_price

            sig_time = pd.Timestamp(sig.signal_time)
            future_bars = day_5m[day_5m.index > sig_time]

            if future_bars.empty:
                brain_diag.trades_skipped_no_future_bars += 1
                continue

            # Bucket-aware risk size, then apply brain size_mult, then cap by notional.
            market = "NSE" if _is_india_for_sizing(symbol) else "US"
            bucket = get_symbol_bucket(symbol, market=market)
            base_shares = _risk_based_size(
                equity=equity, bucket=bucket,
                entry=entry, stop=stop, notional_cap_pct=position_pct,
            )
            position_size = int(base_shares * size_mult) if base_shares > 0 else 0
            if position_size <= 0:
                brain_diag.trades_skipped_no_future_bars += 1
                continue

            brain_diag.trades_opened += 1
            brain_diag.accepted_signals += 1

            sim = simulate_exit(
                direction=sig.direction,
                entry_price=entry, initial_stop=stop, initial_target=target,
                qty=float(position_size), signal_time=sig_time,
                future_bars=future_bars, exit_plan=sig.exit_plan, symbol=symbol,
                max_hold_bars_default=strategy.default_config.get("max_hold_bars", 60),
            )
            agg = aggregate_pnl(sig.direction, entry, sim)
            exit_price = agg["exit_price"] if agg["qty"] > 0 else entry
            exit_time  = sim.legs[-1].time if sim.legs else sig_time
            outcome    = sim.primary_outcome
            hold_bars  = sim.hold_bars

            brain_diag.trades_closed += 1
            vol_pct = profile.volatility_pct if profile else 1.0
            entry_fill = fill_sim.fill_entry(sig.direction, entry, position_size, vol_pct)
            exit_fill  = fill_sim.fill_exit(sig.direction, exit_price, position_size, vol_pct)
            fill_summary = fill_sim.summarize_trade(
                symbol, sig.direction, position_size, entry_fill, exit_fill
            )

            pnl = fill_summary.net_pnl
            gross_pnl = fill_summary.gross_pnl
            pnl_pct = pnl / (entry * position_size) * 100 if entry * position_size > 0 else 0.0
            brain_total_commission += fill_summary.total_commission
            brain_total_slippage   += fill_summary.total_slippage
            equity += pnl

            filtered_trades.append({
                "date": str(date),
                "symbol": symbol,
                "strategy": strategy_name,
                "direction": sig.direction,
                "entry_price": round(entry_fill.fill_price, 4),
                "exit_price": round(exit_fill.fill_price, 4),
                "stop_price": round(stop, 4),
                "target_price": round(target, 4),
                "entry_time": _ts_str(sig_time, _mkt_tz),
                "exit_time": _ts_str(exit_time, _mkt_tz),
                "hold_bars": hold_bars,
                "pnl": round(pnl, 2),
                "gross_pnl": round(gross_pnl, 2),
                "commission": round(fill_summary.total_commission, 4),
                "slippage": round(fill_summary.total_slippage, 4),
                "pnl_pct": round(pnl_pct, 4),
                "outcome": outcome,
                "regime": regime,
                "confidence": sig.confidence,
                "brain_size_multiplier": size_mult,
                "market_state": ms.state if ms else "UNKNOWN",
                "market_state_confidence": round(ms.confidence, 3) if ms else 0.0,
                # ExitPlan / sizing diagnostics
                "bucket": bucket,
                "position_size": float(position_size),
                "final_stop": round(sim.final_stop, 4),
                "breakeven_hit": sim.breakeven_hit,
                "trail_activated": sim.trail_activated,
                "exit_legs": [
                    {"qty": leg.qty, "price": round(leg.price, 4),
                     "reason": leg.reason, "time": _ts_str(leg.time, _mkt_tz)}
                    for leg in sim.legs
                ],
            })

    brain_result = _compute_metrics(filtered_trades, initial_capital, equity, symbol, strategy_name,
                                    total_commission=brain_total_commission, total_slippage=brain_total_slippage)

    # Finalise brain diagnostics and attach to result
    brain_diag.regime_distribution = brain_regime_counts
    brain_diag.strategies_run = [strategy_name]
    brain_diag.finalise()
    brain_result["diagnostics"] = brain_diag.to_dict()

    # Combine both into a comparison dict
    raw_m = raw.get("metrics", {})
    brain_m = brain_result.get("metrics", {})
    comparison = {}
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl", "max_drawdown_pct", "sharpe_ratio"]:
        comparison[key] = {
            "raw": raw_m.get(key, 0),
            "brain": brain_m.get(key, 0),
        }

    # Raw vs Simulated P&L table (fill simulation impact)
    raw_net = raw_m.get("net_pnl", raw_m.get("total_pnl", 0))
    brain_net = brain_m.get("net_pnl", brain_m.get("total_pnl", 0))
    fill_impact = {
        "raw_net_pnl": raw_net,
        "brain_net_pnl": brain_net,
        "brain_total_commission": round(brain_total_commission, 2),
        "brain_total_slippage": round(brain_total_slippage, 2),
        "brain_gross_pnl": brain_m.get("gross_pnl", brain_net),
        "pnl_reduction_pct": round(
            (brain_m.get("gross_pnl", brain_net) - brain_net) / abs(brain_m.get("gross_pnl", brain_net)) * 100, 1
        ) if brain_m.get("gross_pnl", brain_net) != 0 else 0.0,
    }

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "raw": raw,
        "brain_filtered": brain_result,
        "comparison": comparison,
        "fill_impact": fill_impact,
    }


def run_scan(
    symbols: list[str],
    enabled_strategies: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Scan multiple symbols for current signals."""
    all_signals: list[dict] = []
    for symbol in symbols:
        _time.sleep(0.5)
        try:
            result = run_signals(symbol, enabled_strategies)
            all_signals.extend(result.get("signals", []))
        except Exception:
            continue
    all_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return all_signals


# Static fallback used when the scanner cannot run (import error, no data, etc.)
_FALLBACK_SYMBOLS: list[str] = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]


def get_session_symbols(max_symbols: int = 20) -> list[str]:
    """
    Use the DayTradingScanner to choose today's watchlist.

    The scanner ranks symbols by liquidity, volatility, and pre-market activity,
    then the brain adjusts scores based on the current market state.

    Falls back gracefully to _FALLBACK_SYMBOLS if the scanner is unavailable
    for any reason (import error, no data, pre-market before data is ready).
    This keeps all existing callers working unchanged.
    """
    try:
        from app.services.strategy.daytrading.scanners import DayTradingScanner, DayTradingScannerConfig
        scanner = DayTradingScanner(
            config=DayTradingScannerConfig(),
            brain=_brain,
        )
        watchlist = scanner.get_intraday_watchlist(max_symbols=max_symbols)
        symbols = [r.symbol for r in watchlist]
        if symbols:
            return symbols
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning("Scanner unavailable, using fallback symbols: %s", e)

    return list(_FALLBACK_SYMBOLS)


def run_simulation_backtest(
    symbol: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
    direction_mode: str = "long_only",
    trail_mode: str = "atr",
    partial_tp: bool = True,
    risk_per_trade_pct: float = 0.01,
    max_daily_loss_pct: float = 2.0,
    max_trades_per_day: int = 6,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """
    Replay the exact auto-trader logic bar-by-bar on historical data.

    Entry:  strategy.generate_signals() (same as auto-trader native mode) —
            filtered by direction_mode, confidence, R:R.
    Exit:   ExitManager — stop hits, trailing, momentum fade, EOD flatten.
    Stops:  PositionManager — breakeven at +1R, partial TP, trail activation.
    Risk:   RiskGovernor — daily loss cap, max trades/day, cooldown bars.
    Regime: classify_market_state() once per day (same brain as live).

    strategy_name: if given, only signals from that strategy are considered.
    """
    from app.services.strategy.daytrading.autotrader.exit_manager import ExitManager
    from app.services.strategy.daytrading.autotrader.position_manager import PositionManager
    from app.services.strategy.daytrading.autotrader.trade_state import TradeStateMachine, State
    from app.services.strategy.daytrading.brain.risk_governor import RiskGovernor
    from app.services.strategy.daytrading.brain.market_state import (
        classify_market_state, MarketStateResult,
    )
    from app.services.markets import is_india_symbol
    from datetime import datetime as _dt

    _is_india = is_india_symbol(symbol)
    _sess     = market_session(symbol)

    # ── Fetch history ─────────────────────────────────────────────────────────
    df_5m  = fetch_intraday(symbol, interval="5m",  period=period)
    df_15m = fetch_intraday(symbol, interval="15m", period=period)
    if df_5m.empty:
        return {"error": f"No 5m data for {symbol} over {period}"}

    # SPY / Nifty for market regime (best-effort, graceful fallback)
    try:
        _ref_sym = "^NSEI" if _is_india else "SPY"
        df_ref = fetch_intraday(_ref_sym, interval="5m", period=period)
    except Exception:
        df_ref = pd.DataFrame()

    # Decide which strategies to use for entry signals
    _entry_strategies = (
        [STRATEGY_MAP[strategy_name]] if strategy_name and strategy_name in STRATEGY_MAP
        else list(STRATEGY_MAP.values())
    )

    # ── Risk governor config ──────────────────────────────────────────────────
    risk_governor = RiskGovernor(config={
        "max_daily_loss_pct":    max_daily_loss_pct,
        "max_trades_per_day":    max_trades_per_day,
        "max_consecutive_losses": 3,
        "max_open_positions":    1,
        "size_reduction_after_loss": 0.5,
        "size_reset_after_win":  True,
    })
    exit_manager     = ExitManager(symbol=symbol)
    position_manager = PositionManager(trail_mode=trail_mode, partial_tp=partial_tp)
    tsm              = TradeStateMachine()

    all_trades: list[dict] = []
    equity       = initial_capital
    equity_curve: list[float] = [equity]

    _COOLDOWN_LOSS = 2
    _COOLDOWN_WIN  = 1
    _can_short = direction_mode in ("short_only", "both")
    _can_long  = direction_mode in ("long_only",  "both")

    trading_dates = sorted({ts.date() for ts in df_5m.index})
    logger.info("sim_backtest %s: %d bars, %d days, strategy=%s",
                symbol, len(df_5m), len(trading_dates), strategy_name or "all")

    for day in trading_dates:
        day_5m  = df_5m[df_5m.index.date  == day]
        day_15m = df_15m[df_15m.index.date == day] if not df_15m.empty else pd.DataFrame()
        if len(day_5m) < 15:
            continue

        # Classify regime once per day using all bars up to this date
        hist_5m = df_5m[df_5m.index.date <= day]
        hist_ref = df_ref[df_ref.index.date <= day] if not df_ref.empty else pd.DataFrame()
        try:
            ms_result = classify_market_state(
                hist_5m.tail(200),
                hist_ref.tail(200) if not hist_ref.empty else None,
            )
            ms_str = ms_result.state
        except Exception:
            ms_str = "UNKNOWN"
            ms_result = MarketStateResult(state=ms_str, confidence=0.5, reasons=[])

        # Generate ALL entry signals for the day from strategy(ies)
        # (same call the auto-trader makes via NativeStrategyEntry)
        day_signals: list[dict] = []
        for strat in _entry_strategies:
            try:
                sigs = strat.generate_signals(
                    symbol=symbol,
                    df_5m=df_5m[df_5m.index.date <= day],
                    df_15m=df_15m[df_15m.index.date <= day] if not df_15m.empty else pd.DataFrame(),
                    regime=ms_str,
                    config=strat.default_config,
                )
                for s in (sigs or []):
                    if not s.get("signal_time"):
                        continue
                    direction = str(s.get("direction", "")).upper()
                    if direction == "BUY" and not _can_long:
                        continue
                    if direction == "SELL" and not _can_short:
                        continue
                    day_signals.append(s)
            except Exception:
                pass

        # Index signals by bar timestamp for O(1) lookup
        sig_by_bar: dict = {}
        for s in day_signals:
            try:
                ts = pd.Timestamp(s["signal_time"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize(_sess.tz)
                sig_by_bar.setdefault(ts, []).append(s)
            except Exception:
                pass

        # Reset per-day state
        day_trades: list[dict] = []
        bars_since_exit = 0
        last_exit_loss  = False
        if tsm.has_position:
            tsm.reset_after_exit()
        exit_manager.reset()
        position_manager.reset()

        # Walk bar by bar
        for bar_ts in day_5m.index:
            curr_close = float(day_5m.loc[bar_ts, "Close"])
            bar_time   = bar_ts.time() if hasattr(bar_ts, "time") else bar_ts.to_pydatetime().time()

            # EOD force-flatten
            if bar_time >= _sess.close_time and tsm.has_position:
                pnl = _sim_close(tsm, curr_close, "EOD flatten", bar_ts, initial_capital, day_trades)
                equity += pnl; equity_curve.append(equity)
                bars_since_exit = 0; last_exit_loss = pnl < 0
                exit_manager.reset(); position_manager.reset(); tsm.reset_after_exit()
                break

            if tsm.state == State.FLAT:
                # Cooldown
                cooldown = _COOLDOWN_LOSS if last_exit_loss else _COOLDOWN_WIN
                if bars_since_exit < cooldown:
                    bars_since_exit += 1
                    continue

                # Risk governor
                rs  = risk_governor.build_risk_state(day_trades, initial_capital, 0)
                gov = risk_governor.check_can_trade(rs)
                if not gov.allowed:
                    break

                # Use signal if one fires on this bar
                bar_sigs = sig_by_bar.get(bar_ts, [])
                if not bar_sigs:
                    continue

                # Pick highest-confidence signal
                sig = max(bar_sigs, key=lambda s: float(s.get("confidence", 0)))
                conf = float(sig.get("confidence", 0))
                rr   = float(sig.get("r_multiple", 0))
                if conf < 0.45 or rr < 1.5:
                    continue

                entry_px = float(sig.get("entry_price") or curr_close)
                stop_px  = float(sig.get("stop_price")  or 0)
                tgt_px   = float(sig.get("target_price") or 0)
                direction = str(sig.get("direction", "BUY")).upper()
                side      = "LONG" if direction == "BUY" else "SHORT"

                if stop_px <= 0:
                    continue
                stop_dist = abs(entry_px - stop_px)
                if stop_dist <= 0:
                    continue

                risk_dollar = equity * risk_per_trade_pct * gov.size_multiplier
                qty = max(1.0, round(risk_dollar / stop_dist, 0))

                # 0.05% slippage on fill
                fill = round(entry_px * (1.0005 if direction == "BUY" else 0.9995), 4)

                tsm.open_position(
                    symbol=symbol, side=side,
                    entry_price=fill, qty=qty,
                    stop=stop_px, target=tgt_px,
                    strategy=str(sig.get("strategy", "")),
                    entry_reason=str(sig.get("reason", "")),
                    exit_plan=sig.get("exit_plan"),
                )
                # Stamp entry time as the bar timestamp (not now())
                tsm.entry_time = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else _dt.now()
                position_manager.reset(); exit_manager.reset()
                bars_since_exit = 0

            elif tsm.has_position:
                bars_since_exit = 0
                hist_now_5m  = df_5m[df_5m.index  <= bar_ts]
                hist_now_15m = df_15m[df_15m.index <= bar_ts] if not df_15m.empty else pd.DataFrame()

                # Position manager: stop moves, partial TPs, trail activation
                pm = position_manager.evaluate(tsm, hist_now_5m, None, ms_str)
                if pm and pm.action == "MOVE_STOP" and pm.new_stop:
                    tsm.current_stop = pm.new_stop
                elif pm and pm.action == "ACTIVATE_TRAIL":
                    if hasattr(tsm, "state"):
                        from app.services.strategy.daytrading.autotrader.trade_state import State as _State
                        tsm.transition(_State.TRAILING, pm.reason)

                # Exit manager: stop hit, trail hit, EOD, momentum fade
                em = exit_manager.evaluate(tsm, hist_now_5m, None, ms_str)
                if em.action == "FULL_EXIT":
                    exit_px = em.exit_price or curr_close
                    pnl = _sim_close(tsm, exit_px, em.reason, bar_ts, initial_capital, day_trades)
                    equity += pnl; equity_curve.append(equity)
                    last_exit_loss = pnl < 0; bars_since_exit = 0
                    exit_manager.reset(); position_manager.reset(); tsm.reset_after_exit()
                elif em.action == "MOVE_STOP" and em.new_stop:
                    tsm.current_stop = em.new_stop
                elif em.action == "PARTIAL_EXIT":
                    # Scale out: reduce qty, keep position
                    scale_qty = round(tsm.qty * 0.5, 0)
                    if scale_qty >= 1:
                        tsm.qty = max(1.0, tsm.qty - scale_qty)

        all_trades.extend(day_trades)

    # ── Build output ──────────────────────────────────────────────────────────
    return _build_sim_metrics(all_trades, equity_curve, initial_capital, symbol, period, strategy_name)


def _sim_close(tsm, exit_price: float, reason: str, bar_ts,
               initial_capital: float, day_trades: list) -> float:
    """Close the TSM position, record trade dict, return P&L."""
    from datetime import datetime as _dt
    entry = tsm.entry_price
    qty   = tsm.qty
    side  = tsm.side
    slip  = exit_price * 0.0005 * qty
    gross = (exit_price - entry) * qty if side == "LONG" else (entry - exit_price) * qty
    pnl   = gross - slip

    exit_dt = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else _dt.now()
    rec = tsm.close_position(exit_price=exit_price, exit_time=exit_dt, exit_reason=reason)

    day_trades.append({
        "date":        str(exit_dt.date()),
        "direction":   side,
        "strategy":    tsm.strategy or "",
        "entry_price": round(entry, 4),
        "exit_price":  round(exit_price, 4),
        "entry_time":  tsm.entry_time.isoformat() if tsm.entry_time else "",
        "exit_time":   exit_dt.isoformat(),
        "hold_bars":   getattr(rec, "hold_bars", 0) if rec else 0,
        "qty":         qty,
        "pnl":         round(pnl, 2),
        "pnl_pct":     round(pnl / initial_capital * 100, 4),
        "outcome":     reason[:50],
        "regime":      "",
        "confidence":  0.0,
    })
    return pnl


def _build_sim_metrics(
    trades: list[dict], equity_curve: list[float],
    initial_capital: float, symbol: str, period: str,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    import numpy as np
    if not trades:
        return {
            "metrics": {
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "net_pnl": 0, "total_return_pct": 0,
                "max_drawdown_pct": 0, "sharpe_ratio": 0,
                "avg_hold_bars": 0, "avg_pnl_per_trade": 0,
            },
            "trades": [], "equity_curve": equity_curve,
            "simulation_mode": True, "strategy": strategy_name or "all",
        }

    pnls      = [t["pnl"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    peak = equity_curve[0]; max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100 if peak > 0 else 0)

    pnl_arr = np.array(pnls)
    sharpe = (float(pnl_arr.mean() / pnl_arr.std()) * (252 ** 0.5)
              if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0.0)

    return {
        "metrics": {
            "total_trades":      len(trades),
            "win_rate":          round(len(wins) / len(pnls) * 100, 1),
            "profit_factor":     round(gross_win / gross_loss, 2) if gross_loss > 0 else 0,
            "total_pnl":         round(total_pnl, 2),
            "net_pnl":           round(total_pnl, 2),
            "gross_pnl":         round(gross_win, 2),
            "total_return_pct":  round(total_pnl / initial_capital * 100, 2),
            "max_drawdown_pct":  round(max_dd, 2),
            "sharpe_ratio":      round(sharpe, 2),
            "avg_hold_bars":     round(sum(t.get("hold_bars", 0) for t in trades) / len(trades), 1),
            "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
        },
        "trades":        trades,
        "equity_curve":  equity_curve,
        "simulation_mode": True,
        "symbol":        symbol,
        "period":        period,
        "strategy":      strategy_name or "all",
    }
