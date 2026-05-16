"""
scripts/debug_bb_supertrend_daytrading.py

Focused diagnostic tool for BollingerMomentum and SupertrendTrend strategies.
Prints detailed per-day breakdowns including squeeze detection, band widths,
and Supertrend direction flips.

Usage:
    python scripts/debug_bb_supertrend_daytrading.py
    python scripts/debug_bb_supertrend_daytrading.py --symbols NVDA TSLA --period 60d
    python scripts/debug_bb_supertrend_daytrading.py --strategy BollingerMomentum --verbose
"""
from __future__ import annotations

import argparse
import sys
import os
import logging
from dataclasses import dataclass, field

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from app.services.strategy.daytrading.runner import fetch_intraday
from app.services.strategy.daytrading.market_open import ET, compute_vwap, regime_allows_strategy
from app.services.strategy.daytrading.strategies import STRATEGY_MAP
from app.services.strategy.daytrading.brain.symbol_profiles import SymbolAnalyzer
from app.services.strategy.daytrading.brain.config_adjuster import ConfigAdjuster
from app.services.strategy.daytrading.strategies.bollinger_momentum import BollingerMomentum
from app.services.strategy.daytrading.strategies.supertrend_trend import _compute_supertrend

DEFAULT_SYMBOLS  = ["NVDA", "TSLA", "AAPL", "SPY"]
DEFAULT_PERIOD   = "60d"
NEW_STRATEGIES   = ["BollingerMomentum", "SupertrendTrend"]

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


@dataclass
class DetailedResult:
    symbol: str
    strategy: str
    period: str
    bars_5m: int = 0
    trading_days: int = 0
    days_skipped_regime: int = 0
    raw_signals: int = 0
    trades_executed: int = 0
    # BB-specific
    days_in_squeeze: int = 0
    avg_band_width: float = 0.0
    breakout_days: int = 0
    # ST-specific
    days_macro_bullish: int = 0
    days_macro_bearish: int = 0
    days_5m_aligned: int = 0
    pullback_days: int = 0
    error: str = ""


def _compute_regime(df_5m: pd.DataFrame, date_: object, prior_close: float | None) -> str:
    day_bars = df_5m[df_5m.index.date == date_]
    if prior_close is None:
        return "CHOPPY"
    open_p  = float(day_bars["Open"].iloc[0])
    current = float(day_bars["Close"].iloc[-1])
    vwap_v  = float(compute_vwap(day_bars).iloc[-1])
    if open_p > prior_close and current > vwap_v:
        return "BULL_OPEN"
    elif open_p < prior_close and current < vwap_v:
        return "BEAR_OPEN"
    return "CHOPPY"


def _run_bb_detailed(symbol: str, period: str, verbose: bool) -> DetailedResult:
    r = DetailedResult(symbol=symbol, strategy="BollingerMomentum", period=period)
    strat = STRATEGY_MAP.get("BollingerMomentum")
    if strat is None:
        r.error = "BollingerMomentum not in STRATEGY_MAP"
        return r

    try:
        df_5m  = fetch_intraday(symbol, "5m", period)
        df_15m = fetch_intraday(symbol, "15m", period)
    except Exception as e:
        r.error = str(e)
        return r

    if df_5m.empty:
        r.error = "No 5m data"
        return r

    r.bars_5m = len(df_5m)
    try:
        profile = SymbolAnalyzer.analyze(df_5m, symbol)
        adj = ConfigAdjuster.adjust("BollingerMomentum", profile)
        cfg = adj.adjusted if adj.has_changes else None
        if verbose and adj.has_changes:
            print(f"  {_c('CONFIG_TUNED', _CYAN)}: {'; '.join(adj.changes[:3])}")
    except Exception:
        cfg = None

    dates = sorted(set(df_5m.index.date))
    r.trading_days = len(dates)

    band_widths = []

    for date_ in dates:
        day = df_5m[df_5m.index.date == date_].copy()
        if len(day) < 4:
            continue

        prev_days = df_5m[df_5m.index.date < date_]
        prior_close = float(prev_days["Close"].iloc[-1]) if not prev_days.empty else None
        regime = _compute_regime(df_5m, date_, prior_close)

        if not regime_allows_strategy(regime, "BollingerMomentum"):
            r.days_skipped_regime += 1
            if verbose:
                print(f"  {_c('REGIME_SKIP', _YELLOW)} {date_}: {regime}")
            continue

        # Compute BB on this day's bars
        from app.services.strategy.daytrading.strategies.bollinger_momentum import BollingerMomentum as _BB
        inst = _BB()
        eff_cfg = cfg or dict(inst.default_config)

        import ta.volatility as tav
        if len(day) >= eff_cfg["bb_length"]:
            from ta.volatility import BollingerBands
            bb = BollingerBands(day["Close"], window=eff_cfg["bb_length"], window_dev=eff_cfg["bb_std"])
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            mid   = bb.bollinger_mavg()
            widths = ((upper - lower) / mid.replace(0, float("nan"))).dropna()
            if not widths.empty:
                band_widths.extend(widths.tolist())

            # Check if any bar was in a squeeze
            lookback = eff_cfg["contraction_lookback"]
            squeeze_found = False
            for i in range(lookback, len(day)):
                if i < len(widths):
                    w = widths.iloc[i] if i < len(widths) else float("nan")
                    hist = widths.iloc[max(0, i - lookback): i]
                    if not hist.empty and not pd.isna(w):
                        thresh = hist.quantile(eff_cfg["contraction_percentile"])
                        if w <= thresh:
                            squeeze_found = True
                            break
            if squeeze_found:
                r.days_in_squeeze += 1

        sigs = strat.generate_signals(day, df_15m[df_15m.index.date == date_] if not df_15m.empty else pd.DataFrame(), symbol, cfg, regime)
        if sigs:
            r.breakout_days += 1
            r.raw_signals += len(sigs)
            if verbose:
                for s in sigs:
                    print(f"  {_c('SIGNAL', _CYAN)} {date_} {s.direction} @ {s.entry_price:.2f} conf={s.confidence:.2f}")

    r.avg_band_width = sum(band_widths) / len(band_widths) if band_widths else 0.0
    r.trades_executed = r.raw_signals  # BB only generates 1 per day max
    return r


def _run_st_detailed(symbol: str, period: str, verbose: bool) -> DetailedResult:
    r = DetailedResult(symbol=symbol, strategy="SupertrendTrend", period=period)
    strat = STRATEGY_MAP.get("SupertrendTrend")
    if strat is None:
        r.error = "SupertrendTrend not in STRATEGY_MAP"
        return r

    try:
        df_5m  = fetch_intraday(symbol, "5m", period)
        df_15m = fetch_intraday(symbol, "15m", period)
    except Exception as e:
        r.error = str(e)
        return r

    if df_5m.empty:
        r.error = "No 5m data"
        return r

    r.bars_5m = len(df_5m)
    try:
        profile = SymbolAnalyzer.analyze(df_5m, symbol)
        adj = ConfigAdjuster.adjust("SupertrendTrend", profile)
        cfg = adj.adjusted if adj.has_changes else None
        if verbose and adj.has_changes:
            print(f"  {_c('CONFIG_TUNED', _CYAN)}: {'; '.join(adj.changes[:3])}")
    except Exception:
        cfg = None

    dates = sorted(set(df_5m.index.date))
    r.trading_days = len(dates)

    default_cfg = strat.default_config
    eff = cfg or dict(default_cfg)

    for date_ in dates:
        day_5m  = df_5m[df_5m.index.date == date_].copy()
        day_15m = df_15m[df_15m.index.date == date_].copy() if not df_15m.empty else pd.DataFrame()
        if len(day_5m) < 4:
            continue

        prev_days = df_5m[df_5m.index.date < date_]
        prior_close = float(prev_days["Close"].iloc[-1]) if not prev_days.empty else None
        regime = _compute_regime(df_5m, date_, prior_close)

        if not regime_allows_strategy(regime, "SupertrendTrend"):
            r.days_skipped_regime += 1
            continue

        # 15m macro direction
        if not day_15m.empty and len(day_15m) >= eff["st_length"] + 2:
            st15 = _compute_supertrend(day_15m, eff["st_length"], eff["st_multiplier"])
            if st15 is not None:
                last_dir = int(st15["direction"].iloc[-1])
                if last_dir == 1:
                    r.days_macro_bullish += 1
                else:
                    r.days_macro_bearish += 1
        else:
            if regime == "BULL_OPEN":
                r.days_macro_bullish += 1
            elif regime == "BEAR_OPEN":
                r.days_macro_bearish += 1

        # 5m ST alignment
        if len(day_5m) >= eff["st_length"] + 2:
            st5 = _compute_supertrend(day_5m, eff["st_length"], eff["st_multiplier"])
            if st5 is not None:
                last_dir5 = int(st5["direction"].iloc[-1])
                if last_dir5 == 1 and regime == "BULL_OPEN":
                    r.days_5m_aligned += 1
                elif last_dir5 == -1 and regime == "BEAR_OPEN":
                    r.days_5m_aligned += 1

        sigs = strat.generate_signals(day_5m, day_15m, symbol, cfg, regime)
        if sigs:
            r.pullback_days += 1
            r.raw_signals += len(sigs)
            if verbose:
                for s in sigs:
                    print(f"  {_c('SIGNAL', _CYAN)} {date_} {s.direction} @ {s.entry_price:.2f} conf={s.confidence:.2f} {s.reason[:80]}")

    r.trades_executed = r.raw_signals
    return r


def print_report(results: list[DetailedResult]) -> None:
    for r in results:
        tag = _c(f"[{r.symbol} / {r.strategy} / {r.period}]", _CYAN)
        print(f"\n{tag}")

        if r.error:
            print(f"  {_c('ERROR: ' + r.error, _RED)}")
            continue

        print(f"  Data:       {r.bars_5m} bars, {r.trading_days} trading days")
        print(f"  Regime:     {r.days_skipped_regime} days blocked by regime")

        if r.strategy == "BollingerMomentum":
            print(f"  BB Squeeze: {r.days_in_squeeze}/{r.trading_days - r.days_skipped_regime} usable days in squeeze")
            print(f"  Avg BW:     {r.avg_band_width:.4f} (band_width = (upper-lower)/mid)")
            print(f"  Breakouts:  {r.breakout_days} days with breakout signal")
        else:
            usable = r.trading_days - r.days_skipped_regime
            print(f"  15m macro:  {r.days_macro_bullish} bull / {r.days_macro_bearish} bear out of {usable} usable days")
            print(f"  5m aligned: {r.days_5m_aligned} days where 5m ST aligns with macro")
            print(f"  Pullbacks:  {r.pullback_days} days with valid pullback entry")

        sig_label = _c(str(r.raw_signals), _GREEN if r.raw_signals > 0 else _RED)
        print(f"  Signals:    {sig_label} raw signals")

        if r.raw_signals == 0:
            if r.strategy == "BollingerMomentum":
                if r.days_in_squeeze == 0:
                    print(f"  {_c('DIAGNOSIS: No squeeze days detected — BB never contracted', _RED)}")
                else:
                    print(f"  {_c('DIAGNOSIS: Squeeze found but breakout conditions never met (check vol/RSI thresholds)', _YELLOW)}")
            else:
                if r.days_5m_aligned == 0:
                    print(f"  {_c('DIAGNOSIS: 5m and 15m Supertrend never aligned on same day', _RED)}")
                else:
                    print(f"  {_c('DIAGNOSIS: ST aligned but pullback-to-ST/EMA condition never triggered', _YELLOW)}")
        else:
            print(f"  {_c('STATUS: OK', _GREEN)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BollingerMomentum + SupertrendTrend diagnostic")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--strategy", default=None, choices=NEW_STRATEGIES,
                        help="Run one strategy only (default: both)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    strategies = [args.strategy] if args.strategy else NEW_STRATEGIES
    symbols = [s.upper() for s in args.symbols]

    print(_c("\nBollingerMomentum + SupertrendTrend Diagnostic", _BOLD))
    print(f"Symbols:    {symbols}")
    print(f"Strategies: {strategies}")
    print(f"Period:     {args.period}")
    print()

    results: list[DetailedResult] = []

    for sym in symbols:
        for strat in strategies:
            label = f"{sym:6s} / {strat}"
            print(f"Running {label}…", end="", flush=True)
            if strat == "BollingerMomentum":
                r = _run_bb_detailed(sym, args.period, args.verbose)
            else:
                r = _run_st_detailed(sym, args.period, args.verbose)
            results.append(r)

            if r.error:
                print(_c(f"  ERROR: {r.error}", _RED))
            elif r.raw_signals > 0:
                print(_c(f"  {r.raw_signals} signals", _GREEN))
            else:
                print(_c("  0 signals", _RED))

    print(_c(f"\n{'='*60}", _BOLD))
    print_report(results)

    print(_c(f"\n{'='*60}\nSummary", _BOLD))
    for r in results:
        tag = f"{r.symbol}/{r.strategy}"
        if r.error:
            print(f"  {_c(tag, _RED)}: ERROR {r.error}")
        elif r.raw_signals > 0:
            print(f"  {_c(tag, _GREEN)}: {r.raw_signals} signals ({r.trading_days} days, {r.days_skipped_regime} regime-blocked)")
        else:
            print(f"  {_c(tag, _RED)}: 0 signals ({r.trading_days} days, {r.days_skipped_regime} regime-blocked)")


if __name__ == "__main__":
    main()
