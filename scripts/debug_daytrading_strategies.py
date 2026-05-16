"""
scripts/debug_daytrading_strategies.py

Diagnostic tool — runs every strategy on a set of symbols and prints a
structured breakdown of exactly where the pipeline succeeds or fails.

Usage:
    python scripts/debug_daytrading_strategies.py
    python scripts/debug_daytrading_strategies.py --symbols NVDA TSLA --period 60d
    python scripts/debug_daytrading_strategies.py --strategy ORBBreakout --verbose
"""
from __future__ import annotations

import argparse
import sys
import os
import logging
from dataclasses import dataclass, field
from datetime import date

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from app.services.strategy.daytrading.runner import fetch_intraday
from app.services.strategy.daytrading.market_open import (
    ET, compute_vwap, regime_allows_strategy, validate_vwap_resets,
)
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP
from app.services.strategy.daytrading.pipeline_diagnostics import PipelineDiagnostics
from app.services.strategy.daytrading.brain.symbol_profiles import SymbolAnalyzer
from app.services.strategy.daytrading.brain.config_adjuster import ConfigAdjuster

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS  = ["NVDA", "TSLA", "AAPL", "SPY"]
DEFAULT_PERIODS  = ["30d", "60d"]
DEFAULT_STRATS   = [s.name for s in ALL_STRATEGIES]

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


@dataclass
class StratResult:
    symbol: str
    strategy: str
    period: str
    bars_5m: int = 0
    trading_days: int = 0
    days_skipped_short: int = 0
    days_skipped_regime: int = 0
    regime_dist: dict = field(default_factory=dict)
    raw_signals: int = 0
    raw_buy: int = 0
    raw_sell: int = 0
    hold_filtered: int = 0
    trades_executed: int = 0
    trades_skipped_no_bars: int = 0
    avg_hold_bars: float = 0.0
    top_rejection: str = ""
    error: str = ""


def run_one(symbol: str, strategy_name: str, period: str, verbose: bool = False) -> StratResult:
    r = StratResult(symbol=symbol, strategy=strategy_name, period=period)

    strategy = STRATEGY_MAP.get(strategy_name)
    if strategy is None:
        r.error = f"Unknown strategy: {strategy_name}"
        return r

    diag = PipelineDiagnostics(symbol=symbol, period=period)

    try:
        df_5m  = fetch_intraday(symbol, "5m",  period,  diag=diag)
        df_15m = fetch_intraday(symbol, "15m", "730d" if period in ("730d", "180d") else period, diag=diag)
    except Exception as e:
        r.error = f"Data fetch failed: {e}"
        return r

    if df_5m.empty:
        r.error = "No 5m data returned"
        return r

    r.bars_5m = len(df_5m)

    # ── Auto-tune config (matches run_backtest behaviour) ─────────────────────
    try:
        profile = SymbolAnalyzer.analyze(df_5m, symbol)
        adj = ConfigAdjuster.adjust(strategy_name, profile)
        effective_config = adj.adjusted if adj.has_changes else None
        if adj.has_changes and verbose:
            print(f"    {_c('CONFIG_TUNED', _CYAN)}: {'; '.join(adj.changes[:3])}")
    except Exception:
        effective_config = None

    # ── VWAP sanity check ─────────────────────────────────────────────────────
    vwap_check = validate_vwap_resets(df_5m)
    if not vwap_check["ok"] and verbose:
        print(f"    {_c('VWAP WARNING', _YELLOW)}: {vwap_check['issues']}")

    dates_all = sorted(set(df_5m.index.date))
    r.trading_days = len(dates_all)

    regime_counts: dict[str, int] = {}
    hold_bars_list: list[int] = []

    for date_ in dates_all:
        day_5m  = df_5m[df_5m.index.date == date_]
        day_15m = df_15m[df_15m.index.date == date_] if not df_15m.empty else pd.DataFrame()

        if len(day_5m) < 4:
            r.days_skipped_short += 1
            if verbose:
                print(f"    {_c('SKIP', _YELLOW)} {date_}: only {len(day_5m)} bars")
            continue

        # Regime detection (same as run_backtest)
        prev_days = df_5m[df_5m.index.date < date_]
        if prev_days.empty:
            regime = "CHOPPY"
        else:
            prior_close = float(prev_days["Close"].iloc[-1])
            open_p      = float(day_5m["Open"].iloc[0])
            current     = float(day_5m["Close"].iloc[-1])
            vwap_series = compute_vwap(day_5m)
            vwap_val    = float(vwap_series.iloc[-1])
            if open_p > prior_close and current > vwap_val:
                regime = "BULL_OPEN"
            elif open_p < prior_close and current < vwap_val:
                regime = "BEAR_OPEN"
            else:
                regime = "CHOPPY"

        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if not regime_allows_strategy(regime, strategy_name):
            r.days_skipped_regime += 1
            if verbose:
                print(f"    {_c('REGIME_SKIP', _YELLOW)} {date_}: {regime} blocks {strategy_name}")
            continue

        # ── Run strategy ──────────────────────────────────────────────────────
        try:
            signals = strategy.generate_signals(day_5m, day_15m, symbol, effective_config, regime)
        except Exception as e:
            if verbose:
                print(f"    {_c('ERROR', _RED)} {date_}: strategy raised {e}")
            continue

        for sig in signals:
            if sig.direction == "HOLD":
                r.hold_filtered += 1
                continue

            r.raw_signals += 1
            if sig.direction == "BUY":
                r.raw_buy += 1
            else:
                r.raw_sell += 1

            if verbose:
                print(
                    f"    {_c('SIGNAL', _CYAN)} {date_} {sig.direction} @ {sig.entry_price:.2f} "
                    f"stop={sig.stop_price:.2f} target={sig.target_price:.2f} "
                    f"conf={sig.confidence:.2f} regime={regime}"
                )

            # ── Simulate execution ────────────────────────────────────────────
            sig_time     = pd.Timestamp(sig.signal_time)
            future_bars  = day_5m[day_5m.index > sig_time]

            if future_bars.empty:
                r.trades_skipped_no_bars += 1
                if verbose:
                    print(f"      {_c('NO_BARS', _YELLOW)}: no future bars after {sig_time.time()}")
                continue

            r.trades_executed += 1
            entry   = sig.entry_price
            stop    = sig.stop_price
            target  = sig.target_price
            hold    = 0
            outcome = "OPEN"

            for _, bar in future_bars.iterrows():
                hold += 1
                hi, lo = float(bar["High"]), float(bar["Low"])
                if sig.direction == "BUY":
                    if lo  <= stop:   outcome = "STOPPED"; break
                    if hi  >= target: outcome = "TARGET";  break
                else:
                    if hi  >= stop:   outcome = "STOPPED"; break
                    if lo  <= target: outcome = "TARGET";  break
                if hold >= strategy.default_config.get("max_hold_bars", 60):
                    outcome = "TIME_EXIT"; break

            if outcome == "OPEN":
                outcome = "EOD_EXIT"

            hold_bars_list.append(hold)

            if verbose:
                print(f"      {_c(outcome, _GREEN if outcome == 'TARGET' else _RED)} hold={hold}bars")

    r.regime_dist      = regime_counts
    r.avg_hold_bars    = sum(hold_bars_list) / len(hold_bars_list) if hold_bars_list else 0.0

    # Infer top rejection reason for zero-trade cases
    if r.raw_signals == 0 and r.trading_days > 0:
        usable = r.trading_days - r.days_skipped_short - r.days_skipped_regime
        if usable == 0:
            r.top_rejection = "all_days_blocked_by_regime_or_too_short"
        else:
            r.top_rejection = "strategy_conditions_never_met"

    return r


def print_summary_table(results: list[StratResult]) -> None:
    col_w = [8, 24, 6, 5, 5, 5, 5, 8, 6, 6, 8, 28]
    hdr = [
        "Symbol", "Strategy", "Period",
        "Bars", "Days", "Skip\nShrt", "Skip\nReg",
        "Raw\nSigs", "Buy", "Sell",
        "Trades", "Top Rejection / Note",
    ]

    sep  = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    def row(cells: list) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w)) + " |"

    print("\n" + sep)
    print(row(hdr))
    print(sep)

    for r in results:
        if r.error:
            note = _c(f"ERROR: {r.error}", _RED)
            trades_str = "ERR"
        elif r.raw_signals == 0 and r.top_rejection:
            note  = _c(r.top_rejection, _YELLOW)
            trades_str = "0"
        elif r.trades_executed == 0 and r.raw_signals > 0:
            note  = _c("signals found but no trades (no future bars?)", _YELLOW)
            trades_str = "0"
        else:
            note  = _c(f"avg hold {r.avg_hold_bars:.1f}bars", _GREEN)
            trades_str = _c(str(r.trades_executed), _GREEN)

        dist_short = "/".join(f"{k[0]}:{v}" for k, v in sorted(r.regime_dist.items()))

        print(row([
            r.symbol, r.strategy, r.period,
            r.bars_5m, r.trading_days,
            r.days_skipped_short, r.days_skipped_regime,
            r.raw_signals, r.raw_buy, r.raw_sell,
            trades_str, note,
        ]))

    print(sep + "\n")


def print_per_symbol_report(results: list[StratResult]) -> None:
    symbols = sorted(set(r.symbol for r in results))
    for sym in symbols:
        sym_results = [r for r in results if r.symbol == sym]
        print(_c(f"\n{'='*70}", _BOLD))
        print(_c(f"  {sym}", _BOLD))
        print(_c(f"{'='*70}", _BOLD))

        for r in sym_results:
            tag = _c(f"[{r.strategy} / {r.period}]", _CYAN)
            if r.error:
                print(f"  {tag}  {_c('ERROR: ' + r.error, _RED)}")
                continue

            raw_label = _c(str(r.raw_signals), _GREEN if r.raw_signals > 0 else _RED)
            trade_label = _c(str(r.trades_executed), _GREEN if r.trades_executed > 0 else _RED)

            print(f"  {tag}")
            print(f"    Data:     {r.bars_5m} bars, {r.trading_days} days")
            regime_str = "  ".join(f"{k}:{v}" for k, v in sorted(r.regime_dist.items()))
            print(f"    Regime:   {regime_str}")
            print(
                f"    Signals:  raw={raw_label}  "
                f"(buy={r.raw_buy} sell={r.raw_sell} hold_filtered={r.hold_filtered})"
            )
            print(
                f"    Skipped:  short_days={r.days_skipped_short}  "
                f"regime_blocked={r.days_skipped_regime}  "
                f"no_future_bars={r.trades_skipped_no_bars}"
            )
            print(f"    Trades:   {trade_label}  avg_hold={r.avg_hold_bars:.1f}bars")

            if r.raw_signals == 0:
                if r.days_skipped_regime == r.trading_days - r.days_skipped_short:
                    print(f"    {_c('DIAGNOSIS: All usable days blocked by regime filter', _RED)}")
                else:
                    print(f"    {_c('DIAGNOSIS: Strategy conditions never satisfied', _YELLOW)}")
            elif r.trades_executed == 0:
                print(f"    {_c('DIAGNOSIS: Signals found but execution failed (no future bars)', _YELLOW)}")
            else:
                print(f"    {_c('STATUS: OK', _GREEN)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-trading strategy diagnostic tool")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to test (default: NVDA TSLA AAPL SPY)")
    parser.add_argument("--periods", nargs="+", default=DEFAULT_PERIODS,
                        help="Periods (default: 30d 60d)")
    parser.add_argument("--strategy", default=None,
                        help="Run one strategy only (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-bar signal details")
    parser.add_argument("--log-level", default="WARNING",
                        help="Python log level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    strategies = [args.strategy] if args.strategy else DEFAULT_STRATS
    symbols    = [s.upper() for s in args.symbols]
    periods    = args.periods

    total = len(symbols) * len(strategies) * len(periods)
    print(_c(f"\nDayTrading Strategy Diagnostic", _BOLD))
    print(f"Symbols:    {symbols}")
    print(f"Strategies: {strategies}")
    print(f"Periods:    {periods}")
    print(f"Total runs: {total}")
    print()

    results: list[StratResult] = []

    for sym in symbols:
        for period in periods:
            for strat in strategies:
                label = f"{sym:6s} / {strat:28s} / {period}"
                print(f"  Running {label}…", end="", flush=True)
                r = run_one(sym, strat, period, verbose=args.verbose)
                results.append(r)

                if r.error:
                    print(_c(f"  ERROR: {r.error}", _RED))
                elif r.trades_executed > 0:
                    print(_c(f"  {r.trades_executed} trades", _GREEN))
                elif r.raw_signals > 0:
                    print(_c(f"  {r.raw_signals} raw signals, 0 trades", _YELLOW))
                else:
                    print(_c(f"  0 raw signals", _RED))

    print_summary_table(results)
    print_per_symbol_report(results)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_runs = len(results)
    ok         = sum(1 for r in results if r.trades_executed > 0)
    no_trades  = sum(1 for r in results if r.trades_executed == 0 and r.raw_signals > 0)
    no_signals = sum(1 for r in results if r.raw_signals == 0 and not r.error)
    errors     = sum(1 for r in results if r.error)

    print(_c(f"\nSummary", _BOLD))
    print(f"  {_c(str(ok), _GREEN)}/{total_runs} runs produced trades")
    print(f"  {_c(str(no_trades), _YELLOW)}/{total_runs} found signals but 0 trades")
    print(f"  {_c(str(no_signals), _RED)}/{total_runs} found 0 raw signals")
    print(f"  {_c(str(errors), _RED)}/{total_runs} errored\n")


if __name__ == "__main__":
    main()
