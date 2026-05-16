"""
Walk-Forward Validation and Symbol Ranking — BollingerMomentum
==============================================================

Runs three IS/OOS window configurations per symbol and produces a
ranked report saved to output/bb_walkforward_<date>.csv and a summary
printed to stdout.

yfinance 5m data limit: 60 calendar days (~42 trading days).
Window configs map the requested train/test ratios to the available data:

  Config A  "3m/1m"  — IS=21td  OOS=10td  (≈3 weeks / 2 weeks)
  Config B  "6m/1m"  — IS=28td  OOS=10td  (≈6 weeks / 2 weeks)
  Config C  "12m/1m" — IS=33td  OOS= 9td  (≈8 weeks / 2 weeks)

Each config uses a non-overlapping OOS window so results are independent.
All data is downloaded once per symbol, then sliced — no redundant API calls.

Usage:
    python scripts/bb_walkforward_report.py
    python scripts/bb_walkforward_report.py --symbols NVDA TSLA AAPL SPY QQQ MSFT
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ENVIRONMENT", "development")

from app.services.strategy.daytrading.runner import run_backtest
from app.services.strategy.daytrading.validation.walkforward import (
    _parse_date,
    _recompute_from_trades,
    _cagr,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

STRATEGY = "BollingerMomentum"
INITIAL_CAPITAL = 10_000.0

# Default symbols to rank
DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "SPY", "QQQ", "MSFT", "AMD", "META"]

# ── Window definitions ─────────────────────────────────────────────────────────
# Each entry: (label, is_trading_days, oos_trading_days)
# yfinance 5m cap is ~42 usable trading days in a 60-calendar-day download.
# We use the most recent 42 trading days available and slice deterministically.
WINDOW_CONFIGS = [
    ("3m_train_1m_test",  21, 10),   # IS ≈ 1 month, OOS ≈ 2 weeks
    ("6m_train_1m_test",  28, 10),   # IS ≈ 6 weeks, OOS ≈ 2 weeks
    ("12m_train_1m_test", 33,  9),   # IS ≈ 8 weeks, OOS ≈ 9 days
]


# ── Data class for per-window metrics ─────────────────────────────────────────

@dataclass
class WindowMetrics:
    symbol: str
    config: str
    is_trading_days: int
    oos_trading_days: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    # IS
    is_trades: int
    is_win_rate: float
    is_avg_win: float
    is_avg_loss: float
    is_profit_factor: float
    is_net_pnl: float
    is_max_drawdown_pct: float
    is_avg_hold_bars: float
    # OOS
    oos_trades: int
    oos_win_rate: float
    oos_avg_win: float
    oos_avg_loss: float
    oos_profit_factor: float
    oos_net_pnl: float
    oos_max_drawdown_pct: float
    oos_avg_hold_bars: float
    # Efficiency
    wfe: float          # OOS_return_pct / IS_return_pct; 0 if IS <= 0
    wfe_label: str      # robust / marginal / overfit / unprofitable
    error: str = ""


@dataclass
class SymbolSummary:
    symbol: str
    windows: list[WindowMetrics] = field(default_factory=list)
    # Aggregated across all valid OOS windows
    total_oos_trades: int = 0
    avg_oos_win_rate: float = 0.0
    avg_oos_pf: float = 0.0
    avg_oos_net_pnl: float = 0.0
    avg_oos_max_dd: float = 0.0
    avg_oos_hold_bars: float = 0.0
    avg_wfe: float = 0.0
    pct_robust_windows: float = 0.0
    # Ranking scores
    consistency_score: float = 0.0   # 0–100
    robustness_label: str = "insufficient_data"
    regime_sensitive: bool = False
    recommendation: str = ""


# ── Core runner ───────────────────────────────────────────────────────────────

def _fetch_all_trades(symbol: str) -> list[dict]:
    """Download 60d of backtest trades for symbol, return raw trade list."""
    result = run_backtest(
        symbol=symbol,
        strategy_name=STRATEGY,
        period="60d",
        initial_capital=INITIAL_CAPITAL,
    )
    return result.get("trades", [])


def _get_trading_dates(trades: list[dict]) -> list[date]:
    """Return sorted unique trading dates from a trade list."""
    dates = set()
    for t in trades:
        d = _parse_date(t.get("date", ""))
        if d != date.min:
            dates.add(d)
    return sorted(dates)


def _slice_trades(trades: list[dict], start: date, end: date) -> list[dict]:
    return [t for t in trades if start <= _parse_date(t.get("date", "")) <= end]


_EMPTY_METRICS: dict = {
    "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
    "total_return_pct": 0.0, "profit_factor": 0.0,
    "max_drawdown_pct": 0.0, "initial_capital": INITIAL_CAPITAL,
    "final_equity": INITIAL_CAPITAL,
    "avg_win": 0.0, "avg_loss": 0.0, "avg_hold_bars": 0.0,
}


def _extended_metrics(trades: list[dict], initial_capital: float) -> dict:
    """_recompute_from_trades plus avg_win, avg_loss, avg_hold_bars."""
    if not trades:
        return dict(_EMPTY_METRICS)
    base = _recompute_from_trades(trades, initial_capital)
    m = base["metrics"]
    if "total_pnl" not in m:
        m["total_pnl"] = 0.0

    df = pd.DataFrame(trades)
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    m["avg_win"]        = round(float(wins["pnl"].mean()),  2) if not wins.empty   else 0.0
    m["avg_loss"]       = round(float(losses["pnl"].mean()), 2) if not losses.empty else 0.0
    m["avg_hold_bars"]  = round(float(df["hold_bars"].mean()), 1) if "hold_bars" in df.columns else 0.0
    return m


def _wfe_from_pnl(is_pnl: float, oos_pnl: float, is_trades: int, oos_trades: int) -> tuple[float, str]:
    """WFE based on net PnL ratio (simpler than CAGR for short windows)."""
    if is_trades == 0:
        return 0.0, "no_is_trades"
    if is_pnl <= 0:
        return 0.0, "unprofitable"
    if oos_trades == 0:
        return 0.0, "no_oos_trades"
    wfe = oos_pnl / is_pnl
    if wfe >= 0.6:
        label = "robust"
    elif wfe >= 0.3:
        label = "marginal"
    elif wfe >= 0.0:
        label = "overfit"
    else:
        label = "oos_loss"
    return round(wfe, 3), label


# ── Per-symbol walk-forward ───────────────────────────────────────────────────

def run_symbol_walkforward(symbol: str) -> SymbolSummary:
    summary = SymbolSummary(symbol=symbol)
    print(f"  Fetching 60d trades for {symbol}...", flush=True)

    try:
        all_trades = _fetch_all_trades(symbol)
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        summary.robustness_label = "error"
        return summary

    if not all_trades:
        print(f"  {symbol}: 0 trades in 60d — skipping")
        summary.robustness_label = "insufficient_data"
        return summary

    # Get the full set of available trading dates from the raw backtest run
    # (not just dates with trades — need all dates to build IS/OOS boundaries).
    # Fetch a broader date index from yfinance daily.
    import yfinance as yf
    daily = yf.download(symbol, period="60d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    all_dates = sorted(daily.index.date)

    if len(all_dates) < 20:
        print(f"  {symbol}: only {len(all_dates)} calendar dates — skipping")
        summary.robustness_label = "insufficient_data"
        return summary

    # Use the most recent N trading days available
    max_days = 42   # conservative 5m lookback limit
    usable_dates = all_dates[-max_days:] if len(all_dates) >= max_days else all_dates

    for cfg_label, is_td, oos_td in WINDOW_CONFIGS:
        total_td = is_td + oos_td
        if len(usable_dates) < total_td:
            print(f"    {symbol} {cfg_label}: only {len(usable_dates)} dates, need {total_td} — skip")
            continue

        # Take the most recent (is_td + oos_td) trading days
        window_dates = usable_dates[-total_td:]
        is_dates  = window_dates[:is_td]
        oos_dates = window_dates[is_td:]

        is_start,  is_end  = is_dates[0],  is_dates[-1]
        oos_start, oos_end = oos_dates[0], oos_dates[-1]

        is_trades  = _slice_trades(all_trades, is_start,  is_end)
        oos_trades = _slice_trades(all_trades, oos_start, oos_end)

        is_m  = _extended_metrics(is_trades,  INITIAL_CAPITAL)
        oos_m = _extended_metrics(oos_trades, INITIAL_CAPITAL)

        wfe, wfe_label = _wfe_from_pnl(
            is_m.get("total_pnl", 0),  oos_m.get("total_pnl", 0),
            is_m.get("total_trades", 0), oos_m.get("total_trades", 0),
        )

        wm = WindowMetrics(
            symbol=symbol,
            config=cfg_label,
            is_trading_days=is_td,
            oos_trading_days=oos_td,
            is_start=str(is_start),
            is_end=str(is_end),
            oos_start=str(oos_start),
            oos_end=str(oos_end),
            is_trades=is_m["total_trades"],
            is_win_rate=is_m["win_rate"],
            is_avg_win=is_m["avg_win"],
            is_avg_loss=is_m["avg_loss"],
            is_profit_factor=is_m["profit_factor"],
            is_net_pnl=is_m["total_pnl"],
            is_max_drawdown_pct=is_m["max_drawdown_pct"],
            is_avg_hold_bars=is_m["avg_hold_bars"],
            oos_trades=oos_m["total_trades"],
            oos_win_rate=oos_m["win_rate"],
            oos_avg_win=oos_m["avg_win"],
            oos_avg_loss=oos_m["avg_loss"],
            oos_profit_factor=oos_m["profit_factor"],
            oos_net_pnl=oos_m["total_pnl"],
            oos_max_drawdown_pct=oos_m["max_drawdown_pct"],
            oos_avg_hold_bars=oos_m["avg_hold_bars"],
            wfe=wfe,
            wfe_label=wfe_label,
        )
        summary.windows.append(wm)
        print(
            f"    {cfg_label}: IS={is_m['total_trades']}t PF={is_m['profit_factor']} "
            f"| OOS={oos_m['total_trades']}t PF={oos_m['profit_factor']} "
            f"WFE={wfe:.2f} [{wfe_label}]"
        )

    _aggregate_summary(summary)
    return summary


def _aggregate_summary(s: SymbolSummary) -> None:
    oos_windows = [w for w in s.windows if w.oos_trades > 0]
    if not oos_windows:
        s.robustness_label = "insufficient_data"
        return

    s.total_oos_trades   = sum(w.oos_trades for w in oos_windows)
    s.avg_oos_win_rate   = round(sum(w.oos_win_rate for w in oos_windows) / len(oos_windows), 1)
    s.avg_oos_pf         = round(sum(w.oos_profit_factor for w in oos_windows) / len(oos_windows), 2)
    s.avg_oos_net_pnl    = round(sum(w.oos_net_pnl for w in oos_windows) / len(oos_windows), 2)
    s.avg_oos_max_dd     = round(sum(w.oos_max_drawdown_pct for w in oos_windows) / len(oos_windows), 2)
    s.avg_oos_hold_bars  = round(sum(w.oos_avg_hold_bars for w in oos_windows) / len(oos_windows), 1)

    valid_wfe_windows = [w for w in s.windows if w.wfe_label not in ("no_is_trades", "unprofitable", "no_oos_trades")]
    if valid_wfe_windows:
        s.avg_wfe = round(sum(w.wfe for w in valid_wfe_windows) / len(valid_wfe_windows), 3)
        s.pct_robust_windows = round(sum(1 for w in valid_wfe_windows if w.wfe >= 0.6) / len(valid_wfe_windows), 2)
    else:
        s.avg_wfe = 0.0
        s.pct_robust_windows = 0.0

    # ── Consistency score (0–100) ─────────────────────────────────────────────
    # Weighted: PF quality (40pts) + win rate (20pts) + WFE (25pts) + trade count (15pts)
    pf_score  = min(s.avg_oos_pf / 2.0, 1.0) * 40      # PF=2.0 → full 40 pts
    wr_score  = min(s.avg_oos_win_rate / 60.0, 1.0) * 20 # 60% WR → full 20 pts
    wfe_score = min(s.avg_wfe / 0.6, 1.0) * 25           # WFE=0.6 → full 25 pts
    # 5+ OOS trades across windows → full 15pts; scales down below
    cnt_score = min(s.total_oos_trades / 5, 1.0) * 15
    s.consistency_score = round(pf_score + wr_score + wfe_score + cnt_score, 1)

    # ── Regime sensitivity: OOS PF variance across windows ───────────────────
    pf_vals = [w.oos_profit_factor for w in oos_windows]
    if len(pf_vals) >= 2:
        pf_range = max(pf_vals) - min(pf_vals)
        s.regime_sensitive = pf_range > 1.5
    else:
        s.regime_sensitive = False

    # ── Labels and recommendations ────────────────────────────────────────────
    if s.avg_oos_pf >= 1.4 and s.total_oos_trades >= 3:
        s.robustness_label = "robust"
        s.recommendation   = "TRADE — consistent PF and sufficient trade volume"
    elif s.avg_oos_pf >= 1.0 and s.total_oos_trades >= 2:
        s.robustness_label = "marginal"
        s.recommendation   = "MONITOR — marginal PF; increase sample before live"
    elif s.total_oos_trades == 0:
        s.robustness_label = "insufficient_data"
        s.recommendation   = "SKIP — no OOS trades in available window"
    elif s.avg_oos_pf < 1.0:
        s.robustness_label = "unprofitable"
        s.recommendation   = "AVOID — OOS average PF below 1.0"
    else:
        s.robustness_label = "marginal"
        s.recommendation   = "MONITOR"

    if s.regime_sensitive:
        s.recommendation += "; REGIME-SENSITIVE — results vary significantly across windows"


# ── Report output ─────────────────────────────────────────────────────────────

def _print_symbol_table(summaries: list[SymbolSummary]) -> None:
    ranked = sorted(summaries, key=lambda s: s.consistency_score, reverse=True)

    header = (
        f"\n{'='*100}\n"
        f"{'BOLLINGER MOMENTUM — WALK-FORWARD SYMBOL RANKING':^100}\n"
        f"{'='*100}\n"
        f"{'Rank':<5} {'Symbol':<7} {'Score':>6} {'OOS Trades':>10} {'WR%':>6} "
        f"{'Avg PF':>7} {'Net PnL':>8} {'Max DD%':>8} "
        f"{'Avg WFE':>8} {'Robust%':>8} {'Regime?':>8} {'Label':<16} Recommendation"
    )
    print(header)
    print("-" * 140)

    for rank, s in enumerate(ranked, 1):
        regime_flag = "YES" if s.regime_sensitive else "no"
        print(
            f"{rank:<5} {s.symbol:<7} {s.consistency_score:>6.1f} "
            f"{s.total_oos_trades:>10} {s.avg_oos_win_rate:>6.1f} "
            f"{s.avg_oos_pf:>7.2f} {s.avg_oos_net_pnl:>8.0f} "
            f"{s.avg_oos_max_dd:>8.1f} "
            f"{s.avg_wfe:>8.3f} {s.pct_robust_windows*100:>7.0f}% "
            f"{regime_flag:>8}  {s.robustness_label:<16} {s.recommendation}"
        )

    print("\n\nPer-window detail:")
    print(f"{'Symbol':<7} {'Config':<22} {'IS Trades':>9} {'IS PF':>6} {'IS Net':>7} "
          f"{'OOS Trades':>10} {'OOS PF':>7} {'OOS Net':>8} {'WFE':>7} {'Label'}")
    print("-" * 110)
    for s in ranked:
        for w in s.windows:
            print(
                f"{s.symbol:<7} {w.config:<22} {w.is_trades:>9} {w.is_profit_factor:>6.2f} "
                f"{w.is_net_pnl:>7.0f} {w.oos_trades:>10} {w.oos_profit_factor:>7.2f} "
                f"{w.oos_net_pnl:>8.0f} {w.wfe:>7.3f}  {w.wfe_label}"
            )


def _save_csv(summaries: list[SymbolSummary], outdir: str) -> str:
    today = date.today().isoformat()
    path = os.path.join(outdir, f"bb_walkforward_{today}.csv")
    rows = []
    for s in summaries:
        for w in s.windows:
            rows.append({
                "symbol":              s.symbol,
                "config":              w.config,
                "is_start":            w.is_start,
                "is_end":              w.is_end,
                "oos_start":           w.oos_start,
                "oos_end":             w.oos_end,
                "is_trades":           w.is_trades,
                "is_win_rate":         w.is_win_rate,
                "is_avg_win":          w.is_avg_win,
                "is_avg_loss":         w.is_avg_loss,
                "is_profit_factor":    w.is_profit_factor,
                "is_net_pnl":          w.is_net_pnl,
                "is_max_drawdown_pct": w.is_max_drawdown_pct,
                "is_avg_hold_bars":    w.is_avg_hold_bars,
                "oos_trades":          w.oos_trades,
                "oos_win_rate":        w.oos_win_rate,
                "oos_avg_win":         w.oos_avg_win,
                "oos_avg_loss":        w.oos_avg_loss,
                "oos_profit_factor":   w.oos_profit_factor,
                "oos_net_pnl":         w.oos_net_pnl,
                "oos_max_drawdown_pct":w.oos_max_drawdown_pct,
                "oos_avg_hold_bars":   w.oos_avg_hold_bars,
                "wfe":                 w.wfe,
                "wfe_label":           w.wfe_label,
                "symbol_score":        s.consistency_score,
                "robustness_label":    s.robustness_label,
                "regime_sensitive":    s.regime_sensitive,
                "recommendation":      s.recommendation,
            })

    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return path


def _save_json_summary(summaries: list[SymbolSummary], outdir: str) -> str:
    today = date.today().isoformat()
    path = os.path.join(outdir, f"bb_walkforward_{today}_summary.json")
    output = []
    for s in summaries:
        output.append({
            "symbol":              s.symbol,
            "consistency_score":   s.consistency_score,
            "robustness_label":    s.robustness_label,
            "regime_sensitive":    bool(s.regime_sensitive),
            "recommendation":      s.recommendation,
            "total_oos_trades":    s.total_oos_trades,
            "avg_oos_win_rate":    s.avg_oos_win_rate,
            "avg_oos_profit_factor": s.avg_oos_pf,
            "avg_oos_net_pnl":     s.avg_oos_net_pnl,
            "avg_oos_max_dd_pct":  s.avg_oos_max_dd,
            "avg_oos_hold_bars":   s.avg_oos_hold_bars,
            "avg_wfe":             s.avg_wfe,
            "pct_robust_windows":  s.pct_robust_windows,
            "windows": [
                {
                    "config":            w.config,
                    "is_start":          w.is_start, "is_end": w.is_end,
                    "oos_start":         w.oos_start, "oos_end": w.oos_end,
                    "is_trades":         w.is_trades,
                    "is_profit_factor":  w.is_profit_factor,
                    "is_net_pnl":        w.is_net_pnl,
                    "oos_trades":        w.oos_trades,
                    "oos_win_rate":      w.oos_win_rate,
                    "oos_avg_win":       w.oos_avg_win,
                    "oos_avg_loss":      w.oos_avg_loss,
                    "oos_profit_factor": w.oos_profit_factor,
                    "oos_net_pnl":       w.oos_net_pnl,
                    "oos_max_dd":        w.oos_max_drawdown_pct,
                    "oos_avg_hold_bars": w.oos_avg_hold_bars,
                    "wfe":               w.wfe,
                    "wfe_label":         w.wfe_label,
                }
                for w in s.windows
            ],
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BB Momentum walk-forward validator")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to validate (default: NVDA TSLA AAPL SPY QQQ MSFT AMD META)")
    parser.add_argument("--outdir", default="output",
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    symbols = [s.upper() for s in args.symbols]

    print(f"\nBollingerMomentum Walk-Forward Validation")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Window configs: {[c[0] for c in WINDOW_CONFIGS]}")
    print(f"Data limit: 60d / ~42 trading days (yfinance 5m cap)")
    print("=" * 70)

    summaries: list[SymbolSummary] = []
    for sym in symbols:
        print(f"\n[{sym}]")
        s = run_symbol_walkforward(sym)
        summaries.append(s)

    _print_symbol_table(summaries)

    csv_path  = _save_csv(summaries, args.outdir)
    json_path = _save_json_summary(summaries, args.outdir)

    print(f"\n\nSaved:\n  CSV  : {csv_path}\n  JSON : {json_path}")


if __name__ == "__main__":
    main()
