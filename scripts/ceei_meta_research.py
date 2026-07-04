"""CEEI meta-layer batch research driver.

Runs every discovered swing strategy in all coupling modes (base, setup filter,
trigger filter, score filter, ranking, exit assist, veto) over a shared
universe, with one shared CEEI configuration (no per-strategy tuning).

Outputs to output/ceei_meta/:
  summary_by_strategy_and_mode.csv  — full grid + %change vs base + flags
  summary_by_mode.csv               — mode aggregates across strategies
  summary_by_strategy.csv           — per strategy: base metrics + best mode
  by_symbol.csv                     — per (strategy, mode, symbol)
  by_regime.csv                     — per (strategy, mode, SPY trend regime)
  trades_<strategy>_<mode>.csv      — raw trades (only with --save-trades)
  meta_tables.md                    — auto-generated tables

Base-strategy signal series are cached in output/ceei_meta/cache/ so re-runs
and added modes are cheap.

Usage:
    python scripts/ceei_meta_research.py                       # all strategies
    python scripts/ceei_meta_research.py --strategies rsi2_reversion trend_follow
    python scripts/ceei_meta_research.py --exclude supertrend trend_follow
    python scripts/ceei_meta_research.py --symbols AAPL MSFT SPY --period 3y
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest.ceei_meta import (  # noqa: E402
    MODES, CouplingConfig, cached_signal_series, compute_ceei_for,
    discover_strategies, entry_mask, metrics, ranking_selection, simulate,
)
from app.services.market_data.provider import get_ohlcv  # noqa: E402
from ceei_research import UNIVERSE, spy_regimes  # noqa: E402

OUT_DIR = Path("output/ceei_meta")
CACHE_DIR = OUT_DIR / "cache"

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "JPM",
                   "UNH", "CAT", "XOM", "SPY", "QQQ", "IWM"]


def main() -> None:
    ap = argparse.ArgumentParser(description="CEEI meta-layer research")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--universe", action="store_true",
                    help="use the full 49-symbol research universe")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (keeps prior runs intact)")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="whitelist of strategy types (default: all discovered)")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--save-trades", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CouplingConfig()
    strategies = discover_strategies(include=args.strategies, exclude=args.exclude)
    print(f"Strategies ({len(strategies)}): {', '.join(strategies)}")

    symbols = list(UNIVERSE) if args.universe else args.symbols
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_ohlcv(sym, period=args.period)
            if df is not None and len(df) > cfg.warmup_bars + 100:
                data[sym] = df
        except Exception as e:
            print(f"  skip {sym}: {e}")
    print(f"Symbols ({len(data)}): {', '.join(data)}")
    spy = data["SPY"] if "SPY" in data else get_ohlcv("SPY", period=args.period)
    regimes = spy_regimes(spy)
    # tz-naive date index so trade signal_date strings can look regimes up
    regimes.index = pd.DatetimeIndex(regimes.index).tz_localize(None).normalize()
    years = len(spy) / 252

    ceei = {sym: compute_ceei_for(df) for sym, df in data.items()}

    grid_rows, symbol_rows, regime_rows = [], [], []
    base_trades_by_strategy: dict[str, pd.DataFrame] = {}

    for strat in strategies:
        t0 = time.time()
        signals = {sym: cached_signal_series(strat, sym, df, CACHE_DIR, cfg.warmup_bars)
                   for sym, df in data.items()}
        print(f"[{strat}] signals ready ({time.time() - t0:.0f}s)")

        # Candidate entry bars for the ranking mode = base-mode BUY bars
        candidates = {sym: [i for i, s in enumerate(sig) if s == "BUY"]
                      for sym, sig in signals.items()}

        for mode in args.modes:
            allowed = None
            if mode == "E_ranking":
                allowed = ranking_selection(
                    candidates, {s: d.index for s, d in data.items()},
                    {s: ceei[s].ceei_score for s in data}, cfg.ranking_top)
            rows = []
            for sym, df in data.items():
                mask = entry_mask(mode, ceei[sym], cfg)
                rows += simulate(sym, df, signals[sym], mask, ceei[sym], cfg,
                                 exit_assist=(mode == "F_exit_assist"),
                                 allowed_bars=allowed.get(sym) if allowed else None)
            trades = pd.DataFrame(rows)
            if not trades.empty:
                reg = regimes["trend"].reindex(pd.to_datetime(trades["signal_date"]))
                trades["trend_regime"] = reg.fillna("unknown").to_numpy()
            if args.save_trades and not trades.empty:
                trades.to_csv(OUT_DIR / f"trades_{strat}_{mode}.csv", index=False)
            if mode == "A_base":
                base_trades_by_strategy[strat] = trades

            grid_rows.append({"strategy": strat, "mode": mode, **metrics(trades, years)})

            if not trades.empty:
                for sym, grp in trades.groupby("symbol"):
                    symbol_rows.append({"strategy": strat, "mode": mode, "symbol": sym,
                                        **metrics(grp, years)})
                for rg, grp in trades.groupby("trend_regime"):
                    regime_rows.append({"strategy": strat, "mode": mode, "regime": rg,
                                        **metrics(grp, years)})

    grid = pd.DataFrame(grid_rows)
    by_symbol = pd.DataFrame(symbol_rows)
    by_regime = pd.DataFrame(regime_rows)

    # ── %change vs base + flags + symbol-level improvement rate ──────────────
    base = grid[grid["mode"] == "A_base"].set_index("strategy")
    base_sym = by_symbol[by_symbol["mode"] == "A_base"].set_index(["strategy", "symbol"])

    def _enrich(row):
        b = base.loc[row["strategy"]] if row["strategy"] in base.index else None
        if b is None or not b.get("trades") or not row.get("trades"):
            return pd.Series({"exp_change_pct": np.nan, "trade_change_pct": np.nan,
                              "wr_up_exp_down": False, "symbol_improve_rate": np.nan})
        exp_chg = ((row["expectancy"] - b["expectancy"]) / abs(b["expectancy"]) * 100
                   if b["expectancy"] not in (0, np.nan) else np.nan)
        trd_chg = (row["trades"] - b["trades"]) / b["trades"] * 100
        wr_flag = bool(row["win_rate"] > b["win_rate"] and row["expectancy"] < b["expectancy"])
        # symbol-level improvement rate (symbols with >=3 trades in both runs)
        mode_sym = by_symbol[(by_symbol["strategy"] == row["strategy"])
                             & (by_symbol["mode"] == row["mode"])].set_index("symbol")
        improved = total = 0
        for sym in mode_sym.index:
            key = (row["strategy"], sym)
            if key in base_sym.index:
                bs, ms = base_sym.loc[key], mode_sym.loc[sym]
                if bs["trades"] >= 3 and ms["trades"] >= 3:
                    total += 1
                    improved += int(ms["expectancy"] > bs["expectancy"])
        return pd.Series({
            "exp_change_pct": round(exp_chg, 1) if not pd.isna(exp_chg) else np.nan,
            "trade_change_pct": round(trd_chg, 1),
            "wr_up_exp_down": wr_flag,
            "symbol_improve_rate": round(improved / total * 100, 1) if total else np.nan,
        })

    sfx = f"_{args.tag}" if args.tag else ""
    grid = pd.concat([grid, grid.apply(_enrich, axis=1)], axis=1)
    grid.to_csv(OUT_DIR / f"summary_by_strategy_and_mode{sfx}.csv", index=False)
    by_symbol.to_csv(OUT_DIR / f"by_symbol{sfx}.csv", index=False)
    by_regime.to_csv(OUT_DIR / f"by_regime{sfx}.csv", index=False)

    # summary_by_mode: aggregate across strategies (median of per-strategy stats)
    agg_rows = []
    for mode, grp in grid.groupby("mode"):
        g = grp[grp["trades"] > 0]
        agg_rows.append({
            "mode": mode,
            "strategies": len(g),
            "total_trades": int(g["trades"].sum()),
            "median_expectancy": round(g["expectancy"].median(), 3),
            "median_win_rate": round(g["win_rate"].median(), 1),
            "median_exp_change_pct": round(g["exp_change_pct"].median(), 1)
                if mode != "A_base" else 0.0,
            "strategies_improved": int((g["exp_change_pct"] > 0).sum())
                if mode != "A_base" else np.nan,
            "median_trade_change_pct": round(g["trade_change_pct"].median(), 1)
                if mode != "A_base" else 0.0,
            "wr_up_exp_down_count": int(g["wr_up_exp_down"].sum()),
        })
    by_mode = pd.DataFrame(agg_rows)
    by_mode.to_csv(OUT_DIR / f"summary_by_mode{sfx}.csv", index=False)

    # summary_by_strategy: base metrics + best non-base mode by expectancy
    strat_rows = []
    for strat, grp in grid.groupby("strategy"):
        b = grp[grp["mode"] == "A_base"]
        nb = grp[(grp["mode"] != "A_base") & (grp["trades"] >= 10)]
        best = nb.loc[nb["expectancy"].idxmax()] if len(nb) else None
        strat_rows.append({
            "strategy": strat,
            "base_trades": int(b["trades"].iloc[0]) if len(b) else 0,
            "base_expectancy": float(b["expectancy"].iloc[0]) if len(b) and b["trades"].iloc[0] else np.nan,
            "base_win_rate": float(b["win_rate"].iloc[0]) if len(b) and b["trades"].iloc[0] else np.nan,
            "best_mode": best["mode"] if best is not None else "",
            "best_expectancy": best["expectancy"] if best is not None else np.nan,
            "best_exp_change_pct": best["exp_change_pct"] if best is not None else np.nan,
            "best_trade_change_pct": best["trade_change_pct"] if best is not None else np.nan,
        })
    by_strategy = pd.DataFrame(strat_rows).sort_values("best_exp_change_pct", ascending=False)
    by_strategy.to_csv(OUT_DIR / f"summary_by_strategy{sfx}.csv", index=False)

    # tables.md
    def _md(df: pd.DataFrame) -> str:
        d = df.fillna("")
        cols = [str(c) for c in d.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for _, r in d.iterrows()]
        return "\n".join(lines)

    with open(OUT_DIR / f"meta_tables{sfx}.md", "w", encoding="utf-8") as f:
        f.write(f"# CEEI meta-layer tables ({args.period}, {len(data)} symbols, "
                f"{len(strategies)} strategies)\n\n")
        f.write(f"## By mode\n\n{_md(by_mode)}\n\n")
        f.write(f"## By strategy (best mode)\n\n{_md(by_strategy)}\n\n")
        f.write(f"## Full grid\n\n{_md(grid.sort_values(['strategy', 'mode']))}\n\n")

    print("\n=== BY MODE ===");     print(by_mode.to_string(index=False))
    print("\n=== BY STRATEGY ==="); print(by_strategy.to_string(index=False))
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
