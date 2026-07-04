"""CEEI exit-overlay comparison.

Same CEEI long entries (shipped defaults + SMA50 trend filter) across every
exit configuration, so any performance difference is purely the exit layer:

  A_mirrored_sell — legacy: mirrored SELL signal + 20-bar time stop (baseline)
  B_atr_trail     — 1.5-ATR stop, breakeven @1R, 50% partial @1.5R, 2.5-ATR trail
  C_structure     — swing-low structure stop + breakeven + partial + structure trail
  D_chandelier    — chandelier trail (3 ATR from highest high), partial @2R
  E_tightest      — structure+ATR initial and trail, "whichever is more conservative"

Outputs to output/ceei_research/:
  exit_comparison.csv          — headline metrics per config
  exit_reason_mix_by_config.csv
  exit_trades_<config>.csv     — per-trade rows incl. stop evolution events,
                                 partial fills, exit reason, MFE/MAE, R-multiple

Usage:
    python scripts/ceei_exit_research.py [--period 5y] [--configs B_atr_trail C_structure]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.indicators.ceei import CEEIParams, compute_ceei  # noqa: E402
from app.services.strategy.managed_exits import EXIT_PRESETS, manage_trade  # noqa: E402
from ceei_research import OUT_DIR, UNIVERSE, load_universe, metrics  # noqa: E402


def entry_bars(df: pd.DataFrame, params: CEEIParams) -> tuple[list[int], pd.Series]:
    """CEEI BUY signal bars, filtered by the shipped SMA50 trend filter.
    Returns (signal bar indices, full signal series for the mirrored-SELL config)."""
    res = compute_ceei(df["High"], df["Low"], df["Close"], df.get("Volume"), params=params)
    sma50 = df["Close"].rolling(50).mean()
    buys = [
        i for i, s in enumerate(res.signal)
        if s == "BUY" and not pd.isna(sma50.iloc[i]) and df["Close"].iloc[i] > sma50.iloc[i]
    ]
    return buys, res.signal


def run_config(name: str, data: dict[str, pd.DataFrame],
               entries: dict[str, tuple[list[int], pd.Series]]) -> pd.DataFrame:
    cfg = EXIT_PRESETS[name]
    rows = []
    for sym, df in data.items():
        buys, signal = entries[sym]
        next_free = 0
        for t in buys:
            e = t + 1
            if e >= len(df) or e < next_free:
                continue
            trade = manage_trade(sym, df, e, cfg,
                                 sell_signal=signal if cfg.use_sell_signal else None)
            if trade is None:
                continue
            rows.append({
                "config": name, "symbol": sym, "sector": UNIVERSE.get(sym, "?"),
                "entry_date": trade.entry_date, "entry": trade.entry,
                "initial_stop": trade.initial_stop, "r_unit": trade.r_unit,
                "partial_date": trade.partial_date, "partial_price": trade.partial_price,
                "partial_r": trade.partial_r,
                "exit_date": trade.exit_date, "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason, "final_stop": trade.final_stop,
                "stop_moves": trade.stop_moves, "hold_bars": trade.hold_bars,
                "ret_pct": trade.realized_pct, "realized_r": trade.realized_r,
                "mfe_pct": trade.mfe_pct, "mae_pct": trade.mae_pct,
                "mfe_captured_pct": trade.mfe_captured_pct,
                "events": "; ".join(trade.events),
            })
            # no overlapping positions per symbol
            next_free = e + trade.hold_bars + 1
    return pd.DataFrame(rows)


def reason_family(reason: str) -> str:
    if reason.startswith(("stop", "stop_gap")):
        return "stop"
    if reason in ("profit_target", "sell_signal", "time_stop", "end_of_data"):
        return {"profit_target": "target", "sell_signal": "sell_signal",
                "time_stop": "time_stop", "end_of_data": "other"}[reason]
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="CEEI exit-overlay comparison")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--configs", nargs="*", default=list(EXIT_PRESETS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {len(UNIVERSE)} symbols ({args.period}) ...")
    data = load_universe(args.period)
    years = len(data["SPY"]) / 252 if "SPY" in data else 5.0

    params = CEEIParams()
    print("Computing CEEI entries (shared across all configs) ...")
    entries = {sym: entry_bars(df, params) for sym, df in data.items()}
    n_entries = sum(len(b) for b, _ in entries.values())
    print(f"  {n_entries} filtered BUY signals")

    summary_rows, mix_rows = [], []
    for name in args.configs:
        print(f"Running {name} ...")
        trades = run_config(name, data, entries)
        trades.to_csv(OUT_DIR / f"exit_trades_{name}.csv", index=False)
        m = metrics(trades, years)
        # MFE capture is only meaningful on trades that had favorable excursion
        # to capture; a loser with tiny MFE produces absurd negative ratios.
        cap = trades.loc[trades["mfe_pct"] >= 1.0, "mfe_captured_pct"].dropna()
        m["avg_mfe_captured_pct"] = round(cap.mean(), 1) if len(cap) else np.nan
        m["avg_realized_r"] = round(trades["realized_r"].mean(), 3) if len(trades) else np.nan
        m["partial_fill_rate"] = round(trades["partial_price"].notna().mean() * 100, 1) \
            if len(trades) else np.nan
        summary_rows.append({"config": name, **m})

        fam = trades["exit_reason"].map(reason_family).value_counts(normalize=True) * 100
        mix_rows.append({"config": name,
                         **{k: round(float(fam.get(k, 0.0)), 1)
                            for k in ("stop", "target", "sell_signal", "time_stop", "other")}})

    summary = pd.DataFrame(summary_rows)
    mix = pd.DataFrame(mix_rows)
    summary.to_csv(OUT_DIR / "exit_comparison.csv", index=False)
    mix.to_csv(OUT_DIR / "exit_reason_mix_by_config.csv", index=False)

    print("\n=== EXIT CONFIG COMPARISON ===")
    print(summary.to_string(index=False))
    print("\n=== EXIT REASON MIX (%) ===")
    print(mix.to_string(index=False))
    print(f"\nPer-trade CSVs + comparison in {OUT_DIR}/")


if __name__ == "__main__":
    main()
