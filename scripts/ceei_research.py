"""CEEI research hardening harness.

Robustness testing — NOT curve-fit optimization:
  1. Universe-scale validation (liquid US stocks + ETFs), grouped by symbol,
     sector, and volatility bucket.
  2. SPY regime analysis (trend x volatility) of every trade.
  3. Entry/exit diagnostics: MFE, MAE, forward returns at 3/5/10/20 bars.
  4. One-at-a-time parameter sensitivity around the shipped defaults —
     looking for stable zones, not a single best point.
  5. Optional signal-quality filters (SMA50/SMA150 trend, dollar volume,
     ADX, gap) compared against base CEEI.

All outputs land in output/ceei_research/ as CSVs plus an auto-generated
tables file. Trade simulation: enter next-bar open after a BUY, exit
next-bar open after a SELL signal or after max_hold bars — transparent and
identical across every configuration so comparisons are apples-to-apples.

Usage:
    python scripts/ceei_research.py                 # full run, 5y
    python scripts/ceei_research.py --period 3y --max-hold 15
    python scripts/ceei_research.py --skip-sensitivity
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.indicators.ceei import CEEIParams, compute_ceei  # noqa: E402
from app.services.market_data.provider import get_ohlcv  # noqa: E402

OUT_DIR = Path("output/ceei_research")

# ── Universe: liquid US large caps + broad/sector ETFs ────────────────────────
UNIVERSE: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "AVGO": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "ADBE": "Technology",
    # Communication
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "NKE": "Consumer",
    "MCD": "Consumer", "SBUX": "Consumer", "COST": "Consumer", "WMT": "Consumer",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "V": "Financials", "MA": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare",
    # Energy / Industrials
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "UPS": "Industrials",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLF": "ETF", "XLE": "ETF", "XLK": "ETF", "XLV": "ETF",
    "GLD": "ETF", "TLT": "ETF",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data + feature helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_universe(period: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        try:
            df = get_ohlcv(sym, period=period)
            if df is not None and len(df) > 300:
                data[sym] = df
            else:
                print(f"  skip {sym}: insufficient data")
        except Exception as e:
            print(f"  skip {sym}: {e}")
    return data


def spy_regimes(spy: pd.DataFrame) -> pd.DataFrame:
    """Per-date SPY regime tags: trend (bull/bear via SMA200) and volatility
    (high/low via 20d realized vol vs its expanding median)."""
    close = spy["Close"]
    trend = np.where(close > close.rolling(200).mean(), "bull", "bear")
    rv = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol = np.where(rv > rv.expanding(min_periods=100).median(), "high_vol", "low_vol")
    return pd.DataFrame({"trend": trend, "vol": vol}, index=spy.index)


def _adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features used by the optional signal-quality filters,
    evaluated at the SIGNAL bar (entry is next-bar open)."""
    close, volume = df["Close"], df["Volume"]
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - close.shift(1)).abs(),
                    (df["Low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    open_next = df["Open"].shift(-1)
    return pd.DataFrame({
        "above_sma50": close > close.rolling(50).mean(),
        "above_sma150": close > close.rolling(150).mean(),
        "dollar_vol_20d": (close * volume).rolling(20).mean(),
        "adx14": _adx_series(df),
        # Gap of the ENTRY bar's open vs the signal close, in ATRs
        "entry_gap_atr": ((open_next - close) / atr.replace(0, np.nan)),
    }, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ──────────────────────────────────────────────────────────────────────────────

FWD_BARS = (3, 5, 10, 20)


def simulate_trades(
    symbol: str,
    df: pd.DataFrame,
    params: CEEIParams,
    regimes: pd.DataFrame,
    max_hold: int = 20,
    with_features: bool = False,
) -> pd.DataFrame:
    """Long-only simulation of CEEI signals over a full frame.
    Entry: next-bar open after BUY. Exit: next-bar open after SELL, or the
    open max_hold bars after entry, or the final close. Returns one row per
    round trip with diagnostics."""
    res = compute_ceei(df["High"], df["Low"], df["Close"], df.get("Volume"), params=params)
    sig = res.signal.to_numpy()
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    n = len(df)
    feats = entry_features(df) if with_features else None
    reg = regimes.reindex(df.index).ffill()

    rows = []
    t = 0
    while t < n - 1:
        if sig[t] != "BUY":
            t += 1
            continue
        e = t + 1                      # entry bar
        entry = opens[e]
        # find exit
        exit_i, exit_px, exit_reason = None, None, None
        for u in range(e, min(e + max_hold, n)):
            if u > e and sig[u] == "SELL" and u + 1 < n:
                exit_i, exit_px, exit_reason = u + 1, opens[u + 1], "sell_signal"
                break
        if exit_i is None:
            u = e + max_hold
            if u < n:
                exit_i, exit_px, exit_reason = u, opens[u], "time_stop"
            else:
                exit_i, exit_px, exit_reason = n - 1, closes[-1], "end_of_data"
        ret = (exit_px - entry) / entry * 100
        hold = exit_i - e
        mfe = (highs[e:exit_i + 1].max() - entry) / entry * 100
        mae = (lows[e:exit_i + 1].min() - entry) / entry * 100
        row = {
            "symbol": symbol,
            "sector": UNIVERSE.get(symbol, "?"),
            "signal_date": str(df.index[t])[:10],
            "entry_date": str(df.index[e])[:10],
            "exit_date": str(df.index[exit_i])[:10],
            "entry": round(entry, 4), "exit": round(exit_px, 4),
            "ret_pct": round(ret, 3), "hold_bars": hold,
            "exit_reason": exit_reason,
            "mfe_pct": round(mfe, 3), "mae_pct": round(mae, 3),
            "trend_regime": reg["trend"].iloc[t] if not reg.empty else "?",
            "vol_regime": reg["vol"].iloc[t] if not reg.empty else "?",
        }
        for k in FWD_BARS:
            row[f"fwd_{k}"] = round((closes[e + k] - entry) / entry * 100, 3) if e + k < n else np.nan
        if feats is not None:
            f = feats.iloc[t]
            row.update({
                "above_sma50": bool(f["above_sma50"]),
                "above_sma150": bool(f["above_sma150"]),
                "dollar_vol_20d": float(f["dollar_vol_20d"]) if not pd.isna(f["dollar_vol_20d"]) else np.nan,
                "adx14": round(float(f["adx14"]), 1) if not pd.isna(f["adx14"]) else np.nan,
                "entry_gap_atr": round(float(f["entry_gap_atr"]), 2) if not pd.isna(f["entry_gap_atr"]) else np.nan,
            })
        rows.append(row)
        t = exit_i + 1                  # no overlapping positions
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame, years: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    # Sequential-compounding equity for max drawdown
    eq = (1 + r / 100).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax() * 100).max()
    tpy = len(r) / years if years > 0 else np.nan
    sharpe = (r.mean() / r.std() * np.sqrt(tpy)) if len(r) > 2 and r.std() > 0 else np.nan
    return {
        "trades": len(r),
        "win_rate": round(len(wins) / len(r) * 100, 1),
        "avg_win": round(wins.mean(), 2) if len(wins) else np.nan,
        "avg_loss": round(losses.mean(), 2) if len(losses) else np.nan,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if losses.sum() != 0 else np.inf,
        "expectancy": round(r.mean(), 3),
        "max_dd_pct": round(dd, 2),
        "sharpe": round(sharpe, 2) if not pd.isna(sharpe) else np.nan,
        "avg_hold_bars": round(trades["hold_bars"].mean(), 1),
    }


def grouped_metrics(trades: pd.DataFrame, by: str, years: float) -> pd.DataFrame:
    out = []
    for key, grp in trades.groupby(by):
        out.append({by: key, **metrics(grp, years)})
    return pd.DataFrame(out).sort_values("expectancy", ascending=False)


def vol_bucket_map(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Bucket symbols into low/mid/high volatility terciles by median ATR%."""
    atr_pct = {}
    for sym, df in data.items():
        tr = pd.concat([df["High"] - df["Low"],
                        (df["High"] - df["Close"].shift(1)).abs(),
                        (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        atr_pct[sym] = float((tr.rolling(14).mean() / df["Close"]).median() * 100)
    s = pd.Series(atr_pct)
    lo, hi = s.quantile(1 / 3), s.quantile(2 / 3)
    return {sym: ("low_vol" if v <= lo else "high_vol" if v > hi else "mid_vol")
            for sym, v in s.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity + filters
# ──────────────────────────────────────────────────────────────────────────────

SENSITIVITY_GRID: dict[str, list] = {
    "vol_lookback": [20, 25, 30, 35, 40],
    "expansion_threshold": [35.0, 40.0, 45.0, 50.0, 55.0],
    "efficiency_min": [45.0, 50.0, 55.0, 60.0, 65.0],
    "setup_window": [5, 8, 10, 15, 20],
    "buy_threshold": [40.0, 44.0, 48.0, 52.0, 56.0],
}

FILTERS = {
    "sma50_trend": lambda t: t["above_sma50"],
    "sma150_trend": lambda t: t["above_sma150"],
    "dollar_vol_20m": lambda t: t["dollar_vol_20d"] >= 20e6,
    "adx_min_20": lambda t: t["adx14"] >= 20,
    "gap_max_1atr": lambda t: t["entry_gap_atr"] <= 1.0,
}


def run_sensitivity(data, regimes, base: CEEIParams, years, max_hold) -> pd.DataFrame:
    rows = []
    for pname, values in SENSITIVITY_GRID.items():
        for v in values:
            params = replace(base, **{pname: v})
            all_trades = pd.concat(
                [simulate_trades(s, df, params, regimes, max_hold) for s, df in data.items()],
                ignore_index=True,
            )
            rows.append({
                "param": pname, "value": v,
                "is_default": v == getattr(base, pname),
                **metrics(all_trades, years),
            })
            print(f"  sensitivity {pname}={v}: trades={rows[-1].get('trades')} "
                  f"exp={rows[-1].get('expectancy')} wr={rows[-1].get('win_rate')}")
    return pd.DataFrame(rows)


def run_filters(trades: pd.DataFrame, years: float) -> pd.DataFrame:
    rows = [{"filter": "base (none)", **metrics(trades, years)}]
    for name, fn in FILTERS.items():
        mask = fn(trades).fillna(False)
        rows.append({"filter": name, **metrics(trades[mask], years)})
    # All trend+quality filters combined
    combined = trades
    for name in ("sma50_trend", "adx_min_20", "gap_max_1atr"):
        combined = combined[FILTERS[name](combined).fillna(False)]
    rows.append({"filter": "sma50+adx20+gap1atr", **metrics(combined, years)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _md(frame: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    df = frame.fillna("")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_tables(out_dir: Path, header: str) -> None:
    """Build tables.md from the CSVs already on disk."""
    sections = [
        ("Overall", "overall.csv"), ("By sector", "by_sector.csv"),
        ("By volatility bucket", "by_vol_bucket.csv"), ("By regime", "by_regime.csv"),
        ("Entry/exit diagnostics", "entry_exit_diagnostics.csv"),
        ("Exit reason mix", "exit_reason_mix.csv"), ("Filters", "filters.csv"),
        ("By symbol", "by_symbol.csv"), ("Parameter sensitivity", "sensitivity.csv"),
    ]
    with open(out_dir / "tables.md", "w", encoding="utf-8") as f:
        f.write(f"# CEEI research tables ({header})\n\n")
        for title, name in sections:
            path = out_dir / name
            if path.exists():
                f.write(f"## {title}\n\n{_md(pd.read_csv(path))}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="CEEI research hardening")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--max-hold", type=int, default=20)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild tables.md from existing CSVs and exit")
    args = ap.parse_args()

    if args.tables_only:
        write_tables(OUT_DIR, f"{args.period}, max_hold={args.max_hold}")
        print(f"tables.md rebuilt in {OUT_DIR}/")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = CEEIParams()

    print(f"Loading {len(UNIVERSE)} symbols ({args.period}) ...")
    data = load_universe(args.period)
    print(f"Loaded {len(data)} symbols.")
    spy = data.get("SPY")
    if spy is None:
        raise SystemExit("SPY data required for regime tagging")
    regimes = spy_regimes(spy)
    years = len(spy) / 252

    # 1+3+5. Base trades with diagnostics + filter features
    print("Simulating base trades ...")
    trades = pd.concat(
        [simulate_trades(s, df, base, regimes, args.max_hold, with_features=True)
         for s, df in data.items()],
        ignore_index=True,
    )
    trades.to_csv(OUT_DIR / "trades.csv", index=False)
    print(f"  {len(trades)} trades")

    vb = vol_bucket_map(data)
    trades["vol_bucket"] = trades["symbol"].map(vb)

    by_symbol = grouped_metrics(trades, "symbol", years)
    by_sector = grouped_metrics(trades, "sector", years)
    by_volbkt = grouped_metrics(trades, "vol_bucket", years)
    overall = pd.DataFrame([{"group": "ALL", **metrics(trades, years)}])
    by_symbol.to_csv(OUT_DIR / "by_symbol.csv", index=False)
    by_sector.to_csv(OUT_DIR / "by_sector.csv", index=False)
    by_volbkt.to_csv(OUT_DIR / "by_vol_bucket.csv", index=False)
    overall.to_csv(OUT_DIR / "overall.csv", index=False)

    # 2. Regime analysis
    trades["regime"] = trades["trend_regime"] + "/" + trades["vol_regime"]
    by_regime = pd.concat([
        grouped_metrics(trades, "trend_regime", years).rename(columns={"trend_regime": "regime"}),
        grouped_metrics(trades, "vol_regime", years).rename(columns={"vol_regime": "regime"}),
        grouped_metrics(trades, "regime", years),
    ], ignore_index=True)
    by_regime.to_csv(OUT_DIR / "by_regime.csv", index=False)

    # 3. Entry/exit diagnostics
    diag_rows = []
    for col, label in [("mfe_pct", "MFE"), ("mae_pct", "MAE")] + \
                      [(f"fwd_{k}", f"fwd_{k}bars") for k in FWD_BARS]:
        s = trades[col].dropna()
        diag_rows.append({
            "metric": label, "mean": round(s.mean(), 2), "median": round(s.median(), 2),
            "p25": round(s.quantile(.25), 2), "p75": round(s.quantile(.75), 2),
            "pct_positive": round((s > 0).mean() * 100, 1),
        })
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(OUT_DIR / "entry_exit_diagnostics.csv", index=False)
    exit_mix = trades["exit_reason"].value_counts(normalize=True).round(3) * 100
    exit_mix.to_csv(OUT_DIR / "exit_reason_mix.csv")

    # 4. Parameter sensitivity
    if not args.skip_sensitivity:
        print("Running parameter sensitivity (one-at-a-time) ...")
        sens = run_sensitivity(data, regimes, base, years, args.max_hold)
        sens.to_csv(OUT_DIR / "sensitivity.csv", index=False)
    else:
        sens = None

    # 5. Filters
    filt = run_filters(trades, years)
    filt.to_csv(OUT_DIR / "filters.csv", index=False)

    # 6. Auto tables file (narrative summary is CEEI_RESEARCH.md, written separately)
    write_tables(OUT_DIR, f"{args.period}, max_hold={args.max_hold}, "
                          f"{len(data)} symbols, {len(trades)} trades")

    print("\n=== OVERALL ===");            print(overall.to_string(index=False))
    print("\n=== BY REGIME ===");          print(by_regime.to_string(index=False))
    print("\n=== DIAGNOSTICS ===");        print(diagnostics.to_string(index=False))
    print("\n=== EXIT REASONS (%) ===");   print(exit_mix.to_string())
    print("\n=== FILTERS ===");            print(filt.to_string(index=False))
    print("\n=== BY SECTOR ===");          print(by_sector.to_string(index=False))
    print("\n=== BY VOL BUCKET ===");      print(by_volbkt.to_string(index=False))
    if sens is not None:
        print("\n=== SENSITIVITY ===");    print(sens.to_string(index=False))
    print(f"\nAll CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
