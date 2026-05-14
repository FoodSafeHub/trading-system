#!/usr/bin/env python
"""
Automated filter optimizer: grid-searches entry filter thresholds for any
strategy+symbol, validates improvement on OOS walk-forward data, then
saves the best profile to symbol_profiles.json.

Algorithm:
  1. Run full backtest to get trade snapshots + feature distributions
  2. For each feature with moderate+ Cohen's d separation:
     - Test a range of threshold values (percentiles of winning trades)
     - Score each threshold by: filtered_win_rate × sqrt(surviving_trades)
       (penalises filters that discard too many trades)
  3. Assemble the best threshold per feature into a candidate filter set
  4. Verify on OOS walk-forward: compare CAGR before vs after filters
  5. If OOS CAGR improves, save to profile; else save with verified=False

Usage:
    python scripts/optimize_filters.py --strategy BB_Mean_Reversion --symbol NVDA
    python scripts/optimize_filters.py --all --period 5y
    python scripts/optimize_filters.py --strategy EMA_Mean_Reversion --symbol SPY --no-verify
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend, MaCrossoverRsi, BreakoutConsolidation,
    BollingerMeanReversionUptrend, FibPullbackSupport,
)
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.backtest.walkforward_engine import run_rolling_walk_forward
from app.services.backtest.symbol_profiles import (
    SymbolFilterProfile, save_profile, calibrate_from_snapshots,
)
from scripts.analyze_trades import extract_trade_snapshots, analyze_snapshots

STRATEGIES = {
    "EMA_Mean_Reversion":     EmaMeanReversionUptrend(),
    "MA_Crossover_RSI":       MaCrossoverRsi(),
    "Breakout_Consolidation": BreakoutConsolidation(),
    "BB_Mean_Reversion":      BollingerMeanReversionUptrend(),
    "Fib_Pullback_Support":   FibPullbackSupport(),
}

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL"]

# Feature → strategy config key mapping
# When we find a good threshold, we patch the strategy config to apply it
_FEATURE_TO_CONFIG = {
    "volume_ratio":    ("vol_ratio_min",           "min"),
    "atr_pct":         ("atr_skip_threshold",       "max"),  # EMA strat: skip spikes
    "rsi2":            ("rsi_entry_threshold",      "max"),
    "rsi":             ("rsi_min",                  "min"),
    "wick_ratio":      ("wick_ratio_min",           "min"),
    "prox_pct":        ("price_ema_proximity_pct",  "max"),
    "bb_pos":          ("bb_pos_max",               "max"),
    "ema_spread_pct":  ("filter_ema_spread_min",    "min"),
    "bb_depth_pct":    ("filter_bb_depth_min",      "min"),
    "range_atr_ratio": ("filter_range_atr_max",     "max"),
}


# ── Grid search ───────────────────────────────────────────────────────────────

def _percentile(vals: list, p: float) -> float:
    idx = max(0, min(len(vals) - 1, int(len(vals) * p)))
    return sorted(vals)[idx]


def _score_threshold(snapshots, key, thresh, direction) -> tuple[float, float]:
    """
    Score a candidate threshold.
    Returns (score, filtered_win_rate).
    score = filtered_win_rate × sqrt(n_surviving) — rewards both quality and quantity.
    """
    if direction == "min":
        surviving = [s for s in snapshots if (s.get(key) or 0) >= thresh]
    else:
        surviving = [s for s in snapshots if (s.get(key) or 999) <= thresh]

    n = len(surviving)
    if n < 5:
        return 0.0, 0.0
    if n / len(snapshots) < 0.30:  # discard if fewer than 30% trades survive
        return 0.0, 0.0

    wins = sum(1 for s in surviving if s["outcome"] == "win")
    wr   = wins / n
    return wr * (n ** 0.5), wr


def grid_search_filters(
    strategy_name: str,
    snapshots: list[dict],
    analysis: dict,
    min_cohens_d: float = 0.20,
) -> dict[str, float]:
    """
    For each feature with |Cohen's d| >= min_cohens_d, find the best threshold.
    Returns dict of {config_key: threshold_value}.
    """
    wins   = [s for s in snapshots if s["outcome"] == "win"]
    losses = [s for s in snapshots if s["outcome"] == "loss"]
    baseline_wr = len(wins) / len(snapshots) if snapshots else 0.0

    best_filters: dict[str, float] = {}

    ranked = sorted(
        [(k, v) for k, v in analysis.items() if abs(v.get("cohens_d") or 0) >= min_cohens_d],
        key=lambda x: abs(x[1]["cohens_d"]),
        reverse=True,
    )

    for feat_key, v in ranked:
        if feat_key not in _FEATURE_TO_CONFIG:
            continue
        cfg_key, direction = _FEATURE_TO_CONFIG[feat_key]

        # Build candidate thresholds from percentiles of winning-trade values
        w_vals = sorted([s[feat_key] for s in wins if isinstance(s.get(feat_key), (int, float))])
        if len(w_vals) < 4:
            continue

        if direction == "min":
            # We want a minimum: scan 10th–60th percentile of wins
            candidates = sorted({round(_percentile(w_vals, p), 3)
                                  for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]})
        else:
            # We want a maximum: scan 40th–90th percentile of wins (higher end)
            candidates = sorted({round(_percentile(w_vals, p), 3)
                                  for p in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]},
                                 reverse=True)

        best_score = 0.0
        best_thresh = 0.0
        best_wr = baseline_wr

        for thresh in candidates:
            score, fwr = _score_threshold(snapshots, feat_key, thresh, direction)
            if score > best_score and fwr > baseline_wr + 0.03:
                best_score = score
                best_thresh = thresh
                best_wr = fwr

        if best_thresh > 0.0:
            best_filters[cfg_key] = best_thresh
            print(f"    {feat_key} → {cfg_key} {direction}>= {best_thresh}  "
                  f"(filtered WR {best_wr:.1%} vs baseline {baseline_wr:.1%})")

    return best_filters


# ── Walk-forward verification ─────────────────────────────────────────────────

def _apply_filters(strategy_name: str, filters: dict):
    """Temporarily patch strategy config with filter thresholds."""
    strat = STRATEGIES[strategy_name]
    original = {}
    for k, v in filters.items():
        if k in strat.config:
            original[k] = strat.config[k]
            strat.config[k] = v
    return original


def _restore_filters(strategy_name: str, original: dict):
    strat = STRATEGIES[strategy_name]
    for k, v in original.items():
        strat.config[k] = v


def verify_on_walkforward(
    strategy_name: str,
    symbol: str,
    filters: dict,
    period: str = "5y",
    train_years: float = 3.0,
    test_years: float = 1.0,
) -> tuple[float | None, float | None, bool]:
    """
    Run rolling walk-forward before and after applying filters.
    Returns (wfe_before, wfe_after, improved).
    """
    strat = STRATEGIES[strategy_name]

    try:
        wf_before = run_rolling_walk_forward(
            strat, symbol, period=period,
            train_years=train_years, test_years=test_years,
        )
        cagr_before = wf_before.oos_cagr
        wfe_before  = wf_before.wfe
    except Exception as e:
        print(f"    Walk-forward (before) failed: {e}")
        return None, None, False

    original = _apply_filters(strategy_name, filters)
    try:
        wf_after = run_rolling_walk_forward(
            strat, symbol, period=period,
            train_years=train_years, test_years=test_years,
        )
        cagr_after = wf_after.oos_cagr
        wfe_after  = wf_after.wfe
    except Exception as e:
        print(f"    Walk-forward (after) failed: {e}")
        _restore_filters(strategy_name, original)
        return wfe_before, None, False
    finally:
        _restore_filters(strategy_name, original)

    improved = (cagr_after or 0) > (cagr_before or 0)
    return wfe_before, wfe_after, improved


# ── Profile builder ───────────────────────────────────────────────────────────

def optimize_and_save(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    position_pct: float = 0.20,
    verify_wf: bool = True,
    min_trades: int = 8,
) -> SymbolFilterProfile | None:

    print(f"\n{'='*70}")
    print(f"  Optimizing: {strategy_name} / {symbol} / {period}")
    print(f"{'='*70}")

    # Step 1: extract snapshots
    print("  Step 1: Running backtest and extracting trade snapshots...")
    try:
        snapshots = extract_trade_snapshots(
            strategy_name, symbol, period=period, position_pct=position_pct,
        )
    except Exception as e:
        print(f"  ERROR extracting snapshots: {e}")
        return None

    if len(snapshots) < min_trades:
        print(f"  Skipping — only {len(snapshots)} trades (need ≥{min_trades})")
        return None

    wins = [s for s in snapshots if s["outcome"] == "win"]
    losses = [s for s in snapshots if s["outcome"] == "loss"]
    baseline_wr = len(wins) / len(snapshots) * 100
    print(f"  Trades: {len(snapshots)}  Wins: {len(wins)}  Baseline WR: {baseline_wr:.1f}%")

    # Step 2: feature analysis
    print("  Step 2: Analyzing feature distributions...")
    analysis = analyze_snapshots(snapshots)

    # Step 3: grid search best filters
    print("  Step 3: Grid-searching optimal thresholds...")
    filters = grid_search_filters(strategy_name, snapshots, analysis)

    if not filters:
        print("  No filters improved win rate — saving baseline profile.")

    # Step 4: walk-forward verification
    wfe_before = wfe_after = oos_before = oos_after = None
    verified = False

    if verify_wf and filters:
        print("  Step 4: Verifying on OOS walk-forward...")
        try:
            wfe_before, wfe_after, verified = verify_on_walkforward(
                strategy_name, symbol, filters, period=period,
            )
            print(f"    WFE before: {wfe_before}  WFE after: {wfe_after}  "
                  f"Improved: {verified}")
        except Exception as e:
            print(f"  Walk-forward verification failed: {e}")

    # Step 5: build and save profile using calibrate_from_snapshots
    print("  Step 5: Saving calibrated profile...")
    profile = calibrate_from_snapshots(strategy_name, symbol, snapshots)
    profile.wfe_before  = wfe_before
    profile.wfe_after   = wfe_after
    profile.verified    = verified

    save_profile(profile)
    print(f"  Profile saved: {strategy_name}/{symbol}  "
          f"WR={profile.win_rate_pct:.1f}%  verified={verified}")

    return profile


def main():
    parser = argparse.ArgumentParser(description="Perplexity filter optimizer")
    parser.add_argument("--strategy",  default=None)
    parser.add_argument("--symbol",    default=None)
    parser.add_argument("--period",    default="5y")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--no-verify", action="store_true", help="Skip walk-forward verification")
    args = parser.parse_args()

    verify = not args.no_verify

    if args.all:
        results = []
        for sname in STRATEGIES:
            for sym in DEFAULT_SYMBOLS:
                try:
                    p = optimize_and_save(sname, sym, period=args.period, verify_wf=verify)
                    if p:
                        results.append(p)
                except Exception as e:
                    print(f"  ERROR {sname}/{sym}: {e}")

        print(f"\n{'='*70}")
        print(f"  Optimization complete: {len(results)} profiles saved")
        print(f"  Verified (OOS improvement): {sum(1 for p in results if p.verified)}")
        verified = [p for p in results if p.verified]
        if verified:
            print(f"\n  Best verified profiles (by win rate):")
            for p in sorted(verified, key=lambda x: x.win_rate_pct, reverse=True)[:10]:
                print(f"    {p.strategy}/{p.symbol}: WR={p.win_rate_pct:.1f}%  "
                      f"trades={p.n_trades}  wfe_after={p.wfe_after}")
    else:
        if not args.strategy or not args.symbol:
            parser.error("Provide --strategy and --symbol, or use --all")
        if args.strategy not in STRATEGIES:
            parser.error(f"Unknown strategy. Choose from: {list(STRATEGIES)}")
        optimize_and_save(
            args.strategy, args.symbol.upper(),
            period=args.period, verify_wf=verify,
        )


if __name__ == "__main__":
    main()
