# Perplexity strategy registry — current decision pass

**As of 2026-06-02.** Authoritative source artifact:
[`reports/perplexity_strategy_decisions.md`](../../../../reports/perplexity_strategy_decisions.md).

Two class-level flags control deployment:

| flag combo | meaning | runner behaviour |
|---|---|---|
| `enabled=True`, `research_only=False` | **LIVE** | included in `run_perplexity_signal` |
| `enabled=True`, `research_only=True` | **RESEARCH-ONLY** | skipped live; runs in backtest / walkforward |
| `enabled=False` | **RETIRE** | skipped live, but class stays importable |

The runner ([runner.py](runner.py)) skips both `not enabled` and `research_only`. The backtest engine ([../../backtest/perplexity_engine.py](../../backtest/perplexity_engine.py)) ignores both flags so research evaluation isn't blocked.

## Current classification

### KEEP (live)
- `EMA_Mean_Reversion` — net +$4,549 / 76 trades / WR 61.8% (2y / 10 symbols, costed)
- `BB_Mean_Reversion`  — net +$2,898 / 122 trades / WR 59% (2y / 10 symbols, costed)

### RESEARCH-ONLY (held for research; positive but fragile)
- `Breakout_Consolidation` — net +$834 / 18 trades / WR 55.6%; hold 15.8d (too long)
- `RSI_Swing_Reversal` — only 2 trades / 2y (sample too small)

### NEEDS-FOLLOW-UP (engine still distorts the verdict)

All 4 daily candlestick momentum patterns emit short-side `direction="SELL"` entries that the perplexity engine currently drops (long-only execution path). Long-side alone is half the strategy's edge surface — verdict deferred until the engine grows a short-entry path **or** these strategies stop emitting SELL for bearish patterns.

- `Daily_Engulfing_Volume`
- `Daily_NR_Breakout`
- `Daily_Three_Bar_Push`
- `Daily_Hammer_Star`

### RETIRE (off; code preserved for future re-enablement)
- `Supertrend_Swing` — net −$2,085 / 22 trades / WR 41%
- `BB_Breakout`     — net −$2,329 / 43 trades / WR 49% (true PF=0.80)
- `MA_Crossover_RSI` — net −$7,102 / 57 trades / WR 40%
- `Fib_Pullback_Support` — net −$11,793 / 97 trades / WR 47% (worst performer)

## How to re-evaluate or re-enable a strategy

1. Read the per-class comment block on the class definition — it lists the specific condition for re-enablement (e.g. "requires entry-logic rework", "bump max_hold_bars down then re-benchmark").
2. Run a confirmatory benchmark with the corrected harness:
   ```bash
   python scripts/perplexity_ab_benchmark.py --mode after-costed
   python scripts/perplexity_strategy_decisions.py
   ```
3. If the artifact justifies it, flip the relevant flag (`enabled` or `research_only`) on the class, update [`tests/unit/test_perplexity_registry_decisions.py`](../../../../tests/unit/test_perplexity_registry_decisions.py) sets, and commit referencing the new artifact.

## How the live set was chosen

The KEEP set is the strategies that pass **all four** of:
- net P&L > 0 after `US_DEFAULT` / `INDIA_DEFAULT` costs
- ≥ 20 trades over the 2y / 10-symbol benchmark
- ≥ 50% win-rate
- consistent fire pattern across symbols (`fire ≥ 8/10`)

RETIRE is net-negative after corrections with no straightforward rescue under the current engine. RESEARCH-ONLY is positive but fails one of the sample-size / WR thresholds, **or** has a known harness defect that prevents a fair verdict.

The corrected harness changes (regime parity, max_hold_bars, costs, no same-bar lookahead) are the basis for these numbers — see [`reports/perplexity_ab_COMPARISON.md`](../../../../reports/perplexity_ab_COMPARISON.md) and [`reports/perplexity_regime_AB_round2.md`](../../../../reports/perplexity_regime_AB_round2.md) for the harness validation.
