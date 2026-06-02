# Perplexity strategy registry — current decision pass

**As of 2026-06-02 (5y re-evaluation).** Authoritative source artifacts:
- [`reports/perplexity_strategy_decisions.md`](../../../../reports/perplexity_strategy_decisions.md) — original 2y decision pass
- [`reports/perplexity_5y_research_verdicts.md`](../../../../reports/perplexity_5y_research_verdicts.md) — 5y re-evaluation that promoted 5 strategies

Two class-level flags control deployment:

| flag combo | meaning | runner behaviour |
|---|---|---|
| `enabled=True`, `research_only=False` | **LIVE** | included in `run_perplexity_signal` |
| `enabled=True`, `research_only=True` | **RESEARCH-ONLY** | skipped live; runs in backtest / walkforward |
| `enabled=False` | **RETIRE** | skipped live, but class stays importable |

The runner ([runner.py](runner.py)) skips both `not enabled` and `research_only`. The backtest engine ([../../backtest/perplexity_engine.py](../../backtest/perplexity_engine.py)) ignores both flags so research evaluation isn't blocked.

## Current classification

### KEEP (live)
From the 2y pass:
- `EMA_Mean_Reversion` — net +$4,549 / 76 trades / WR 61.8% (2y, costed)
- `BB_Mean_Reversion`  — net +$2,898 / 122 trades / WR 59% (2y, costed)

Promoted by the 5y re-evaluation:
- `Daily_Three_Bar_Push` — net +$6,511 / 26 trades / WR 65.4% (5y)
- `Daily_Engulfing_Volume` — net +$2,756 / 9 trades / WR 66.7% (5y)
- `Breakout_Consolidation` — net +$2,227 / 66 trades / WR 60.6% (5y)
- `Daily_NR_Breakout` — net +$1,392 / 13 trades / WR 61.5% (5y) — **borderline**
- `Daily_Hammer_Star` — net +$896 / 11 trades / WR 63.6% (5y) — **borderline**

### RESEARCH-ONLY (still too sparse to deploy)
- `RSI_Swing_Reversal` — net +$1,011 / **only 5 trades** in 5y / 10 symbols. Needs a wider universe before deciding.

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
