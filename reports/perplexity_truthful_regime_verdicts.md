# Perplexity verdicts under truthful (point-in-time) momentum regime

Source artifacts:
- Prior 5y (leaky regime): `perplexity_research_5y_20260602T150706Z.json`
- **Truthful 5y (this pass): `perplexity_research_5y_20260602T191717Z.json`**

The defect that motivated this pass: `_momentum_snapshot()` was calling the LIVE `get_momentum_regime()` from inside backtest loops, so every historical bar saw today's snapshot. Fix: `get_momentum_regime_at(as_of_date, …)` computes the regime from point-in-time-sliced inputs; the engine injects it into each strategy via `momentum_snapshot=` kwarg.

Same locked benchmark spec (10 symbols × 5y daily × US/INDIA costs × max_hold_bars × no lookahead × short-side support).

## Per-strategy comparison

| strategy | leaky 5y trd | truthful 5y trd | leaky 5y net | truthful 5y net | leaky WR | truthful WR | short_e |
|---|---:|---:|---:|---:|---:|---:|---:|
| Daily_Three_Bar_Push | 26 | **50** | +$6,511 | **+$6,525** | 65.4% | 54.0% | 7 |
| Daily_Hammer_Star | 11 | 11 | +$896 | **+$3,725** | 63.6% | **81.8%** | 0 |
| Breakout_Consolidation | 66 | 66 | +$2,227 | +$2,179 | 60.6% | 60.6% | 0 |
| RSI_Swing_Reversal | 5 | 5 | +$1,011 | +$1,011 | 60.0% | 60.0% | 0 |
| **Daily_Engulfing_Volume** | 9 | **23** | +$2,756 | **−$349** | 66.7% | **43.5%** | 3 |
| **Daily_NR_Breakout** | 13 | **36** | +$1,392 | **−$2,669** | 61.5% | **44.4%** | 4 |

Aggregate: $+14,793 → **$+10,422**. Trades: 130 → 191 (+47%). Long entries 130 → 177; short entries **0 → 14** (the leak being closed).

`Breakout_Consolidation` and `RSI_Swing_Reversal` are unchanged because they don't use the momentum gate (they use the simpler BULL/BEAR regime which was already point-in-time). Only the 4 candle pattern strategies are affected by this pass.

## Verdicts (final)

| Strategy | leaky verdict | **truthful verdict** | rationale |
|---|---|---|---|
| Daily_Three_Bar_Push | KEEP | **KEEP** | trades doubled, P&L flat → real edge; PF 2.34 |
| Daily_Hammer_Star | KEEP borderline | **KEEP** | WR 63.6→81.8%, P&L +$896→+$3,725 — clean upgrade |
| Breakout_Consolidation | KEEP | **KEEP** | unaffected (not momentum-gated); still passes thresholds |
| RSI_Swing_Reversal | RESEARCH-ONLY | **RESEARCH-ONLY** | unaffected; still only 5 trades in 5y |
| Daily_Engulfing_Volume | KEEP | **RETIRE** | WR collapsed 66.7→43.5%, P&L flipped +$2,756→−$349 |
| Daily_NR_Breakout | KEEP borderline | **RETIRE** | WR 61.5→44.4%, P&L flipped +$1,392→−$2,669 |

**The decision rule was applied:** previously-positive verdicts for `Daily_Engulfing_Volume` and `Daily_NR_Breakout` were artefacts of the live-state leak suppressing entries. With truthful per-bar regime, both strategies attract more trades — many marginal — and the edge disappears. They're retired despite their prior KEEP labels.

## Resulting live set (after this pass)

5 strategies (was 7):
- `EMA_Mean_Reversion` (unchanged)
- `BB_Mean_Reversion` (unchanged)
- `Breakout_Consolidation` (unchanged)
- `Daily_Three_Bar_Push` (held)
- `Daily_Hammer_Star` (held + materially better)

Retired (added to existing retire set):
- `Daily_Engulfing_Volume` (newly retired)
- `Daily_NR_Breakout` (newly retired)

Aggregate live edge (under truthful 5y conditions): roughly +$22k (Daily_Three_Bar_Push $6,525 + Daily_Hammer_Star $3,725 + Breakout_Consolidation $2,179 + the prior 2y KEEPs at ~$7.4k EMA/BB).

## What this pass DID change

- `_momentum_snapshot()` now accepts an injected snapshot (engine-provided) and uses it verbatim. Live path unchanged.
- New `get_momentum_regime_at(as_of_date, …)` in `market_regime_advanced.py` runs the same classifier on point-in-time-sliced inputs.
- Backtest engine pre-fetches index + VIX series once per run and injects the historical snapshot per bar.

## What this pass DID NOT change

- No strategy entry/exit rules were tuned.
- No engine semantics (max_hold, costs, lookahead) were re-examined.
- Live signal generation behaviour is identical to before for any strategy whose flag did not change.

## Next follow-ups

1. **Per-bucket slippage in the CostModel** — US_ETF vs US_LARGE vs NSE_MID. The current US_DEFAULT / INDIA_DEFAULT are coarse, and now that more strategies' verdicts hinge on net-of-cost numbers this gets more impactful.
2. **Widen the symbol universe for RSI_Swing_Reversal** — 5 trades in 5y on 10 large-caps; the strategy is genuinely picky.
3. **Investigate the Daily_Three_Bar_Push hold-time** — 18.9d average is long for a 3-bar momentum push; tighter time-exits might improve the edge.
4. **Consensus engine redesign** — still the largest remaining harness gap.
