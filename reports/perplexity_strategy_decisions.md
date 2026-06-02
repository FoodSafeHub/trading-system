# Perplexity strategy decisions — corrected harness

Generated: `2026-06-02T14:20:59.392305+00:00`

Source artifacts:
- baseline (pre-fix harness): `perplexity_ab_baseline_20260602T025048Z.json`
- after-nocost (post-fix engine, costs OFF — GROSS): `perplexity_ab_after-nocost_20260602T030024Z.json`
- after-costed (post-fix engine + regime parity + costs ON — NET): `perplexity_ab_after-costed_20260602T033035Z.json`

Symbols (10): `AAPL, MSFT, NVDA, SPY, TSLA, RELIANCE, BHARTIARTL, INFY, TCS, HDFCBANK` — period `2y`

Decision rules (deterministic from the columns below):
- **KEEP** -- net > 0, trades >= 20, WR >= 50%
- **RESEARCH-ONLY** -- net >= 0 but trade count < 20 OR WR < 50% (positive but fragile)
- **NEEDS-FOLLOW-UP** -- net < 0 OR < 5 trades AND a known harness defect still distorts the verdict
  (the 4 daily-candlestick momentum patterns emit short SELLs the engine still drops)
- **RETIRE** -- net < 0 after all known corrections, no rescue path visible in current engine

**Note on PF**: the `pf` column comes from the benchmark's per-symbol PF average
which is biased upward when a symbol has no losing trades. Verified independently
for the two borderline cases (Breakout_Consolidation: true PF=1.17 not 0.65;
BB_Breakout: true PF=0.80 not 2.57). Decisions weight net pnl + trade count + WR
over the displayed PF.

| decision | strategy | fire | trd | wr% | pf | hold_d | gross | net | cost | Δ vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **KEEP** | EMA_Mean_Reversion | 9/10 | 76 | 61.8 | 1.82 | 5.0 | $+7,662 | $+4,549 | $3,113 | $-3,828 |
| **KEEP** | BB_Mean_Reversion | 10/10 | 122 | 59.0 | 1.40 | 5.7 | $+11,650 | $+2,898 | $8,753 | $-9,691 |
| **RESEARCH-ONLY** | Breakout_Consolidation | 8/10 | 18 | 55.6 | 0.65 | 15.8 | $+3,102 | $+834 | $2,268 | $+151 |
| **RESEARCH-ONLY** | RSI_Swing_Reversal | 2/10 | 2 | 50.0 | 0.00 | 8.0 | $+169 | $+144 | $24 | $-24 |
| **NEEDS-FOLLOW-UP** | Daily_NR_Breakout | 1/10 | 2 | 100.0 | 0.00 | 2.0 | $+1,524 | $+1,506 | $18 | $-18 |
| **NEEDS-FOLLOW-UP** | Daily_Hammer_Star | 2/10 | 2 | 100.0 | 0.00 | 4.5 | $+922 | $+900 | $22 | $+91 |
| **NEEDS-FOLLOW-UP** | Daily_Engulfing_Volume | 3/10 | 3 | 66.7 | 0.00 | 3.7 | $+822 | $+793 | $28 | $+593 |
| **NEEDS-FOLLOW-UP** | Daily_Three_Bar_Push | 2/10 | 2 | 50.0 | 0.00 | 6.0 | $-539 | $-567 | $28 | $-28 |
| **RETIRE** | Supertrend_Swing | 10/10 | 22 | 40.9 | 1.20 | 21.1 | $-1,213 | $-2,085 | $872 | $-838 |
| **RETIRE** | BB_Breakout | 9/10 | 43 | 48.8 | 2.57 | 7.3 | $-742 | $-2,329 | $1,586 | $-2,214 |
| **RETIRE** | MA_Crossover_RSI | 10/10 | 57 | 40.4 | 0.75 | 7.0 | $-6,524 | $-7,102 | $578 | $-413 |
| **RETIRE** | Fib_Pullback_Support | 10/10 | 97 | 47.4 | 0.97 | 6.6 | $-5,555 | $-11,793 | $6,237 | $-274 |

## Rationale by strategy

### EMA_Mean_Reversion — **KEEP**

net +$4,549 on 76 trades, WR 61.8% (PF~1.82); survived cost drag of $3,113

### BB_Mean_Reversion — **KEEP**

net +$2,898 on 122 trades, WR 59.0% (PF~1.40); survived cost drag of $8,753

### Breakout_Consolidation — **RESEARCH-ONLY**

net +$834 but only 18 trades -- positive edge is fragile; do not deploy without further validation

### RSI_Swing_Reversal — **RESEARCH-ONLY**

only 2 trades across 2/10 symbols (2y) -- too sparse to deploy or retire on; pnl=$+144 may be sample noise

### Daily_NR_Breakout — **NEEDS-FOLLOW-UP**

long-side only: net $+1,506 on 2 long trades. The strategy also emits short-side SELL entries that the engine currently drops (known harness defect). Re-evaluate after that fix.

### Daily_Hammer_Star — **NEEDS-FOLLOW-UP**

long-side only: net $+900 on 2 long trades. The strategy also emits short-side SELL entries that the engine currently drops (known harness defect). Re-evaluate after that fix.

### Daily_Engulfing_Volume — **NEEDS-FOLLOW-UP**

long-side only: net $+793 on 3 long trades. The strategy also emits short-side SELL entries that the engine currently drops (known harness defect). Re-evaluate after that fix.

### Daily_Three_Bar_Push — **NEEDS-FOLLOW-UP**

long-side only: net $-567 on 2 long trades. The strategy also emits short-side SELL entries that the engine currently drops (known harness defect). Re-evaluate after that fix.

### Supertrend_Swing — **RETIRE**

net $-2,085 on 22 trades, WR 40.9%, PF 1.20; cost drag $872; vs baseline Δ$-838 — persistently negative after corrections

### BB_Breakout — **RETIRE**

net $-2,329 on 43 trades, WR 48.8%, PF 2.57; cost drag $1,586; vs baseline Δ$-2,214 — persistently negative after corrections

### MA_Crossover_RSI — **RETIRE**

net $-7,102 on 57 trades, WR 40.4%, PF 0.75; cost drag $578; vs baseline Δ$-413 — persistently negative after corrections

### Fib_Pullback_Support — **RETIRE**

net $-11,793 on 97 trades, WR 47.4%, PF 0.97; cost drag $6,237; vs baseline Δ$-274 — persistently negative after corrections
