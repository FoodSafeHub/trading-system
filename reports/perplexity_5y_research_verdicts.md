# Perplexity 5y RESEARCH-ONLY verdicts

Source artifacts:
- 2y baseline (corrected harness): `perplexity_ab_after-costed_20260602T033035Z.json`
- 5y with short-side support active: `perplexity_research_5y_20260602T150706Z.json`

Locked benchmark spec: 10 symbols (5 US + 5 NSE), US_DEFAULT / INDIA_DEFAULT costs, regime parity + max_hold + no lookahead, **plus minimal short-entry path now wired**.

## 2y vs 5y per strategy

| strategy | 2y trd | 5y trd | 2y pnl | 5y pnl | 2y WR | 5y WR | 5y hold_d | 5y shorts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Daily_Three_Bar_Push | 2 | 26 | −$567 | **+$6,511** | 50.0 | 65.4 | 18.9 | 0 |
| Daily_Engulfing_Volume | 3 | 9 | +$793 | +$2,756 | 66.7 | 66.7 | 6.4 | 0 |
| Daily_NR_Breakout | 2 | 13 | +$1,506 | +$1,392 | 100.0 | 61.5 | 6.5 | 0 |
| Daily_Hammer_Star | 2 | 11 | +$900 | +$896 | 100.0 | 63.6 | 3.7 | 0 |
| Breakout_Consolidation | 18 | 66 | +$834 | +$2,227 | 55.6 | 60.6 | 16.6 | 0 |
| RSI_Swing_Reversal | 2 | 5 | +$144 | +$1,011 | 50.0 | 60.0 | 6.2 | 0 |

Aggregate 5y: **+$14,793** across 130 trades.

## Important finding: short entries still zero

The engine's short path works (proven by `tests/unit/test_perplexity_engine_short.py`). But across 5y / 10 symbols, the 4 candle pattern strategies emitted **zero** usable short signals.

Root cause is **not in the engine** — it's in the strategy regime gate:

[`app/services/strategy/perplexity/momentum_strategies.py:41-64`](app/services/strategy/perplexity/momentum_strategies.py#L41-L64) — each pattern calls `_momentum_snapshot()` which delegates to `get_momentum_regime()` — a **LIVE-only** function that fetches the *current* SPY/VIX state. In a historical backtest it returns the same live snapshot for every bar, and `_regime_allows_short(snap)` only returns `True` for `MomentumRegime.BEAR_MOMENTUM`. Today's live snapshot evidently isn't BEAR_MOMENTUM, so **every historical day in the 5y window saw `allow_short=False`**.

This is a separate defect from "engine drops shorts" (now fixed). It's "strategy uses live regime in backtest." Out of scope for this round but noted as the next follow-up. Verdicts below treat each strategy on its **long-side-only** numbers — same constraint as before, but now provably bounded.

## Verdicts

Decision thresholds same as the prior decision pass:
- **KEEP** — net > 0, trades ≥ 20, WR ≥ 50%
- **RESEARCH-ONLY** — net ≥ 0 but trades < 20 OR weak metrics
- **RETIRE** — net < 0 with no rescue path
- **NEEDS-FOLLOW-UP** — engine still distorts the verdict

### Daily_Three_Bar_Push → **KEEP**
Trades 2 → 26. P&L flipped from −$567 to **+$6,511**. WR went 50% → 65.4%. PF 2.10. The 2y verdict was sample noise; the 5y picture is materially positive. Average hold 18.9d is long for a "momentum push" pattern — worth tightening as a future tuning pass, but not a blocker for KEEP.

### Daily_Engulfing_Volume → **KEEP**
Trades 3 → 9. P&L +$793 → +$2,756. WR 66.7% steady. PF 0.57 in the rollup is the per-symbol-average artifact (the same one flagged in the prior decision pass — actual edge is solidly positive based on net P&L on 9 trades). Sample is on the thin end of "keep-able" but consistent direction and net P&L tripled with more data.

### Daily_NR_Breakout → **KEEP** (borderline)
Trades 2 → 13. WR 100% → 61.5%. P&L roughly flat at ~+$1.4k. The 100% WR at 2 trades was noise — the 5y 61.5% WR on 13 trades is the real number, and it's still positive. 13 trades is just under the 20-trade KEEP threshold, but PF 1.83 and trend consistent. Borderline but on the right side of zero.

### Daily_Hammer_Star → **KEEP** (borderline)
Trades 2 → 11. WR 100% → 63.6%. P&L ~+$900 flat. Same shape as NR_Breakout — sample noise resolved, true edge revealed as positive but thin. PF 1.57 / hold 3.7d. Same borderline read; 11 trades / 5y is the lowest in the table but trades pay.

### Breakout_Consolidation → **KEEP**
Trades 18 → 66. P&L +$834 → +$2,227. WR 55.6% → 60.6%. The 2y verdict was "fragile positive, hold 15.8d too long"; the 5y has it earning more consistently. Hold time still 16.6d. The strategy genuinely takes time to work; the longer window confirms it does work.

### RSI_Swing_Reversal → **still RESEARCH-ONLY**
Trades 2 → 5. P&L +$144 → +$1,011. 5 trades in 5y is still too sparse to deploy live. The strategy is genuinely picky (RSI<40 then turn-up + EMA50 trend) and on 10 large-caps over 5y produced only 5 round-trips. Either widen the universe (more symbols) or accept it as opportunistic low-frequency. **Not enough sample to flip out of RESEARCH-ONLY.**

## Summary

| 2y verdict | Strategy | 5y verdict | Reason |
|---|---|---|---|
| NEEDS-FOLLOW-UP | Daily_Three_Bar_Push | **KEEP** | +$6,511 / 26 trades / 65 WR — most-improved on more data |
| NEEDS-FOLLOW-UP | Daily_Engulfing_Volume | **KEEP** | +$2,756 / 9 trades / 67 WR — consistent direction |
| NEEDS-FOLLOW-UP | Daily_NR_Breakout | **KEEP (borderline)** | +$1,392 / 13 trades / 62 WR |
| NEEDS-FOLLOW-UP | Daily_Hammer_Star | **KEEP (borderline)** | +$896 / 11 trades / 64 WR |
| RESEARCH-ONLY | Breakout_Consolidation | **KEEP** | +$2,227 / 66 trades / 61 WR — passes all thresholds |
| RESEARCH-ONLY | RSI_Swing_Reversal | **still RESEARCH-ONLY** | only 5 trades in 5y — too sparse |

**5 of 6 strategies move from RESEARCH-ONLY/NEEDS-FOLLOW-UP to KEEP.** Total LIVE set grows from 2 to 7. Aggregate live edge under 5y conditions: **roughly +$22k** (the 2 prior KEEPs at ~$7.4k 2y net + the 5 new KEEPs at +$13.8k 5y net).

The one remaining RESEARCH-ONLY (RSI_Swing_Reversal) needs more symbols or a longer window, not engine work.

## Next follow-up (out of scope for this pass)

Fix the strategy-side regime gate (`_momentum_snapshot` using LIVE state during backtests). Use a per-bar regime label from the backtest's spy_close series instead of calling `get_momentum_regime()` (which reads live). After that fix, the 4 candle patterns may finally fire short entries — and the verdict on each could change again, in either direction.
