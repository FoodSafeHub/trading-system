# Perplexity engine — round 2 (regime parity + portfolio fixes)

Same locked benchmark spec as round 1 (10 symbols × 2y × 12 strategies × identical injected data × US_DEFAULT + INDIA_DEFAULT cost models). The only code changes between this and round 1's `after-costed` are:

- `_regime_from_spy` now delegates to the live `detect_market_regime` (BULL gate now requires SMA(50) ≥ SMA(200); DEEP_BEAR now uses 52w-DD).
- `portfolio_engine` gained `max_hold_bars` + `cost_model` (not exercised by the perplexity_engine benchmark — separately covered by its targeted tests).

## Aggregate

| metric | round 1 after-costed (pre-regime-fix) | round 2 after-costed (regime-fix) | Δ |
|---|---:|---:|---:|
| Total P&L | −$13,681 | **−$12,252** | +$1,429 |
| Trades | 458 | **446** | −12 |
| Wins | 241 | **236** | −5 |
| Losses | 217 | **210** | −7 |

The regime fix removed ~12 trades net — these are days the OLD backtest classifier said BULL (so BULL-gated strategies fired) but the LIVE classifier (now both) said BEAR (so live wouldn't have fired). Most of those were marginal/losing, hence aggregate P&L improved by +$1,429.

## Per-strategy delta (regime fix only)

| strategy | r1 trades | r2 trades | Δ trades | r1 pnl | r2 pnl | Δ pnl |
|---|---:|---:|---:|---:|---:|---:|
| BB_Mean_Reversion | 122 | 122 | 0 | +$4,114 | +$2,898 | −$1,216 |
| EMA_Mean_Reversion | 80 | 76 | −4 | +$3,601 | +$4,549 | +$948 |
| Breakout_Consolidation | 21 | 18 | −3 | +$2,145 | +$834 | −$1,311 |
| Daily_NR_Breakout | 2 | 2 | 0 | +$1,506 | +$1,506 | 0 |
| Daily_Hammer_Star | 2 | 2 | 0 | +$900 | +$900 | 0 |
| Daily_Engulfing_Volume | 3 | 3 | 0 | +$794 | +$793 | −$1 |
| RSI_Swing_Reversal | 2 | 2 | 0 | +$144 | +$144 | 0 |
| Daily_Three_Bar_Push | 2 | 2 | 0 | −$567 | −$567 | 0 |
| Supertrend_Swing | 22 | 22 | 0 | −$2,085 | −$2,085 | 0 |
| BB_Breakout | 43 | 43 | 0 | −$2,856 | −$2,329 | +$527 |
| **MA_Crossover_RSI** | 64 | **57** | **−7** | −$9,860 | **−$7,102** | **+$2,758** |
| Fib_Pullback_Support | 97 | 97 | 0 | −$11,515 | −$11,793 | −$278 |

`MA_Crossover_RSI` is the biggest mover: 7 fewer trades, +$2,758 P&L. That makes sense — it's gated `if regime != BULL: HOLD`, and the old backtest mislabel let it trade through several "death cross" windows that the live engine would have blocked. Removing those trades (which were net-losing on the bad regime read) is exactly the harness becoming more truthful.

`Breakout_Consolidation` and `EMA_Mean_Reversion` are also BULL-only and lost a handful of trades; their P&L went different directions because the specific trades removed happened to be a mix of winners (Breakout) and losers (EMA).

`BB_Mean_Reversion` is "BULL or mild BEAR" — fewer days were classified BULL, but it can still run in mild BEAR. Net trade count unchanged (122 → 122). Some trade fills shifted because of the different regime context affecting the strategy's internal gates.

## Behavior change classification

| change | direction | explanation |
|---|---|---|
| BULL → BEAR reclassifications | net −12 trades | death-cross windows (close > SMA200 but SMA50 < SMA200) no longer pass live's BULL gate |
| Strategy ranking | unchanged | top-2 still EMA/BB mean-reversion; bottom-2 still Fib/MA-crossover |
| Strategy fire-rate (fire/10) | unchanged for all | regime fix shifts which days fire, not which symbols |
| Win-rate movement | ≤2pp | adjusting *which* days fire, not the engine math |
| Honesty | strictly improved | backtest can no longer label a day BULL when live would say BEAR |

No regressions in trade-set diversity (no strategy went from firing to silent). No surprising sign flips. The aggregate moved in the expected direction (truthier classifier → fewer phantom BULL trades → smaller drawdown from bad regime calls).

## Conclusion

**ACCEPT** the regime delegation and **ACCEPT** the portfolio_engine fixes. Detailed recommendations in the final report.
