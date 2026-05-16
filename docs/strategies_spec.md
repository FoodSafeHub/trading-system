# Day-Trading Strategy Specifications

**Single source of truth** for all seven intraday strategies.
Every platform consuming these rules — Python code, Webull/Schwab plugins, and
Perplexity/Claude research prompts — should reference this document.

**Code location:** `app/services/strategy/daytrading/strategies/`  
**Regime logic:** `app/services/strategy/daytrading/market_open.py`  
**Auto-tuning:** `app/services/strategy/daytrading/brain/config_adjuster.py`

---

## Market Regimes

All strategies receive a regime label each trading day. The regime is computed
once per day from the 5m data and controls which strategies are allowed to fire.

| Regime | Definition | Strategies allowed |
|--------|------------|-------------------|
| `BULL_OPEN` | Opened above prior close **and** current price > VWAP | All 7 |
| `BEAR_OPEN` | Opened below prior close **and** current price < VWAP | EMAMomentum, OpeningGapFade, VolumeSpikeReversal only |
| `CHOPPY` | Neither BULL nor BEAR condition clearly met | All 7 (confidence penalised −0.20) |

**Hard cutoffs (all strategies):**
- No new entries after **3:15 PM ET** (global hard stop: `LAST_ENTRY_TIME = 15:15`)
- Each strategy has its own earlier cutoff — see per-strategy details below.

---

## Strategy 1 — ORBBreakout

**File:** `strategies/orb_breakout.py`  
**Timeframe:** 5m  
**Typical trades/day:** 0.5–1 per symbol  
**Best regime:** BULL_OPEN  
**Avoid:** BEAR_OPEN (blocked by regime filter); CHOPPY (allowed but penalised)

### Concept
Define the opening range (ORB) as the High/Low of the first 15 minutes (3 × 5m bars).
Enter when price **closes** above the ORB high by at least a 0.1% buffer.
The buffer eliminates the majority of fakeout entries.

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `orb_minutes` | 15 | Opening range window in minutes (= 3 bars on 5m) |
| `entry_buffer_pct` | 0.10 | Close must be ≥ 0.10% above ORB high |
| `vol_multiple` | 1.8 | Volume must be ≥ 1.8× the 20-bar rolling average |
| `tp_multiplier` | 2.0 | Target = entry + ORB height × 2.0 |
| `atr_stop_mult` | 1.0 | Stop = entry − ATR(14) × 1.0 |
| `rsi_period` | 14 | RSI lookback |
| `rsi_min` | 52 | RSI must be ≥ 52 (above 50 = confirmed momentum) |
| `max_orb_atr_ratio` | 2.5 | Skip if ORB height > 2.5 × (orb_bars × ATR) — too wide |
| `min_rr` | 2.0 | Minimum risk:reward required to take the trade |
| `max_hold_bars` | 48 | Maximum hold = 48 bars = 4 hours |

### LONG Entry (only direction — no shorts)
All conditions must be true on the same 5m bar:

1. Regime is **not** BEAR_OPEN
2. At least 6 bars of today's data available
3. Bar time is before **11:30 AM ET** (`ORB_LAST_ENTRY`)
4. `close > orb_high × (1 + entry_buffer_pct / 100)`
5. `volume ≥ vol_multiple × 20-bar avg volume`
6. `RSI(14) ≥ rsi_min` (52)
7. `ORB height ≤ max_orb_atr_ratio × (orb_bars × ATR)` — range must be tradeable
8. Previous bar's low > bar before that's low (higher-low structure — no lower low into breakout)

### Stop Placement
```
stop = entry − ATR(14) × atr_stop_mult
stop = max(stop, orb_low)   ← never let stop go above ORB low
```

### Target
```
target = entry + orb_height × tp_multiplier
```

### Trailing / Exit
- No trailing stop in backtest; exit at target, stop, or max_hold_bars (EOD if none hit)
- Auto trader trails with ATR-based trail once trade is +1.5R (regime-adjusted)

### When to Avoid
- BEAR_OPEN days (regime filter blocks it)
- Very wide opening ranges (> 2.5× expected ATR range)
- After 11:30 AM ET
- Low volume open (< 1.8× avg)
- Lower-low structure into the breakout (momentum exhaustion)

---

## Strategy 2 — VWAPMeanReversion

**File:** `strategies/vwap_mean_reversion.py`  
**Timeframe:** 5m  
**Typical trades/day:** 0.5–2 per symbol  
**Best regime:** BULL_OPEN only (hard filter — returns nothing in other regimes)  
**Avoid:** Any trending day where price never returns to VWAP

### Concept
Fade a pullback below VWAP when RSI is oversold, the bar has a lower-wick
reversal pattern, and volume confirms buyers are stepping in.

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `vwap_atr_distance` | 0.4 | Price must be ≥ 0.4 × ATR below VWAP |
| `rsi_oversold` | 38 | RSI(14) must be below this |
| `wick_ratio_min` | 0.40 | Lower wick > 40% of bar range |
| `vol_ratio_min` | 1.3 | Volume ≥ 1.3× 20-bar avg |
| `atr_stop_mult` | 1.0 | Stop = entry − ATR × 1.0 |
| `atr_target_mult` | 1.8 | Target = VWAP + ATR × 1.8 × 0.5 (past VWAP) |
| `max_hold_bars` | 20 | ~100 minutes max hold |
| `min_rr` | 1.5 | Minimum R:R |
| `max_vwap_drop_pct` | 0.05 | Reject if VWAP dropped > 0.05% in last 3 bars (trending down) |

### LONG Entry (only direction)
Time window: **9:45 AM – 2:30 PM ET**

1. Regime = BULL_OPEN (hard requirement)
2. `(VWAP − close) / ATR ≥ vwap_atr_distance` (0.4)
3. `RSI(14) < rsi_oversold` (38)
4. Entry bar closes **green** (close > open)
5. Lower wick ratio > `wick_ratio_min` (40%)
6. `volume ≥ vol_ratio_min × avg_vol` (1.3×)
7. Previous bar also below VWAP
8. VWAP not in freefall (< 0.05% drop in last 3 bars)

### Stop / Target
```
stop   = entry − ATR × atr_stop_mult
target = VWAP + ATR × atr_target_mult × 0.5
```

---

## Strategy 3 — EMAMomentum

**File:** `strategies/ema_momentum.py`  
**Timeframe:** 15m  
**Typical trades/day:** 1–3 per symbol  
**Best regime:** BULL_OPEN, CHOPPY; shorts in BEAR_OPEN  
**Avoid:** First 30 minutes of session; after 2:00 PM ET

### Concept
Two setups: (A) fresh EMA 9/21 crossover with MACD confirmation, or (B) EMA9
bounce on an already-bullish EMA stack (more frequent; primary driver).

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ema_fast` | 9 | Fast EMA |
| `ema_slow` | 21 | Slow EMA |
| `macd_fast/slow/signal` | 12/26/9 | Standard MACD |
| `rsi_low` | 40 | RSI lower bound for long entries |
| `rsi_high` | 70 | RSI upper bound |
| `atr_stop_mult` | 1.2 | Stop = EMA9 − ATR × 1.2 |
| `atr_tp_mult` | 2.5 | Target = entry + ATR × 2.5 |
| `min_rr` | 1.8 | |
| `max_hold_bars` | 16 | 16 × 15m = 4 hours |
| `ema9_bounce_atr` | 0.3 | Bar low must come within 0.3 × ATR of EMA9 for bounce setup |

### LONG Entry (Setup A — Crossover)
1. EMA9 crossed above EMA21 this bar (was below last bar)
2. MACD histogram > 0
3. RSI 40–70
4. Close > VWAP
5. Time: 10:00 AM – 2:00 PM ET

### LONG Entry (Setup B — EMA9 Bounce, more common)
1. EMA9 already > EMA21 (aligned bullish)
2. Bar low ≤ EMA9 + 0.3 × ATR (pullback touched EMA9)
3. Close > EMA9
4. Previous bar also closed above EMA9
5. MACD histogram > 0
6. RSI 40–70, close > VWAP

### SHORT Entry (BEAR_OPEN only — mirrors of A and B)
- Mirror of above with EMA9 < EMA21, MACD < 0, RSI 30–60, close < VWAP

### Stop / Target
```
Long:  stop = EMA9 − ATR × atr_stop_mult;  target = entry + ATR × atr_tp_mult
Short: stop = EMA9 + ATR × atr_stop_mult;  target = entry − ATR × atr_tp_mult
```

---

## Strategy 4 — OpeningGapFade

**File:** `strategies/opening_gap_fade.py`  
**Timeframe:** 15m  
**Typical trades/day:** 0–1 (gap days only)  
**Best regime:** All (works in any regime)  
**Avoid:** Earnings, Fed days, macro catalyst gaps, gaps > 2.5%

### Concept
Fade overextended gaps (0.5–2.5%) that lack news volume. Statistical edge:
> 60% of non-news gaps in this range partially fill within 90 minutes.

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `gap_min_pct` | 0.50 | Minimum gap size |
| `gap_max_pct` | 2.5 | Maximum — larger gaps are often gap-and-go |
| `rsi_overbought` | 62 | For gap-up fade entries |
| `rsi_oversold` | 38 | For gap-down fade entries |
| `max_vol_ratio` | 1.8 | Skip if opening volume > 1.8× daily avg / 26 bars |
| `target_fill_pct` | 0.60 | Target = 60% gap fill toward prior close |
| `stop_beyond_pct` | 0.35 | Stop = 0.35% beyond opening price |
| `min_body_pct` | 0.40 | First 15m bar body > 40% of range (real direction) |
| `min_rr` | 1.5 | |

### Entry (Gap-Up Fade → SHORT)
1. Gap up 0.5–2.5%
2. First 15m bar closes **below** its open (bearish)
3. RSI(14) of second 15m bar > 62
4. Volume < 1.8× avg (no news catalyst)
5. Body of first bar > 40% of range
6. 5m confirmation: a bearish 5m candle in 9:45–10:30 window (if available)
7. Before 10:30 AM ET

### Entry (Gap-Down Fade → LONG)
Mirror: gap down 0.5–2.5%, first bar bullish, RSI < 38, same volume/body checks.

### Stop / Target
```
Gap-up:   entry = first 15m close (or 5m confirm bar close)
          stop   = opening_price × (1 + 0.35%)
          target = opening_price − gap_size × 0.60  (60% fill)

Gap-down: entry = first 15m close
          stop   = opening_price × (1 − 0.35%)
          target = opening_price + gap_size × 0.60
```

---

## Strategy 5 — VolumeSpikeReversal

**File:** `strategies/volume_spike_reversal.py`  
**Timeframe:** 5m (with 15m RSI confirmation)  
**Typical trades/day:** 0.5–2 per symbol  
**Best regime:** All regimes (longs always; shorts in BEAR_OPEN only)  
**Avoid:** Last 45 minutes (end-of-day volume spikes are rebalancing, not reversals)

### Concept
Catch capitulation bottoms (or exhaustion tops) identified by an extreme volume
spike at RSI extremes with a rejection wick candle.

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `spike_multiple` | 2.5 | Volume must be ≥ 2.5× 20-bar avg |
| `rsi_max` | 38 | For BUY: RSI(14,5m) must be < 38 |
| `rsi_min_short` | 62 | For SELL: RSI > 62 |
| `decline_pct` | 0.4 | Price must have moved ≥ 0.4% in last 3 bars |
| `wick_ratio_min` | 0.35 | Rejection wick > 35% of bar range |
| `min_bar_atr_mult` | 0.6 | Bar range > 0.6 × ATR (not a doji) |
| `atr_stop_mult` | 1.0 | |
| `atr_tp_mult` | 2.0 | |
| `require_largest_vol` | True | Must be peak volume bar in last 10 bars |
| `max_hold_bars` | 10 | Scalp — 50 minutes max |
| `min_rr` | 1.6 | |

### BUY Entry (all regimes)
1. Time: 9:45 AM – 3:00 PM ET
2. Volume spike ≥ 2.5× avg AND is the peak of last 10 bars
3. Bar range > 0.6 × ATR
4. RSI(5m) < 38
5. Price declined ≥ 0.4% in prior 3 bars
6. Lower wick > 35% of bar range
7. 15m RSI < 55 and not still sharply deteriorating

### SHORT Entry (BEAR_OPEN only)
Mirror: RSI > 62, price rose ≥ 0.4% in 3 bars, upper wick > 35%.

### Stop / Target
```
stop   = entry ± ATR × atr_stop_mult
target = entry ± ATR × atr_tp_mult
```

---

## Strategy 6 — BollingerMomentum

**File:** `strategies/bollinger_momentum.py`  
**Code reference:** See `default_config` in `BollingerMomentum` class  
**Timeframe:** 5m  
**Typical trades/day:** 0–1 per symbol  
**Best regime:** BULL_OPEN (longs), BEAR_OPEN (shorts); CHOPPY allowed but penalised −0.05  
**Avoid:** No active squeeze (bands never contracted); after 2:30 PM ET

### Concept
Bollinger Bands contract during low-volatility coiling periods. When band width
falls into its lowest 20% of the last 40 bars (the "squeeze"), the market is
building energy for a directional breakout. A decisive close **outside** the band
with EMA slope, RSI momentum, and volume confirmation signals the release.

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `bb_length` | 20 | BB SMA window |
| `bb_std` | 2.0 | Standard deviation multiplier |
| `contraction_lookback` | 40 | Bars of history to define squeeze context |
| `contraction_percentile` | 0.20 | Band width must be in bottom 20% of lookback |
| `ema_fast` | 9 | EMA for slope/trend direction filter |
| `rsi_period` | 14 | RSI lookback |
| `rsi_min_long` | 50 | RSI lower bound for longs (momentum confirmed) |
| `rsi_max_long` | 70 | RSI upper bound for longs (not overbought) |
| `rsi_min_short` | 30 | RSI lower bound for shorts |
| `rsi_max_short` | 50 | RSI upper bound for shorts |
| `vol_rel_min` | 1.2 | Volume ≥ 1.2× 20-bar avg |
| `atr_stop_mult` | 1.0 | ATR stop multiplier (secondary stop floor) |
| `r_multiple_target` | 2.0 | Target = entry + risk × 2.0 |
| `max_hold_bars` | 60 | 5 hours max |

**Band width definition:**
```
bb_width = (bb_upper − bb_lower) / bb_mid
```
Squeeze is active when `bb_width ≤ quantile(contraction_percentile)` of the last
`contraction_lookback` bars.

### LONG Entry
All conditions must be true simultaneously on a 5m bar before 2:30 PM ET:

1. `bb_width ≤ squeeze_threshold` (currently in lowest 20% of last 40 bars)
2. `close > bb_upper` — decisive close above upper band
3. `EMA9 > EMA9[prior bar]` — EMA slope rising
4. `rsi_min_long ≤ RSI(14) ≤ rsi_max_long` (50–70)
5. `volume ≥ vol_rel_min × 20-bar avg volume` (1.2×)
6. `close > VWAP` — aligned with intraday institutional bias

### SHORT Entry
Mirror image:

1. Active squeeze (same condition)
2. `close < bb_lower` — close below lower band
3. `EMA9 < EMA9[prior bar]` — EMA slope falling
4. `rsi_min_short ≤ RSI(14) ≤ rsi_max_short` (30–50)
5. `volume ≥ 1.2× avg`
6. `close < VWAP`

### Stop Placement
```
Long:
  stop_candidate_1 = bar_low − 0.25 × ATR(14)
  stop_candidate_2 = entry − ATR(14) × atr_stop_mult
  stop = min(stop_candidate_1, stop_candidate_2)   ← use the tighter of the two

Short:
  stop_candidate_1 = bar_high + 0.25 × ATR(14)
  stop_candidate_2 = entry + ATR(14) × atr_stop_mult
  stop = max(stop_candidate_1, stop_candidate_2)
```

### Target
```
risk   = |entry − stop|
target = entry + risk × r_multiple_target   (Long)
target = entry − risk × r_multiple_target   (Short)
```
Minimum R:R enforced: **1.5** (rejects trade if R:R < 1.5).

### Trailing / Exit
- Backtest: exit at target, stop, or EOD (no trailing)
- Auto trader: ATR trail after +2.0R for TREND regimes; candle trail in CHOPPY

### Max Hold
60 bars = 5 hours (unlikely to be needed intraday)

### When to Avoid
- No squeeze present (band width never in bottom 20%)
- After 2:30 PM ET (`BB_LAST_ENTRY`)
- BEAR_OPEN day with a long setup (regime scoring will penalise −0.15)
- Earnings or macro news day (gap opens destroy the squeeze setup)
- Very thin volume (< 1.2× avg often signals pre/post-market noise)

### Auto-Tuning (config_adjuster.py)
| Profile | Changes |
|---------|---------|
| Low volatility | `contraction_percentile → 0.30`, `vol_rel_min → 1.0`, `r_multiple_target → 1.5` |
| High volatility | `contraction_percentile → 0.15`, `r_multiple_target → 2.5`, `atr_stop_mult → 1.3` |

---

## Strategy 7 — SupertrendTrend

**File:** `strategies/supertrend_trend.py`  
**Code reference:** See `default_config` in `SupertrendTrend` class  
**Timeframes:** 5m (execution) + 15m (macro direction filter)  
**Typical trades/day:** 0–2 per symbol  
**Best regime:** BULL_OPEN (longs), BEAR_OPEN (shorts)  
**Avoid:** CHOPPY days (confidence penalised −0.10); after 3:00 PM ET

### Concept
The 15m Supertrend establishes the macro directional bias for the session.
Once the bias is confirmed, wait for a 5m pullback to the 5m Supertrend
line or EMA20, then enter on a reclaim candle. This gives a tighter stop
and better R:R than chasing the breakout.

### Supertrend Formula
Supertrend is not available in the `ta` library; it is implemented manually
using Wilder's ATR smoothing:

```
HL2    = (High + Low) / 2
Upper  = HL2 + multiplier × ATR(length)   ← initial upper band
Lower  = HL2 − multiplier × ATR(length)   ← initial lower band

Band adjustment (per bar):
  Upper[i] = min(Upper[i], Upper[i-1])  if Close[i-1] <= Upper[i-1]
  Lower[i] = max(Lower[i], Lower[i-1]) if Close[i-1] >= Lower[i-1]

Direction:
  if prior ST was on upper band:
    if Close <= Upper → ST = Upper, direction = BEARISH (−1)
    else              → ST = Lower, direction = BULLISH (+1)
  else:
    if Close >= Lower → ST = Lower, direction = BULLISH (+1)
    else              → ST = Upper, direction = BEARISH (−1)
```

ATR uses **Wilder's smoothing** (not simple rolling):
```
ATR[0..length-1] = simple average of TR
ATR[j] = (ATR[j-1] × (length − 1) + TR[j]) / length
```

### Indicators & Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `st_length` | 10 | ATR period for Supertrend calculation |
| `st_multiplier` | 3.0 | Band width multiplier |
| `ema_pullback` | 20 | EMA period for secondary pullback level |
| `rsi_period` | 14 | |
| `rsi_min_long` | 45 | RSI lower bound for longs |
| `rsi_max_long` | 70 | RSI upper bound for longs |
| `rsi_min_short` | 30 | |
| `rsi_max_short` | 55 | |
| `vol_rel_min` | 1.1 | Volume ≥ 1.1× 20-bar avg |
| `atr_stop_mult` | 1.5 | Stop = ST_line − 0.1×ATR, floored at entry − 1.5×ATR |
| `r_multiple_target` | 2.0 | Target = entry + risk × 2.0 |
| `pullback_atr_dist` | 0.5 | How close to ST/EMA the bar low must come (in ATR units) |
| `max_hold_bars` | 80 | ~6.7 hours — strategy is a trend follower |

### 15m Macro Direction
The 15m Supertrend is computed on today's 15m bars (same `st_length=10`, `st_multiplier=3.0`):
- `direction == +1` → macro bullish → LONG setups allowed
- `direction == −1` → macro bearish → SHORT setups allowed
- If < `st_length + 2` 15m bars available, fall back to regime:
  BULL_OPEN → macro bullish, BEAR_OPEN → macro bearish

### LONG Entry
All conditions on a single 5m bar, before 3:00 PM ET:

1. 15m Supertrend direction = **+1** (macro bullish) — or BULL_OPEN fallback
2. 5m Supertrend direction = **+1** (local trend also bullish)
3. `rsi_min_long ≤ RSI(14) ≤ rsi_max_long` (45–70)
4. `volume ≥ vol_rel_min × 20-bar avg` (1.1×)
5. Bar is a **bullish reclaim candle**: `close > open`
6. **Pullback condition** (either one):
   - `|bar_low − ST_line_5m| ≤ pullback_atr_dist × ATR` (bar low touched near ST line)
   - `|bar_low − EMA20| ≤ pullback_atr_dist × ATR` (bar low touched near EMA20)

### SHORT Entry
Mirror:

1. 15m ST direction = **−1** (macro bearish) — or BEAR_OPEN fallback
2. 5m ST direction = **−1**
3. `rsi_min_short ≤ RSI(14) ≤ rsi_max_short` (30–55)
4. `volume ≥ 1.1× avg`
5. Bar is **bearish**: `close < open`
6. Pullback: `|bar_high − ST_line_5m| ≤ pullback_atr_dist × ATR` or `|bar_high − EMA20| ≤ pullback_atr_dist × ATR`

### Stop Placement
```
Long:
  stop = ST_line_5m − 0.1 × ATR        ← just below ST line
  stop = min(stop, entry − atr_stop_mult × ATR)   ← floor at 1.5×ATR below entry

Short:
  stop = ST_line_5m + 0.1 × ATR
  stop = max(stop, entry + atr_stop_mult × ATR)
```

### Target
```
risk   = |entry − stop|
target = entry + risk × r_multiple_target   (2.0R)
```
Minimum R:R enforced: **1.5**.

### Trailing Exit
The natural trailing mechanism for this strategy is to **follow the 5m Supertrend
line** — exit when the 5m ST flips direction against the trade. This is
implemented in the auto trader's exit manager; in backtests the simpler
target/stop model is used.

### Max Hold
80 bars = ~6.7 hours (designed for full-day trend following; EOD exit if open)

### When to Avoid
- CHOPPY regime (15m and 5m will both be choppy — ST flips frequently, signals
  degrade to noise)
- After 3:00 PM ET (`ST_LAST_ENTRY`)
- Less than `st_length + ema_pullback + 5` bars in today's 5m data (indicators
  not yet reliable)
- Macro and 5m ST contradict each other (e.g., 15m bullish but 5m bearish — skip)

### Auto-Tuning (config_adjuster.py)
| Profile | Changes |
|---------|---------|
| Low volatility | `st_multiplier → 2.0`, `pullback_atr_dist → 0.3`, `atr_stop_mult → 1.2` |
| High volatility | `st_multiplier → 3.5`, `pullback_atr_dist → 0.8`, `r_multiple_target → 2.5` |

---

## Cross-Strategy Summary Table

| Strategy | TF | BULL | BEAR | CHOPPY | Entry cutoff | Max hold | Min R:R |
|----------|-----|------|------|--------|-------------|----------|---------|
| ORBBreakout | 5m | ✅ Best | ❌ Blocked | ⚠️ | 11:30 AM | 4 h | 2.0 |
| VWAPMeanReversion | 5m | ✅ Only | ❌ | ❌ | 2:30 PM | 1.7 h | 1.5 |
| EMAMomentum | 15m | ✅ | ✅ (shorts) | ⚠️ | 2:00 PM | 4 h | 1.8 |
| OpeningGapFade | 15m | ✅ | ✅ | ✅ | 10:30 AM | — | 1.5 |
| VolumeSpikeReversal | 5m | ✅ | ✅ (shorts) | ✅ | 3:00 PM | 50 min | 1.6 |
| **BollingerMomentum** | **5m** | **✅ Best** | **✅ (shorts)** | **⚠️** | **2:30 PM** | **5 h** | **2.0** |
| **SupertrendTrend** | **5m+15m** | **✅ Best** | **✅ Best** | **❌ Avoid** | **3:00 PM** | **6.7 h** | **2.0** |

---

## Perplexity / Claude Research Prompts

Copy-paste any section below directly into Perplexity or Claude.

---

### Prompt — Analyse BollingerMomentum

```
I'm using the following intraday trading strategy called BollingerMomentum.
Please analyse, critique, or scan based on these exact rules.

STRATEGY: BollingerMomentum (5m timeframe, US equities, intraday only)

INDICATORS:
- Bollinger Bands: SMA(20), ±2.0 standard deviations
- Band width = (upper − lower) / middle — normalised squeeze measure
- EMA(9) — slope filter only (rising = bullish, falling = bearish)
- RSI(14)
- ATR(14) — for stop placement
- VWAP (intraday, resets at session open) — directional filter
- Relative volume = current bar volume / 20-bar rolling avg volume

SQUEEZE DEFINITION:
- Look back 40 bars on the 5m chart
- A squeeze is active when the current band width is in the lowest 20th
  percentile of those 40 bars (i.e., the bands are abnormally tight)

LONG ENTRY — all conditions required simultaneously:
1. Active squeeze (band width in lowest 20% of last 40 bars)
2. Current 5m bar closes ABOVE the upper Bollinger Band
3. EMA(9) is rising (current > prior bar)
4. RSI(14) between 50 and 70
5. Volume ≥ 1.2× the 20-bar average
6. Price is above VWAP
7. Bar time is before 2:30 PM ET

SHORT ENTRY — mirror image:
1. Active squeeze
2. Close BELOW the lower Bollinger Band
3. EMA(9) falling
4. RSI(14) between 30 and 50
5. Volume ≥ 1.2× avg
6. Price below VWAP
7. Before 2:30 PM ET

STOP PLACEMENT (Long):
  stop = min(bar_low − 0.25×ATR, entry − 1.0×ATR)

TARGET:
  risk   = entry − stop
  target = entry + risk × 2.0   (2R target)
  Minimum acceptable R:R = 1.5 (skip trade if below)

MAX HOLD: 60 five-minute bars (5 hours); exit EOD if not closed

REGIME FILTER:
- Best in BULL_OPEN (stock opened above prior close and is above VWAP)
- Allowed but penalised in CHOPPY
- Shorts preferred in BEAR_OPEN; longs heavily penalised

AVOID:
- When there is no squeeze (bands are wide / expanding)
- Earnings days, Fed days, or macro catalyst events
- After 2:30 PM ET
- Thin-volume pre/post market conditions

[Insert your analysis question here, e.g.: "Find recent NVDA setups matching
these rules", or "What are the key failure modes of this strategy?"]
```

---

### Prompt — Analyse SupertrendTrend

```
I'm using the following intraday trend-following strategy called SupertrendTrend.
Please analyse, critique, or scan based on these exact rules.

STRATEGY: SupertrendTrend (5m execution, 15m context, US equities, intraday only)

INDICATORS:
- Supertrend: ATR(10) × multiplier 3.0, using Wilder's ATR smoothing
  Formula: HL2 ± 3.0 × ATR(10); bands adjusted to never widen against trend
  Direction: +1 (bullish) when price holds above lower band, −1 (bearish) when below upper band
- EMA(20) — secondary pullback level
- RSI(14)
- ATR(14) — for stop and pullback distance measurement
- Relative volume = bar volume / 20-bar rolling avg

MACRO DIRECTION (15m chart):
- Compute the same Supertrend(10, 3.0) on today's 15m bars
- If 15m ST direction = +1 → macro is BULLISH → only consider long entries
- If 15m ST direction = −1 → macro is BEARISH → only consider short entries
- Fallback if <12 bars of 15m data: use session regime (BULL_OPEN = bullish)

LONG ENTRY — all conditions on a single 5m bar, before 3:00 PM ET:
1. 15m Supertrend direction = +1 (macro bullish)
2. 5m Supertrend direction = +1 (local trend aligned)
3. RSI(14) between 45 and 70
4. Volume ≥ 1.1× 20-bar avg
5. Bar is bullish reclaim candle: close > open
6. Pullback condition (either):
   a. Bar low is within 0.5×ATR of the 5m Supertrend line, OR
   b. Bar low is within 0.5×ATR of EMA(20)

SHORT ENTRY — mirror:
1. 15m ST = −1 (macro bearish)
2. 5m ST = −1
3. RSI between 30 and 55
4. Volume ≥ 1.1× avg
5. Bar closes below open (bearish candle)
6. Bar high within 0.5×ATR of 5m ST line or EMA(20)

STOP PLACEMENT (Long):
  primary: stop = 5m_ST_line − 0.1×ATR   (just below the Supertrend line)
  floor:   stop = min(stop, entry − 1.5×ATR)

TARGET:
  risk   = entry − stop
  target = entry + risk × 2.0   (2R)
  Minimum R:R = 1.5

TRAILING EXIT:
  Follow the 5m Supertrend line — exit when it flips to bearish (−1) against position.
  In practice: trail the stop up to the current 5m ST line as it rises.

MAX HOLD: 80 five-minute bars (~6.7 hours); EOD flatten if not closed

REGIME PREFERENCES:
- Ideal in BULL_OPEN (longs) or BEAR_OPEN (shorts) — confirmed directional day
- Avoid in CHOPPY — 15m and 5m Supertrend both flip frequently, signals are unreliable
- Entry confidence penalised −0.10 in CHOPPY regime

AVOID:
- CHOPPY days where the 15m Supertrend has already flipped 2+ times intraday
- When 5m and 15m ST contradict each other
- After 3:00 PM ET
- Less than 35 bars of today's 5m data (Supertrend + EMA20 not yet valid)

[Insert your analysis question here, e.g.: "Identify TSLA setups matching these
rules over the past 2 weeks", or "Compare this to a simple 200-EMA trend filter"]
```

---

### Prompt — Regime Context for All Strategies

```
I use the following market regime system for all intraday strategies.
When analysing my signals or suggesting trades, always apply this filter first.

REGIME DETECTION (computed once per day from 5m data):

BULL_OPEN:
  - Stock opened above prior day's close, AND
  - Current price is above today's VWAP
  → Favour long-only strategies (ORB, VWAP reversion, Bollinger longs, Supertrend longs)

BEAR_OPEN:
  - Stock opened below prior day's close, AND
  - Current price is below today's VWAP
  → Only defensive strategies allowed:
    EMAMomentum (shorts), OpeningGapFade, VolumeSpikeReversal (shorts)
    ORBBreakout and VWAPMeanReversion are BLOCKED
    BollingerMomentum longs are penalised −0.15 confidence
    SupertrendTrend longs are penalised −0.20 confidence

CHOPPY:
  - Neither BULL nor BEAR condition clearly met
  → All strategies allowed but every confidence score is reduced by −0.20
  → SupertrendTrend should be avoided (ST flips unreliable in ranging markets)

GLOBAL HARD STOP: No new entries after 3:15 PM ET regardless of regime.

Use this regime context when evaluating whether a given candle pattern or
indicator setup is tradeable.
```

---

*Last updated: 2026-05-15. Maintained in sync with `app/services/strategy/daytrading/strategies/`.*
