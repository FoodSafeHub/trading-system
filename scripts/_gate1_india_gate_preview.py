#!/usr/bin/env python
"""GATE 1 INDIA-GATE PREVIEW — read-only, in-memory monkeypatch only.

NOT applied as a code change:
  * No edits to rules.py, no new modules under app/.
  * attach_india_benchmark, _india_bull, and the wrapper are SCRIPT-LOCAL.
  * rules._spy_is_bull is rebound in-process for the duration of main() and
    restored in a finally block before exit.
  * No files, assignments, profiles, strategies.json, scheduler.py, or
    live-trading flags are modified.

Production semantics are matched via column presence (the only routing signal
the wrapper uses):
  * UNIFIED India runs WITH attach_india_benchmark  -> gated (new behaviour)
  * UNIFIED India runs WITHOUT attach                -> own-SMA200 fallback
                                                       (== today's behaviour)
  * LEGACY India runs (never attached)               -> own-SMA200 (unchanged)
  * US runs (attach is a no-op for non-India)        -> own-SMA200 (unchanged)

Only BHARTIARTL is exercised under the gated logic. NVDA/AAPL/FANG remain on
legacy and are only used here to spot-check US invariance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.api.routes.backtest import _trade_metrics
from app.services.backtest.costs import INDIA_DEFAULT
from app.services.backtest.engine import run_backtest
from app.services.market_data import provider as _mdp
from app.services.markets import is_india_symbol
from app.services.scanner.scanner_service import _make_generic_configs, _make_unified_configs
from app.services.strategy import rules

PERIOD = "5y"
INIT = 10_000.0
VIX_HOT = 22.0
NSEI = "^NSEI"
INDIA_VIX = "^INDIAVIX"

# ── Script-local: proposed attach_india_benchmark (NOT added to app/) ──────────

def attach_india_benchmark(df: pd.DataFrame, symbol: str, period: str = "10y") -> pd.DataFrame:
    """In-memory preview of the proposed helper. Fetches each benchmark ONCE
    (not per bar), reindexes to df.index with forward-fill. No-op for non-India."""
    if not is_india_symbol(symbol) or df is None or df.empty:
        return df
    out = df.copy()
    try:
        n = _mdp.get_ohlcv(NSEI, period=period)
        if n is not None and not n.empty and "Close" in n.columns:
            out["nsei_close"] = n["Close"].reindex(out.index, method="ffill")
    except Exception as e:
        print(f"  [attach] ^NSEI fetch failed: {e}")
    try:
        v = _mdp.get_ohlcv(INDIA_VIX, period=period)
        if v is not None and not v.empty and "Close" in v.columns:
            out["india_vix"] = v["Close"].reindex(out.index, method="ffill")
    except Exception as e:
        print(f"  [attach] ^INDIAVIX fetch failed: {e}")
    return out


# ── Script-local: _india_bull + wrapper installed via monkeypatch ──────────────

_orig_spy_is_bull = rules._spy_is_bull


def _india_bull(ohlcv, prices, vix_hot: float = VIX_HOT) -> bool:
    cols = getattr(ohlcv, "columns", []) if ohlcv is not None else []
    if "nsei_close" not in cols:
        return rules._above_sma200(prices)
    nsei = ohlcv["nsei_close"].dropna()
    if len(nsei) < 200:
        return rules._above_sma200(prices)
    sma200 = float(nsei.rolling(200).mean().iloc[-1])
    nsei_bull = float(nsei.iloc[-1]) > sma200
    if "india_vix" in cols:
        v = ohlcv["india_vix"].dropna()
        if len(v) > 0:
            return nsei_bull and float(v.iloc[-1]) < float(vix_hot)
    return nsei_bull  # NSEI-only gate when VIX column is absent


def _wrapped_spy_is_bull(ohlcv, prices):
    """Column-presence routes India vs US:
       * has nsei_close -> India bull logic (NSEI [+ VIX])
       * else            -> original _spy_is_bull (byte-identical US path).
    """
    cols = getattr(ohlcv, "columns", []) if ohlcv is not None else []
    if "nsei_close" in cols:
        return _india_bull(ohlcv, prices)
    return _orig_spy_is_bull(ohlcv, prices)


def install_patch():
    rules._spy_is_bull = _wrapped_spy_is_bull


def restore_patch():
    rules._spy_is_bull = _orig_spy_is_bull


# ── Backtest helpers ───────────────────────────────────────────────────────────

def _bt(stype, sym, params, df, cm):
    r = run_backtest(strategy_name=f"{stype}:{sym}", symbol=sym, strategy_type=stype,
                     params=params, period=PERIOD, initial_capital=INIT, quantity=0,
                     df=df.copy(), cost_model=cm)
    m = _trade_metrics(r)
    m["ret"] = r.total_return_pct
    m["maxdd"] = r.max_drawdown_pct
    m["buys"] = sum(1 for t in r.trades if t.side == "BUY")
    return m, r


def _fmt(m):
    return (f"ret%={m['ret']:8.1f}  exp%={m['expectancy_pct']:6.2f}  wr%={m['win_rate_pct']:5.1f}  "
            f"PF={str(m['profit_factor']):>5}  maxDD%={m['maxdd']:5.1f}  rt={m['round_trips']:>3}  buys={m['buys']}")


def _wins(uni, leg):
    return uni["ret"] > leg["ret"] and uni["expectancy_pct"] >= leg["expectancy_pct"]


def main():
    print("=" * 100)
    print("GATE 1 INDIA-GATE PREVIEW (READ-ONLY; in-memory monkeypatch only)")
    print("BHARTIARTL only | NVDA/AAPL/FANG remain on legacy | nothing in app/ or on disk is modified")
    print("=" * 100)

    sym = "BHARTIARTL"
    leg_cfg = next(c for c in _make_generic_configs(sym) if c.type == "rsi2_mean_reversion")
    uni_cfg = next(c for c in _make_unified_configs(sym) if c.type == "rsi2_reversion")

    df_bh = _mdp.get_ohlcv(sym, period=PERIOD)
    if df_bh is None or df_bh.empty:
        print("FATAL: BHARTIARTL data fetch failed."); return

    # Baseline (PATCH NOT INSTALLED): unified own-SMA200 fallback as it runs today.
    print("\n--- Baseline (patch NOT installed): BHARTIARTL unified, own-SMA200 fallback ---")
    pre_uni_gross, _ = _bt(uni_cfg.type, sym, uni_cfg.params, df_bh, None)
    pre_uni_net,   _ = _bt(uni_cfg.type, sym, uni_cfg.params, df_bh, INDIA_DEFAULT)
    print("  UNIFIED gross   ", _fmt(pre_uni_gross))
    print("  UNIFIED net IN  ", _fmt(pre_uni_net))

    install_patch()
    try:
        # 1. US INVARIANCE — NVDA unified before/after the wrapper, no attach.
        print("\n--- (1) US invariance check (NVDA, no attach) ---")
        df_nv = _mdp.get_ohlcv("NVDA", period=PERIOD)
        uni_nv_cfg = next(c for c in _make_unified_configs("NVDA") if c.type == "rsi2_reversion")
        restore_patch()
        nv_before, _ = _bt("rsi2_reversion", "NVDA", uni_nv_cfg.params, df_nv, None)
        install_patch()
        nv_after,  _ = _bt("rsi2_reversion", "NVDA", uni_nv_cfg.params, df_nv, None)
        nv_ok = (nv_before == nv_after)
        print(f"  NVDA pre-install : ret%={nv_before['ret']:.2f}  rt={nv_before['round_trips']}  buys={nv_before['buys']}")
        print(f"  NVDA wrapper on  : ret%={nv_after['ret']:.2f}   rt={nv_after['round_trips']}   buys={nv_after['buys']}")
        print(f"  US invariance: {'PASS (byte-identical)' if nv_ok else 'FAIL'}")

        # 2. FALLBACK — BHARTIARTL unified with wrapper installed but WITHOUT attach
        #    must match the pre-install baseline byte-identically.
        print("\n--- (2) Fallback (BHARTIARTL unified, wrapper installed, NO attach) ---")
        ungated, _ = _bt(uni_cfg.type, sym, uni_cfg.params, df_bh, None)
        fb_ok = (ungated == pre_uni_gross)
        print("  unified gross   ", _fmt(ungated))
        print(f"  Matches baseline (own-SMA200 fallback): {'PASS (byte-identical)' if fb_ok else 'FAIL'}")

        # 3. ATTACH + NO-LOOKAHEAD + GATED HEAD-TO-HEAD
        print("\n--- (3) Attach ^NSEI / ^INDIAVIX, then gated head-to-head ---")
        df_bh_attached = attach_india_benchmark(df_bh, sym, period="10y")
        nsei_present = "nsei_close" in df_bh_attached.columns
        vix_present  = "india_vix"  in df_bh_attached.columns
        print(f"  Columns attached: nsei_close={nsei_present}, india_vix={vix_present}")

        # No-lookahead causality check (NSEI)
        if nsei_present:
            src = _mdp.get_ohlcv(NSEI, period="10y")["Close"]
            src = src[~src.index.duplicated(keep="last")].sort_index()
            aligned = src.reindex(df_bh_attached.index, method="ffill")
            joined = pd.concat(
                [aligned.rename("src"), df_bh_attached["nsei_close"].rename("att")], axis=1
            ).dropna()
            causal_ok = bool((joined["src"].round(6) == joined["att"].round(6)).all())
            print(f"  No-lookahead causality (NSEI): {'PASS' if causal_ok else 'FAIL'} ({len(joined)} bars compared)")
        else:
            print("  No-lookahead check skipped (^NSEI unavailable; gate will degrade to own-SMA200)")

        leg_g, _ = _bt(leg_cfg.type, sym, leg_cfg.params, df_bh, None)             # legacy, no attach
        leg_n, _ = _bt(leg_cfg.type, sym, leg_cfg.params, df_bh, INDIA_DEFAULT)
        uni_g, _ = _bt(uni_cfg.type, sym, uni_cfg.params, df_bh_attached, None)    # unified, gated
        uni_n, _ = _bt(uni_cfg.type, sym, uni_cfg.params, df_bh_attached, INDIA_DEFAULT)

        print(f"  GROSS legacy    {_fmt(leg_g)}")
        print(f"  GROSS unified G {_fmt(uni_g)}")
        print(f"  NET   legacy    {_fmt(leg_n)}")
        print(f"  NET   unified G {_fmt(uni_n)}")

        # Gate effectiveness: ungated vs gated BUY count for the unified rule.
        delta_buys = pre_uni_gross["buys"] - uni_g["buys"]
        print(f"\n  Gate effectiveness: ungated buys={pre_uni_gross['buys']}  |  gated buys={uni_g['buys']}  |  delta={delta_buys}")
        if delta_buys > 0:
            print("    -> ACTIVE: gate suppressed some entries (good — proves the gate is real)")
        elif delta_buys == 0:
            print("    -> INERT vs ungated: no entries suppressed (gate may be redundant on this history)")
        else:
            print("    -> WARN: gate added BUYs (unexpected); investigate causality")

        # Verdict (same rule as GATE 1)
        gross_win = _wins(uni_g, leg_g)
        net_win   = _wins(uni_n, leg_n)
        sample_ok = uni_g["round_trips"] >= 10
        advance = gross_win and net_win and sample_ok and delta_buys >= 0
        verdict = "ADVANCE" if advance else "HOLD-ON-LEGACY"
        notes = [
            "gross_win" if gross_win else "gross_FAIL",
            "net_win" if net_win else "net_FAIL",
            f"sample({uni_g['round_trips']})",
            f"gate_delta_buys={delta_buys}",
        ]
        print(f"\n--- VERDICT (BHARTIARTL only): {verdict}  [{', '.join(notes)}]  PAPER-ONLY when eligible ---")

    finally:
        restore_patch()

    print("\nDONE (read-only; nothing in app/, profiles, assignments, strategies.json, or flags was modified)")


if __name__ == "__main__":
    main()
