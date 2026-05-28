#!/usr/bin/env python
"""
A/B test the Layer-2 protective hard-stop against the REAL backtest engine.

The protective stop is a broker-resting SELL STOP placed `pct` below the BUY
fill. It triggers intrabar when price touches the stop — something the daily
engine (signals at close, fills at next open) does NOT model. So this harness:

  1. Runs run_backtest with the symbol's LIVE params (trail + calibration) to
     get the canonical round-trips — exactly today's behaviour (OFF).
  2. Replays each round-trip and asks: did any bar between entry and the actual
     exit trade with Low <= entry*(1-pct)?  If yes, the protective stop would
     have closed the trade THERE at the stop price (minus slippage), overriding
     the strategy's later exit. That's the ON leg.

This is the honest "if the hard stop were the binding exit" measurement. It can
only ever cut a trade short, never extend it — so it shows whether an 8% floor
saves you from drawdowns or just clips winners that later recovered.

OFF = strategy's real exits.  ON = same entries, 8% hard stop takes precedence.
Output -> data/protective_stop_ab_report.txt (UTF-8; console is cp1252).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.backtest.engine import run_backtest
from app.services.scanner.scanner_service import _make_generic_configs_full

PERIOD = "5y"
STOP_PCT = 8.0          # matches config.protective_stop_pct default
SLIPPAGE_PCT = 0.10     # stop fills are market orders; assume 0.10% worse than stop
OUT = "data/protective_stop_ab_report.txt"

# The 5 live trail-enabled symbols + their live strategy types.
TARGETS = [
    ("NVDA", "rsi2_mean_reversion"),
    ("TOST", "rsi2_mean_reversion"),
    ("AVPT", "rsi2_mean_reversion"),
    ("ARLO", "vix_spike_reversal"),
    ("COP",  "pullback_ema50"),
]


def _round_trips(result):
    """Pair BUY->SELL fills from a BacktestResult into (entry_date, entry_px,
    exit_date, exit_px) tuples, oldest first."""
    buys = [t for t in result.trades if t.side == "BUY"]
    sells = [t for t in result.trades if "SELL" in t.side]
    n = min(len(buys), len(sells))
    return [(buys[i].date, buys[i].price, sells[i].date, sells[i].price)
            for i in range(n)]


def _stop_adjusted_return(rt, df):
    """Given one round-trip and the OHLCV frame, return (off_ret_frac,
    on_ret_frac, stopped: bool). on leg applies the hard stop if any bar's Low
    between entry (exclusive) and actual exit (inclusive) pierced the stop."""
    entry_date, entry_px, exit_date, exit_px = rt
    off_ret = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0

    stop_level = entry_px * (1 - STOP_PCT / 100.0)
    # Bars strictly after entry through the actual exit date.
    dstr = [str(d)[:10] for d in df.index]
    try:
        i0 = dstr.index(entry_date)
        i1 = dstr.index(exit_date)
    except ValueError:
        return off_ret, off_ret, False
    lows = df["Low"].values
    for i in range(i0 + 1, i1 + 1):
        if float(lows[i]) <= stop_level:
            # Stop fills at stop_level, minus slippage (gap-downs ignored for
            # simplicity — this is conservative-optimistic for the stop).
            fill = stop_level * (1 - SLIPPAGE_PCT / 100.0)
            on_ret = (fill - entry_px) / entry_px
            return off_ret, on_ret, True
    return off_ret, off_ret, False


def _compound(returns):
    eq = 1.0
    for r in returns:
        eq *= (1 + r)
    return (eq - 1) * 100.0


def main():
    lines = [
        "PROTECTIVE HARD-STOP OVERLAY — A/B (real no-lookahead engine entries)",
        f"Period {PERIOD} | stop {STOP_PCT:.0f}% below entry | slippage {SLIPPAGE_PCT:.2f}%",
        "OFF = strategy's real exits.  ON = 8% hard stop takes precedence intrabar.",
        "=" * 100,
        f"{'symbol':<7}{'type':<22}{'OFF ret%':>10}{'ON ret%':>10}"
        f"{'delta pp':>10}{'stopped':>10}{'trades':>9}",
        "-" * 100,
    ]
    agg_off, agg_on, total_stopped, total_trades = [], [], 0, 0
    for sym, stype in TARGETS:
        try:
            cfg = next((c for c in _make_generic_configs_full(sym) if c.type == stype), None)
            if cfg is None:
                lines.append(f"{sym:<7} no config for {stype}"); continue
            df = get_ohlcv(sym, period=PERIOD)
            r = run_backtest(strategy_name=f"{sym}:{stype}", symbol=sym,
                             strategy_type=stype, params=cfg.params,
                             period=PERIOD, quantity=0, df=df)
            rts = _round_trips(r)
            if not rts:
                lines.append(f"{sym:<7}{stype:<22}{'no round-trips':>40}"); continue
            offs, ons, stopped = [], [], 0
            for rt in rts:
                o, n, hit = _stop_adjusted_return(rt, df)
                offs.append(o); ons.append(n)
                if hit:
                    stopped += 1
            off_total = _compound(offs)
            on_total = _compound(ons)
            agg_off.extend(offs); agg_on.extend(ons)
            total_stopped += stopped; total_trades += len(rts)
            lines.append(
                f"{sym:<7}{stype:<22}{off_total:>10.1f}{on_total:>10.1f}"
                f"{on_total - off_total:>+10.1f}{f'{stopped}/{len(rts)}':>10}{len(rts):>9}")
        except Exception as e:
            lines.append(f"{sym:<7} ERROR {type(e).__name__}: {e}")
    if agg_off:
        lines.append("-" * 100)
        lines.append(
            f"  Aggregate (compounded per-symbol-blind): OFF {_compound(agg_off):.1f}% "
            f"-> ON {_compound(agg_on):.1f}%   |  stops hit {total_stopped}/{total_trades} trades")
        lines.append(
            "  Read: a NEGATIVE delta means the 8% stop CLIPPED winners that later "
            "recovered (drag). A POSITIVE delta means it cut losers short (saved).")
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
