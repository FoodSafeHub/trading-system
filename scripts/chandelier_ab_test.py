#!/usr/bin/env python
"""
A/B test the ATR-Chandelier trailing-stop overlay through the REAL backtest
engine (app.services.backtest.engine.run_backtest), which is no-lookahead and
shares the exact exit code the scheduler uses.

OFF = strategy's current live params (fixed-band exits).
ON  = same params + trail_enabled (chandelier ride once a position is up >= trigger).

The overlay only affects exits and only fires while a position is open, so OFF
reproduces today's behaviour exactly. Output -> data/chandelier_ab_report.txt
(UTF-8; this console is cp1252 and chokes on arrows).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.market_data.provider import get_ohlcv
from app.services.backtest.engine import run_backtest
from app.services.scanner.scanner_service import _make_generic_configs_full

PERIOD = "5y"
OUT = "data/chandelier_ab_report.txt"
TRAIL = {"trail_enabled": True, "trail_trigger_pct": 3.0,
         "atr_trail_mult": 3.0, "atr_trail_period": 22}

# (symbol, live strategy_name). NVDA/TOST/AVPT/ARLO/COP are the ones we enabled;
# the rest are kept here so a re-run shows why they stayed on fixed-band.
TARGETS = [
    ("NVDA", "NVDA_RSI2_Mean_Reversion"),
    ("TSLA", "Legacy_TSLA_Fib_Pullback"),
    ("TOST", "TOST_RSI2_Mean_Reversion"),
    ("AVPT", "AVPT_RSI2_Mean_Reversion"),
    ("ARLO", "ARLO_VIX_Spike_Reversal"),
    ("COP",  "COP_Pullback_EMA50"),
    ("SO",   "SO_Pullback_EMA50"),
    ("FANG", "FANG_Pullback_EMA50"),
    ("XOM",  "XOM_Pullback_EMA50"),
    ("ARKG", "ARKG_EMA_MACD_Crossover"),
    ("COLL", "COLL_VIX_Spike_Reversal"),
]

# get_param_overrides already merges the trail params we saved into the live
# configs, so to measure a clean OFF baseline we strip trail_* before running OFF.
TRAIL_KEYS = set(TRAIL.keys())


def _run(sym, sname, df, cfg, trail: bool):
    params = {k: v for k, v in cfg.params.items() if k not in TRAIL_KEYS}
    if trail:
        params.update(TRAIL)
    r = run_backtest(strategy_name=sname, symbol=sym, strategy_type=cfg.type,
                     params=params, period=PERIOD, quantity=0, df=df)
    return r.total_return_pct, r.total_trades, r.win_rate_pct


def main():
    lines = ["CHANDELIER TRAILING-STOP OVERLAY — A/B (real no-lookahead engine)",
             f"Period {PERIOD} | trigger {TRAIL['trail_trigger_pct']}% | "
             f"ATR {TRAIL['atr_trail_period']} x {TRAIL['atr_trail_mult']}",
             "=" * 92,
             f"{'symbol':<7}{'type':<20}{'OFF ret%':>10}{'ON ret%':>10}{'delta pp':>10}"
             f"{'OFF WR':>8}{'ON WR':>8}{'trades':>16}"]
    lines.append("-" * 92)
    rows = []
    for sym, sname in TARGETS:
        try:
            cfg = next((c for c in _make_generic_configs_full(sym) if c.name == sname), None)
            if cfg is None:
                lines.append(f"{sym:<7} no config"); continue
            df = get_ohlcv(sym, period=PERIOD)
            o_ret, o_n, o_wr = _run(sym, sname, df, cfg, trail=False)
            n_ret, n_n, n_wr = _run(sym, sname, df, cfg, trail=True)
            d = n_ret - o_ret
            rows.append((sym, cfg.type, o_ret, n_ret, d))
            lines.append(
                f"{sym:<7}{cfg.type:<20}{o_ret:>10.1f}{n_ret:>10.1f}{d:>+10.1f}"
                f"{o_wr:>8.0f}{n_wr:>8.0f}{f'{o_n}->{n_n}':>16}")
        except Exception as e:
            lines.append(f"{sym:<7} ERROR {type(e).__name__}: {e}")
    if rows:
        improved = sum(1 for r in rows if r[4] > 0)
        lines.append("-" * 92)
        lines.append(f"  Improved: {improved}/{len(rows)}   "
                     f"Aggregate OFF {sum(r[2] for r in rows):.1f}% -> "
                     f"ON {sum(r[3] for r in rows):.1f}%")
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"WROTE {OUT} ({len(rows)} compared)")


if __name__ == "__main__":
    main()
