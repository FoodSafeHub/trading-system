"""Run all 5 strategies x 6 symbols at 20% position sizing over 5y."""
from app.services.strategy.perplexity.strategies import (
    EmaMeanReversionUptrend, MaCrossoverRsi, BreakoutConsolidation,
    BollingerMeanReversionUptrend, FibPullbackSupport,
)
from app.services.backtest.perplexity_engine import run_perplexity_backtest

strategies = [
    EmaMeanReversionUptrend(),
    MaCrossoverRsi(),
    BreakoutConsolidation(),
    BollingerMeanReversionUptrend(),
    FibPullbackSupport(),
]
symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL"]
PERIOD   = "5y"
CAPITAL  = 10_000
POS_PCT  = 0.20

SEP = "-" * 92
print(f"\n5-Year Backtest — 20% position size per trade — $10k starting capital")
print(SEP)
print(f"{'Strategy':<28} {'Symbol':<6} {'Trades':>6} {'Win%':>6} {'CAGR%':>7} {'Return%':>8} {'MaxDD%':>7} {'PF':>5}")
print(SEP)

totals = []
for strat in strategies:
    for sym in symbols:
        try:
            r = run_perplexity_backtest(
                strat, sym, period=PERIOD,
                initial_capital=CAPITAL, position_pct=POS_PCT,
            )
            pf = round(r.profit_factor, 2) if r.profit_factor else 0.0
            print(
                f"{strat.name:<28} {sym:<6} {r.total_trades:>6} "
                f"{r.win_rate_pct:>5.0f}% {r.cagr:>6.1f}% "
                f"{r.total_return_pct:>7.1f}% {r.max_drawdown_pct:>6.1f}% {pf:>5.2f}"
            )
            totals.append({
                "strategy": strat.name, "symbol": sym,
                "trades": r.total_trades, "cagr": r.cagr,
                "wr": r.win_rate_pct, "return": r.total_return_pct,
                "dd": r.max_drawdown_pct, "pf": pf,
            })
        except Exception as e:
            print(f"{strat.name:<28} {sym:<6} ERROR: {e}")

print(SEP)

qualified = [t for t in totals if t["trades"] >= 5]
if qualified:
    best = sorted(qualified, key=lambda x: x["cagr"], reverse=True)[:8]
    print("\nTop 8 by CAGR (min 5 trades):")
    for b in best:
        print(
            f"  {b['strategy']}/{b['symbol']}: "
            f"CAGR={b['cagr']:.1f}%  Return={b['return']:.1f}%  "
            f"WR={b['wr']:.0f}%  DD={b['dd']:.1f}%  Trades={b['trades']}"
        )

    avg_cagr = sum(t["cagr"] for t in qualified) / len(qualified)
    avg_wr   = sum(t["wr"]   for t in qualified) / len(qualified)
    print(f"\nAverage across all qualified combos: CAGR={avg_cagr:.1f}%  WR={avg_wr:.0f}%")
