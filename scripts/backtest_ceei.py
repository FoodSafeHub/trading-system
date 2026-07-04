"""CEEI validation backtest + AMAT vs CEEI comparison.

Runs both custom indicators through the standard backtest engine on the same
symbols/period, prints trade count, win rate, max drawdown, Sharpe, and average
gain/loss per round-trip trade, and dumps per-bar CEEI debug CSVs
(component scores, states, breakout levels, signals) for plotting.

Usage:
    python scripts/backtest_ceei.py                          # defaults, 2y
    python scripts/backtest_ceei.py AAPL MSFT --period 5y
    python scripts/backtest_ceei.py NVDA --setup 60 --expansion 55 --buy 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import pandas as pd  # noqa: E402

from app.services.backtest.engine import BacktestResult, run_backtest  # noqa: E402
from app.services.indicators.ceei import CEEIParams, compute_ceei  # noqa: E402
from app.services.market_data.provider import get_ohlcv  # noqa: E402

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def round_trip_stats(result: BacktestResult) -> tuple[float | None, float | None]:
    """Average gain% of winning round trips and average loss% of losing ones,
    reconstructed by pairing sequential BUY -> SELL fills."""
    gains: list[float] = []
    losses: list[float] = []
    entry: float | None = None
    for t in result.trades:
        if t.side == "BUY" and entry is None:
            entry = t.price
        elif t.side == "SELL" and entry is not None:
            pct = (t.price - entry) / entry * 100
            (gains if pct >= 0 else losses).append(pct)
            entry = None
    avg_gain = sum(gains) / len(gains) if gains else None
    avg_loss = sum(losses) / len(losses) if losses else None
    return avg_gain, avg_loss


def fmt(v: float | None, spec: str = ".2f") -> str:
    return format(v, spec) if v is not None else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="CEEI backtest + AMAT comparison")
    ap.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--period", default="2y")
    ap.add_argument("--setup", type=float, default=70.0, help="compression setup threshold")
    ap.add_argument("--setup-window", type=int, default=10)
    ap.add_argument("--expansion", type=float, default=45.0, help="expansion trigger threshold")
    ap.add_argument("--efficiency", type=float, default=55.0, help="efficiency confirmation floor")
    ap.add_argument("--buy", type=float, default=48.0, help="CEEI buy threshold")
    ap.add_argument("--sell", type=float, default=48.0, help="CEEI sell threshold")
    ap.add_argument("--no-amat", action="store_true", help="skip the AMAT comparison run")
    ap.add_argument("--csv-dir", default="output")
    args = ap.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS

    ceei_params = {
        "setup_threshold": args.setup,
        "setup_window": args.setup_window,
        "expansion_threshold": args.expansion,
        "efficiency_min": args.efficiency,
        "buy_threshold": args.buy,
        "sell_threshold": args.sell,
    }

    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    runs = [("ceei", ceei_params)]
    if not args.no_amat:
        runs.append(("amat", {}))

    header = (f"{'Strategy':<7}{'Symbol':<8}{'Return%':>9}{'Trades':>8}{'Win%':>7}"
              f"{'MaxDD%':>8}{'Sharpe':>8}{'AvgWin%':>9}{'AvgLoss%':>10}")
    print("\n" + header)
    print("-" * len(header))

    for symbol in symbols:
        for strat, params in runs:
            try:
                result = run_backtest(
                    strategy_name=f"{strat}_{symbol}",
                    symbol=symbol,
                    strategy_type=strat,
                    params=params,
                    period=args.period,
                )
            except Exception as e:
                print(f"{strat:<7}{symbol:<8} FAILED: {e}")
                continue
            avg_gain, avg_loss = round_trip_stats(result)
            sharpe = fmt(result.sharpe_ratio)
            print(f"{strat:<7}{symbol:<8}{result.total_return_pct:>9.2f}{result.total_trades:>8}"
                  f"{result.win_rate_pct:>7.1f}{result.max_drawdown_pct:>8.2f}{sharpe:>8}"
                  f"{fmt(avg_gain):>9}{fmt(avg_loss):>10}")

        # Per-bar CEEI debug CSV + score distribution for threshold calibration
        try:
            df = get_ohlcv(symbol, period=args.period)
            r = compute_ceei(
                df["High"], df["Low"], df["Close"], df.get("Volume"),
                params=CEEIParams(
                    setup_threshold=args.setup, setup_window=args.setup_window,
                    expansion_threshold=args.expansion, efficiency_min=args.efficiency,
                    buy_threshold=args.buy, sell_threshold=args.sell,
                ),
                log_context=symbol,
            )
            debug = pd.DataFrame({
                "close": df["Close"],
                "compression_score": r.compression_score,
                "expansion_score": r.expansion_score,
                "expansion_direction": r.expansion_direction,
                "efficiency_score": r.efficiency_score,
                "ceei_score": r.ceei_score,
                "setup_state": r.setup_state,
                "trigger_state": r.trigger_state,
                "breakout_level": r.breakout_level,
                "breakdown_level": r.breakdown_level,
                "signal": r.signal,
            })
            out = csv_dir / f"ceei_debug_{symbol}.csv"
            debug.to_csv(out)
            q = lambda s, p: float(s.dropna().quantile(p))  # noqa: E731
            print(f"        {symbol} score p50/p80/p95 — compression: "
                  f"{q(r.compression_score, .5):.0f}/{q(r.compression_score, .8):.0f}/{q(r.compression_score, .95):.0f}"
                  f"  expansion: {q(r.expansion_score, .5):.0f}/{q(r.expansion_score, .8):.0f}/{q(r.expansion_score, .95):.0f}"
                  f"  efficiency: {q(r.efficiency_score, .5):.0f}/{q(r.efficiency_score, .8):.0f}/{q(r.efficiency_score, .95):.0f}"
                  f"  ceei: {q(r.ceei_score, .5):.0f}/{q(r.ceei_score, .8):.0f}/{q(r.ceei_score, .95):.0f}"
                  f"  setups={int(r.setup_state.sum())} triggers={int(r.trigger_state.sum())}"
                  f"  (csv: {out})")
        except Exception as e:
            print(f"        {symbol} debug csv failed: {e}")


if __name__ == "__main__":
    main()
