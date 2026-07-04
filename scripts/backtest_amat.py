"""Quick AMAT validation backtest.

Runs the 'amat' strategy through the standard backtest engine over historical
OHLCV, prints per-symbol metrics, and dumps a per-bar debug CSV
(close / trend_spine / amat_score / divergence / signal) for plotting.

Usage:
    python scripts/backtest_amat.py                        # default symbols, 2y
    python scripts/backtest_amat.py AAPL MSFT --period 5y
    python scripts/backtest_amat.py NVDA --buy 70 --sell 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import pandas as pd  # noqa: E402

from app.services.backtest.engine import run_backtest  # noqa: E402
from app.services.indicators.amat import AMATParams, compute_amat  # noqa: E402
from app.services.market_data.provider import get_ohlcv  # noqa: E402

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY"]


def main() -> None:
    ap = argparse.ArgumentParser(description="AMAT indicator backtest")
    ap.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--period", default="2y")
    ap.add_argument("--buy", type=float, default=65.0, help="BUY score threshold")
    ap.add_argument("--sell", type=float, default=35.0, help="SELL score threshold")
    ap.add_argument("--multiplier", type=float, default=1.0, help="Trend Spine ATR multiplier")
    ap.add_argument("--accel-step", type=int, default=3)
    ap.add_argument("--div-lookback", type=int, default=10)
    ap.add_argument("--csv-dir", default="output", help="where to write per-bar debug CSVs")
    args = ap.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS

    params = {
        "buy_threshold": args.buy,
        "sell_threshold": args.sell,
        "multiplier": args.multiplier,
        "accel_step": args.accel_step,
        "divergence_lookback": args.div_lookback,
    }

    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Symbol':<8}{'Return%':>9}{'Trades':>8}{'Win%':>7}{'MaxDD%':>8}{'Sharpe':>8}  Divergent bars")
    print("-" * 70)

    for symbol in symbols:
        try:
            result = run_backtest(
                strategy_name=f"amat_{symbol}",
                symbol=symbol,
                strategy_type="amat",
                params=params,
                period=args.period,
            )
        except Exception as e:  # data fetch or engine failure — keep going
            print(f"{symbol:<8} FAILED: {e}")
            continue

        # Per-bar debug series for plotting/audit
        div_bars = "n/a"
        try:
            df = get_ohlcv(symbol, period=args.period)
            amat = compute_amat(
                df["High"], df["Low"], df["Close"],
                df.get("Volume"),
                params=AMATParams(
                    buy_threshold=args.buy, sell_threshold=args.sell,
                    multiplier=args.multiplier, accel_step=args.accel_step,
                    divergence_lookback=args.div_lookback,
                ),
                log_context=symbol,
            )
            debug = pd.DataFrame({
                "close": df["Close"],
                "trend_spine": amat.trend_spine,
                "amat_score": amat.score,
                "weighted_accel": amat.weighted_acceleration,
                "divergence_penalty": amat.divergence,
                "signal": amat.signal,
            })
            out = csv_dir / f"amat_debug_{symbol}.csv"
            debug.to_csv(out)
            div_bars = f"{int(amat.divergence.sum())} (csv: {out})"
        except Exception as e:
            div_bars = f"debug csv failed: {e}"

        sharpe = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "n/a"
        print(f"{symbol:<8}{result.total_return_pct:>9.2f}{result.total_trades:>8}"
              f"{result.win_rate_pct:>7.1f}{result.max_drawdown_pct:>8.2f}{sharpe:>8}  {div_bars}")

    print("\nCompare against a baseline, e.g.:")
    print('  run_backtest(..., strategy_type="supertrend", ...) on the same symbols/period.')


if __name__ == "__main__":
    main()
