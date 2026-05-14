"""
Walk-Forward Validator for day trading strategies.

Methodology:
  - Fetch daily OHLCV data (up to 10y) to define calendar windows.
  - Each window: 70% in-sample (IS) training, 30% out-of-sample (OOS) test.
  - Windows step forward by step_months (default 6m).
  - Within each window, run_backtest() on the intraday data (5m bars) for that date range.
  - Compute WFE (Walk-Forward Efficiency) = OOS_CAGR / IS_CAGR.
  - WFE > 0.6 → robust strategy, 0.3–0.6 → marginal, < 0.3 → overfit.

Usage:
    result = run_walkforward("AAPL", "ORBBreakout", years=5)
    print(result.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MIN_IS_TRADES = 1      # skip window if IS produced fewer trades
_MIN_OOS_TRADES = 1     # OOS must have at least this many trades to count WFE


@dataclass
class WindowResult:
    window_idx: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    # IS metrics
    is_trades: int
    is_total_pnl: float
    is_cagr: float
    is_win_rate: float
    is_profit_factor: float
    is_max_drawdown: float
    # OOS metrics
    oos_trades: int
    oos_total_pnl: float
    oos_cagr: float
    oos_win_rate: float
    oos_profit_factor: float
    oos_max_drawdown: float
    # Efficiency
    wfe: float                  # OOS_CAGR / IS_CAGR; None if IS_CAGR <= 0
    wfe_label: str              # "robust" / "marginal" / "overfit" / "unprofitable"
    # Equity curves (list of floats from initial_capital)
    is_equity_curve: list[float] = field(default_factory=list)
    oos_equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window_idx,
            "is_start": str(self.is_start),
            "is_end": str(self.is_end),
            "oos_start": str(self.oos_start),
            "oos_end": str(self.oos_end),
            "is": {
                "trades": self.is_trades,
                "total_pnl": round(self.is_total_pnl, 2),
                "cagr_pct": round(self.is_cagr * 100, 2),
                "win_rate": round(self.is_win_rate, 2),
                "profit_factor": round(self.is_profit_factor, 2),
                "max_drawdown_pct": round(self.is_max_drawdown, 2),
                "equity_curve": [round(v, 2) for v in self.is_equity_curve],
            },
            "oos": {
                "trades": self.oos_trades,
                "total_pnl": round(self.oos_total_pnl, 2),
                "cagr_pct": round(self.oos_cagr * 100, 2),
                "win_rate": round(self.oos_win_rate, 2),
                "profit_factor": round(self.oos_profit_factor, 2),
                "max_drawdown_pct": round(self.oos_max_drawdown, 2),
                "equity_curve": [round(v, 2) for v in self.oos_equity_curve],
            },
            "wfe": round(self.wfe, 3),
            "wfe_label": self.wfe_label,
        }


@dataclass
class WalkForwardResult:
    symbol: str
    strategy: str
    initial_capital: float
    windows: list[WindowResult] = field(default_factory=list)
    # Aggregate stats across all valid windows
    avg_wfe: float = 0.0
    median_wfe: float = 0.0
    pct_windows_robust: float = 0.0    # % windows with WFE > 0.6
    total_oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_cagr: float = 0.0
    robustness_label: str = "insufficient_data"
    error: str = ""

    def summary(self) -> str:
        lines = [
            f"Walk-Forward: {self.strategy} on {self.symbol}",
            f"  Windows run : {len(self.windows)}",
            f"  Avg WFE     : {self.avg_wfe:.3f}  (>0.6 = robust)",
            f"  Median WFE  : {self.median_wfe:.3f}",
            f"  % Robust    : {self.pct_windows_robust:.0%}",
            f"  OOS trades  : {self.total_oos_trades}",
            f"  OOS CAGR    : {self.oos_cagr*100:.1f}%",
            f"  Verdict     : {self.robustness_label.upper()}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "initial_capital": self.initial_capital,
            "windows": [w.to_dict() for w in self.windows],
            "summary": {
                "avg_wfe": round(self.avg_wfe, 3),
                "median_wfe": round(self.median_wfe, 3),
                "pct_windows_robust": round(self.pct_windows_robust, 3),
                "total_oos_trades": self.total_oos_trades,
                "oos_win_rate": round(self.oos_win_rate, 2),
                "oos_cagr_pct": round(self.oos_cagr * 100, 2),
                "robustness_label": self.robustness_label,
            },
            "error": self.error,
        }


class WalkForwardValidator:
    """
    Runs rolling walk-forward analysis on a day trading strategy.

    Parameters
    ----------
    symbol          : ticker, e.g. "AAPL"
    strategy_name   : e.g. "ORBBreakout" — must be in STRATEGY_MAP
    years           : how many years of history to use (max ~10 for 5m, realistically 2 for yfinance)
    is_pct          : fraction of each window used for in-sample (default 0.70)
    step_months     : how many months each window advances (default 6)
    initial_capital : backtest starting equity
    position_pct    : fraction of equity per trade
    """

    def __init__(
        self,
        symbol: str,
        strategy_name: str,
        years: int = 2,
        is_pct: float = 0.70,
        step_months: int = 3,
        initial_capital: float = 10_000.0,
        position_pct: float = 0.95,
    ):
        self.symbol = symbol.upper()
        self.strategy_name = strategy_name
        self.years = min(years, 2)      # yfinance 5m limit is 60d; we use periods
        self.is_pct = is_pct
        self.step_months = step_months
        self.initial_capital = initial_capital
        self.position_pct = position_pct

    def run(self) -> WalkForwardResult:
        result = WalkForwardResult(
            symbol=self.symbol,
            strategy=self.strategy_name,
            initial_capital=self.initial_capital,
        )

        try:
            windows = self._build_windows()
            if not windows:
                result.error = "No windows generated — insufficient date range"
                return result

            for idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
                win = self._run_window(idx, is_start, is_end, oos_start, oos_end)
                if win is not None:
                    result.windows.append(win)
                    logger.info(
                        "Window %d: IS %s→%s OOS %s→%s WFE=%.3f (%s)",
                        idx, is_start, is_end, oos_start, oos_end, win.wfe, win.wfe_label
                    )

            self._aggregate(result)

        except Exception as e:
            logger.exception("WalkForward error: %s", e)
            result.error = str(e)

        return result

    # ── Window building ───────────────────────────────────────────────────────

    def _build_windows(self) -> list[tuple[date, date, date, date]]:
        """
        Build (is_start, is_end, oos_start, oos_end) tuples.

        yfinance 5m data limit: 60 days. So each window = 60 trading days total.
        IS = first 70% (~42d), OOS = last 30% (~18d).
        Windows step by step_months converted to ~21 trading days/month.
        """
        today = date.today()
        # yfinance 5m hard limit is 60 calendar days from today (~42 trading days).
        # We split those 42 usable trading days into IS/OOS windows.
        # With 70/30 split and step = OOS length, we get overlapping but non-OOS-reusing windows.
        max_trading_days = 40        # safe 5m lookback in trading days (~58 calendar days)
        trading_days_per_window = max_trading_days
        is_days = int(trading_days_per_window * self.is_pct)   # ~29
        oos_days = trading_days_per_window - is_days            # ~13
        step_trading_days = max(oos_days, 5)                    # step by OOS length

        windows: list[tuple[date, date, date, date]] = []
        max_windows = 6  # practical cap given data constraints

        daily = self._fetch_daily_calendar()
        if daily is None:
            return []

        cutoff_start = today - timedelta(days=60)
        trading_dates = sorted(d for d in daily.index.date if cutoff_start <= d < today)

        if len(trading_dates) < trading_days_per_window:
            return []

        start_idx = 0
        while start_idx + trading_days_per_window <= len(trading_dates):
            window_dates = trading_dates[start_idx: start_idx + trading_days_per_window]
            is_dates = window_dates[:is_days]
            oos_dates = window_dates[is_days:]

            if len(is_dates) >= 8 and len(oos_dates) >= 4:
                windows.append((
                    is_dates[0], is_dates[-1],
                    oos_dates[0], oos_dates[-1],
                ))

            start_idx += step_trading_days
            if len(windows) >= max_windows:
                break

        return windows

    def _fetch_daily_calendar(self) -> pd.DataFrame | None:
        try:
            period = f"{self.years}y"
            df = yf.download(self.symbol, period=period, interval="1d", progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.warning("Failed to fetch daily calendar: %s", e)
            return None

    # ── Single window ─────────────────────────────────────────────────────────

    def _run_window(
        self,
        idx: int,
        is_start: date, is_end: date,
        oos_start: date, oos_end: date,
    ) -> WindowResult | None:
        # Convert date ranges to yfinance period strings — we'll use date-range download
        try:
            is_result = self._backtest_daterange(is_start, is_end)
            oos_result = self._backtest_daterange(oos_start, oos_end)
        except Exception as e:
            logger.warning("Window %d backtest failed: %s", idx, e)
            return None

        is_m = is_result.get("metrics", {})
        oos_m = oos_result.get("metrics", {})

        is_trades = is_m.get("total_trades", 0)
        oos_trades = oos_m.get("total_trades", 0)

        if is_trades < _MIN_IS_TRADES:
            logger.debug("Window %d skipped — IS only %d trades", idx, is_trades)
            return None

        is_days = (is_end - is_start).days
        oos_days = (oos_end - oos_start).days

        is_cagr = _cagr(is_m.get("total_return_pct", 0) / 100, is_days)
        oos_cagr = _cagr(oos_m.get("total_return_pct", 0) / 100, oos_days)

        # WFE
        if is_cagr <= 0:
            wfe = 0.0
        elif oos_trades < _MIN_OOS_TRADES:
            wfe = 0.0
        else:
            wfe = oos_cagr / is_cagr

        wfe_label = _wfe_label(wfe, is_cagr)

        return WindowResult(
            window_idx=idx,
            is_start=is_start, is_end=is_end,
            oos_start=oos_start, oos_end=oos_end,
            is_trades=is_trades,
            is_total_pnl=is_m.get("total_pnl", 0),
            is_cagr=is_cagr,
            is_win_rate=is_m.get("win_rate", 0),
            is_profit_factor=is_m.get("profit_factor", 0),
            is_max_drawdown=is_m.get("max_drawdown_pct", 0),
            oos_trades=oos_trades,
            oos_total_pnl=oos_m.get("total_pnl", 0),
            oos_cagr=oos_cagr,
            oos_win_rate=oos_m.get("win_rate", 0),
            oos_profit_factor=oos_m.get("profit_factor", 0),
            oos_max_drawdown=oos_m.get("max_drawdown_pct", 0),
            wfe=wfe,
            wfe_label=wfe_label,
            is_equity_curve=is_result.get("equity_curve", []),
            oos_equity_curve=oos_result.get("equity_curve", []),
        )

    def _backtest_daterange(self, start: date, end: date) -> dict[str, Any]:
        """Run backtest for a specific date range by downloading 5m data for that span."""
        from app.services.strategy.daytrading.runner import run_backtest

        # yfinance accepts start/end strings for intraday (within 60d limit)
        days_back = (date.today() - start).days
        # Use period close to the window — yfinance doesn't support arbitrary start/end
        # for intraday well, so we download enough and slice by date
        period = _days_to_period(days_back + (end - start).days + 5)

        result = run_backtest(
            symbol=self.symbol,
            strategy_name=self.strategy_name,
            period=period,
            initial_capital=self.initial_capital,
            position_pct=self.position_pct,
        )

        # Slice trades to the date window
        trades = result.get("trades", [])
        sliced = [
            t for t in trades
            if start <= _parse_date(t.get("date", "")) <= end
        ]

        if not sliced:
            return {"metrics": {"total_trades": 0, "total_return_pct": 0}, "equity_curve": [self.initial_capital]}

        return _recompute_from_trades(sliced, self.initial_capital)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, result: WalkForwardResult) -> None:
        wins = result.windows
        if not wins:
            result.robustness_label = "insufficient_data"
            return

        wfes = [w.wfe for w in wins if w.is_cagr > 0]
        if not wfes:
            result.robustness_label = "unprofitable_is"
            return

        result.avg_wfe = sum(wfes) / len(wfes)
        result.median_wfe = float(pd.Series(wfes).median())
        result.pct_windows_robust = sum(1 for w in wfes if w >= 0.6) / len(wfes)
        result.total_oos_trades = sum(w.oos_trades for w in wins)

        oos_wins_total = sum(w.oos_trades * (w.oos_win_rate / 100) for w in wins if w.oos_trades > 0)
        result.oos_win_rate = (oos_wins_total / result.total_oos_trades * 100) if result.total_oos_trades > 0 else 0

        # Chained OOS CAGR: compound all OOS returns
        oos_returns = [w.oos_cagr for w in wins if w.oos_trades >= _MIN_OOS_TRADES]
        if oos_returns:
            total_days = sum((w.oos_end - w.oos_start).days for w in wins if w.oos_trades >= _MIN_OOS_TRADES)
            combined_return = 1.0
            for r in oos_returns:
                days = 365 / 4   # approx per window
                combined_return *= (1 + r) ** (days / 365)
            result.oos_cagr = combined_return - 1.0
        else:
            result.oos_cagr = 0.0

        if result.avg_wfe >= 0.6:
            result.robustness_label = "robust"
        elif result.avg_wfe >= 0.3:
            result.robustness_label = "marginal"
        else:
            result.robustness_label = "overfit"


# ── Module-level convenience ──────────────────────────────────────────────────

def run_walkforward(
    symbol: str,
    strategy_name: str,
    years: int = 2,
    is_pct: float = 0.70,
    step_months: int = 3,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.95,
) -> WalkForwardResult:
    """Convenience wrapper — creates validator and runs it."""
    v = WalkForwardValidator(
        symbol=symbol,
        strategy_name=strategy_name,
        years=years,
        is_pct=is_pct,
        step_months=step_months,
        initial_capital=initial_capital,
        position_pct=position_pct,
    )
    result = v.run()
    logger.info("\n%s", result.summary())
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cagr(total_return: float, days: int) -> float:
    if days <= 0:
        return 0.0
    years = days / 365.0
    try:
        return (1 + total_return) ** (1 / years) - 1
    except Exception:
        return 0.0


def _wfe_label(wfe: float, is_cagr: float) -> str:
    if is_cagr <= 0:
        return "unprofitable"
    if wfe >= 0.6:
        return "robust"
    if wfe >= 0.3:
        return "marginal"
    return "overfit"


def _days_to_period(days: int) -> str:
    if days <= 7:
        return "7d"
    if days <= 30:
        return "30d"
    if days <= 60:
        return "60d"
    return "60d"   # yfinance 5m hard cap


def _parse_date(date_str: str) -> date:
    try:
        return date.fromisoformat(date_str[:10])
    except Exception:
        return date.min


def _recompute_from_trades(trades: list[dict], initial_capital: float) -> dict[str, Any]:
    """Rebuild metrics from a sliced trade list."""
    if not trades:
        return {"metrics": {"total_trades": 0, "total_return_pct": 0}, "equity_curve": [initial_capital]}

    df = pd.DataFrame(trades)
    equity = initial_capital
    curve = [equity]
    for pnl in df["pnl"]:
        equity += pnl
        curve.append(round(equity, 2))

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gross_profit = wins["pnl"].sum() if not wins.empty else 0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return_pct = (equity - initial_capital) / initial_capital * 100

    peak = initial_capital
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "trades": trades,
        "equity_curve": curve,
        "metrics": {
            "total_trades": len(df),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(df["pnl"].sum(), 2),
            "total_return_pct": round(total_return_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "initial_capital": initial_capital,
            "final_equity": round(equity, 2),
        },
    }
