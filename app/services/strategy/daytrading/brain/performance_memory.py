"""
Performance Memory — rolling per-strategy, per-symbol, per-market-state tracker.

Tracks the last N trades per (strategy, symbol, market_state) slice and computes:
  - win rate
  - avg P&L %
  - profit factor
  - streak info

If a strategy is underperforming in the current market state (based on recent
history), the brain lowers its weight or disables it temporarily.

Storage is in-memory (Python dict) and survives the process lifetime.
For persistence across restarts, call load_from_records() with saved trade dicts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategySliceStats:
    strategy: str
    symbol: str
    market_state: str
    trades: int = 0
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    recent_pnl_pct: list[float] = field(default_factory=list)  # last 20

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades * 100 if self.trades > 0 else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return sum(self.recent_pnl_pct) / len(self.recent_pnl_pct) if self.recent_pnl_pct else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else (
            float("inf") if self.gross_profit > 0 else 0.0
        )

    @property
    def expectancy(self) -> float:
        if self.trades == 0:
            return 0.0
        wr = self.wins / self.trades
        avg_w = self.gross_profit / self.wins if self.wins > 0 else 0.0
        n_losses = self.trades - self.wins
        avg_l = self.gross_loss / n_losses if n_losses > 0 else 0.0
        return wr * avg_w - (1 - wr) * avg_l

    def is_underperforming(self, min_trades: int = 5, min_win_rate: float = 40.0) -> bool:
        if self.trades < min_trades:
            return False   # not enough data to judge
        return self.win_rate < min_win_rate or self.expectancy < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "market_state": self.market_state,
            "trades": self.trades,
            "win_rate": round(self.win_rate, 1),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 4),
            "is_underperforming": self.is_underperforming(),
        }


class PerformanceMemory:
    """
    In-memory rolling performance tracker.
    Key: (strategy, symbol, market_state)
    """

    def __init__(self, window: int = 20):
        self._window = window
        # (strategy, symbol, market_state) → StrategySliceStats
        self._stats: dict[tuple, StrategySliceStats] = defaultdict(
            lambda: StrategySliceStats(strategy="", symbol="", market_state="")
        )
        # Hour-of-day performance: (strategy, hour) → deque of pnl_pct
        self._hourly: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=window))

    def record_trade(self, trade: dict) -> None:
        """
        Add a completed trade to memory.
        trade must have: strategy, symbol, market_state (or regime), pnl, pnl_pct,
                         entry_time (for hour extraction).
        """
        strategy = trade.get("strategy", "")
        symbol = trade.get("symbol", "")
        # Accept both 'market_state' and 'regime' keys for compatibility
        market_state = trade.get("market_state") or trade.get("regime", "UNKNOWN")
        pnl = float(trade.get("pnl", 0))
        pnl_pct = float(trade.get("pnl_pct", 0))

        key = (strategy, symbol, market_state)
        s = self._stats[key]
        s.strategy = strategy
        s.symbol = symbol
        s.market_state = market_state
        s.trades += 1

        if pnl > 0:
            s.wins += 1
            s.gross_profit += pnl
        else:
            s.gross_loss += abs(pnl)

        s.recent_pnl_pct.append(pnl_pct)
        if len(s.recent_pnl_pct) > self._window:
            s.recent_pnl_pct.pop(0)

        # Hour-of-day tracking
        try:
            import pandas as pd
            hour = pd.Timestamp(trade.get("entry_time", "")).hour
            self._hourly[(strategy, hour)].append(pnl_pct)
        except Exception:
            pass

    def load_from_records(self, trades: list[dict]) -> None:
        """Replay a list of historical trades into memory."""
        for t in trades:
            self.record_trade(t)

    def get_strategy_weight(
        self,
        strategy: str,
        symbol: str,
        market_state: str,
        min_trades: int = 5,
    ) -> float:
        """
        Returns a weight 0.0–1.0.
        1.0 = full weight (good or insufficient history).
        0.5 = reduced (underperforming).
        0.0 = temporarily disabled (strongly negative expectancy).
        """
        key = (strategy, symbol, market_state)
        s = self._stats.get(key)
        if s is None or s.trades < min_trades:
            return 1.0   # not enough data — give benefit of the doubt

        if s.expectancy < -0.3:
            return 0.0   # strongly negative: disable
        if s.is_underperforming():
            return 0.5   # weak: half weight
        return 1.0

    def get_slice_stats(
        self, strategy: str, symbol: str, market_state: str
    ) -> StrategySliceStats | None:
        return self._stats.get((strategy, symbol, market_state))

    def get_all_stats(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._stats.values() if s.trades > 0]

    def get_best_hours(self, strategy: str, top_n: int = 3) -> list[dict]:
        """Return the top N hours of day for a given strategy."""
        results = []
        for (strat, hour), pnl_list in self._hourly.items():
            if strat != strategy or len(pnl_list) == 0:
                continue
            avg = sum(pnl_list) / len(pnl_list)
            wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
            results.append({"hour": hour, "trades": len(pnl_list), "avg_pnl": round(avg, 4), "win_rate": round(wr, 1)})
        results.sort(key=lambda x: x["avg_pnl"], reverse=True)
        return results[:top_n]

    def explain_weight(
        self, strategy: str, symbol: str, market_state: str
    ) -> str:
        """Human-readable explanation of why a strategy has its current weight."""
        key = (strategy, symbol, market_state)
        s = self._stats.get(key)
        if s is None or s.trades == 0:
            return f"No history for {strategy} on {symbol} in {market_state} — full weight assumed."
        w = self.get_strategy_weight(strategy, symbol, market_state)
        if w == 0.0:
            return (
                f"{strategy} on {symbol}/{market_state}: {s.trades} trades, "
                f"win rate {s.win_rate:.0f}%, expectancy {s.expectancy:.3f}% — DISABLED (negative expectancy)."
            )
        if w == 0.5:
            return (
                f"{strategy} on {symbol}/{market_state}: {s.trades} trades, "
                f"win rate {s.win_rate:.0f}% — REDUCED (below 40% win rate threshold)."
            )
        return (
            f"{strategy} on {symbol}/{market_state}: {s.trades} trades, "
            f"win rate {s.win_rate:.0f}%, PF {s.profit_factor:.2f} — FULL weight."
        )


# Module-level singleton — shared across the process lifetime
_global_memory = PerformanceMemory(window=20)


def get_global_memory() -> PerformanceMemory:
    return _global_memory
