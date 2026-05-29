from __future__ import annotations

"""Relative-strength rotation API (Phase 1, additive).

New endpoints only — does not touch scanner/recommendations/backtest output.
Surfaces the standalone rs_rotation service (rank + portfolio backtest harness).
"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.services.strategy.rs_rotation import (
    backtest_rs_rotation, rank_relative_strength,
)

router = APIRouter(prefix="/rs-rotation", tags=["rs-rotation"])


def _parse_symbols(symbols: str) -> list[str]:
    syms = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="symbols (comma-separated) is required")
    return syms


@router.get("/rank")
def rank(symbols: str, benchmark: str = "SPY", period: str = "2y"):
    """Rank a comma-separated basket by relative strength vs the benchmark."""
    ranked = rank_relative_strength(_parse_symbols(symbols), benchmark, period)
    return [asdict(r) for r in ranked]


@router.get("/backtest")
def backtest(
    symbols: str,
    benchmark: str = "SPY",
    period: str = "5y",
    top_n: int = 5,
    rebalance_days: int = 21,
    initial_capital: float = 100_000.0,
):
    """Backtest the equal-weight top-N rotation over the basket."""
    try:
        result = backtest_rs_rotation(
            _parse_symbols(symbols), benchmark_symbol=benchmark, period=period,
            top_n=top_n, rebalance_days=rebalance_days, initial_capital=initial_capital,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return asdict(result)
