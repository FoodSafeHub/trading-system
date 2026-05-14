"""
FillSimulator — realistic fill modeling for backtests and paper trading.

Models three fill costs:
  commission   — fixed per-share or per-trade cost
  slippage     — market impact as basis points of entry price
  spread       — bid/ask half-spread (default 0 since yfinance gives mid)

Usage:
  sim = FillSimulator(FillConfig(commission_per_share=0.005, slippage_bps=2.0))
  fill = sim.fill_entry("BUY", entry_price=170.50, qty=100)
  fill = sim.fill_exit("BUY", exit_price=171.85, qty=100)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FillConfig:
    commission_per_share: float = 0.0    # Alpaca = $0; IBKR ≈ $0.005
    commission_min: float = 0.0          # minimum per order
    slippage_bps: float = 2.0            # 2 bps = 0.02% — conservative for liquid names
    spread_bps: float = 0.5              # half-spread estimate
    fill_model: Literal["market", "limit", "vwap"] = "market"

    # Slippage multiplier by volatility bucket (applied to slippage_bps)
    low_vol_mult: float = 0.5    # stable names: half slippage
    high_vol_mult: float = 2.0   # volatile names: double slippage


@dataclass
class FillResult:
    side: str                  # "BUY" or "SELL"
    qty: float
    raw_price: float           # price before costs
    fill_price: float          # effective price after slippage/spread
    commission: float
    slippage_cost: float
    total_cost: float          # commission + slippage (dollar amount)
    cost_bps: float            # total cost in basis points

    @property
    def gross_value(self) -> float:
        return self.raw_price * self.qty

    @property
    def net_value(self) -> float:
        """What you actually pay (BUY) or receive (SELL) after all costs."""
        if self.side == "BUY":
            return self.gross_value + self.total_cost
        return self.gross_value - self.total_cost


@dataclass
class TradeFillSummary:
    symbol: str
    direction: str
    qty: float
    entry_fill: FillResult
    exit_fill: FillResult
    gross_pnl: float          # ignoring costs
    net_pnl: float            # after all costs
    total_commission: float
    total_slippage: float
    round_trip_cost_bps: float

    @property
    def cost_drag_pct(self) -> float:
        return self.round_trip_cost_bps / 100.0


class FillSimulator:
    """
    Stateless fill simulator. Call fill_entry() and fill_exit() for each leg,
    then summarize_trade() for the net result.
    """

    def __init__(self, config: FillConfig | None = None):
        self.config = config or FillConfig()

    def fill_entry(
        self,
        side: str,
        raw_price: float,
        qty: float,
        volatility_pct: float = 1.0,
    ) -> FillResult:
        """
        Simulate the entry fill cost.
        BUY: slippage moves price UP (you pay more).
        SELL short: slippage moves price DOWN (you receive less).
        """
        return self._fill(side, raw_price, qty, volatility_pct, is_entry=True)

    def fill_exit(
        self,
        direction: str,
        raw_price: float,
        qty: float,
        volatility_pct: float = 1.0,
    ) -> FillResult:
        """
        Simulate the exit fill cost.
        BUY exit (close long = SELL): slippage moves price DOWN.
        SELL exit (cover short = BUY): slippage moves price UP.
        """
        exit_side = "SELL" if direction == "BUY" else "BUY"
        return self._fill(exit_side, raw_price, qty, volatility_pct, is_entry=False)

    def summarize_trade(
        self,
        symbol: str,
        direction: str,
        qty: float,
        entry_fill: FillResult,
        exit_fill: FillResult,
    ) -> TradeFillSummary:
        if direction == "BUY":
            gross_pnl = (exit_fill.raw_price - entry_fill.raw_price) * qty
        else:
            gross_pnl = (entry_fill.raw_price - exit_fill.raw_price) * qty

        total_cost = entry_fill.total_cost + exit_fill.total_cost
        net_pnl = gross_pnl - total_cost
        round_trip_bps = entry_fill.cost_bps + exit_fill.cost_bps

        return TradeFillSummary(
            symbol=symbol,
            direction=direction,
            qty=qty,
            entry_fill=entry_fill,
            exit_fill=exit_fill,
            gross_pnl=round(gross_pnl, 4),
            net_pnl=round(net_pnl, 4),
            total_commission=round(entry_fill.commission + exit_fill.commission, 4),
            total_slippage=round(entry_fill.slippage_cost + exit_fill.slippage_cost, 4),
            round_trip_cost_bps=round(round_trip_bps, 2),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fill(
        self,
        side: str,
        raw_price: float,
        qty: float,
        volatility_pct: float,
        is_entry: bool,
    ) -> FillResult:
        cfg = self.config

        # Volatility-adjusted slippage
        if volatility_pct < 1.0:
            vol_mult = cfg.low_vol_mult
        elif volatility_pct > 1.8:
            vol_mult = cfg.high_vol_mult
        else:
            vol_mult = 1.0

        slip_bps = cfg.slippage_bps * vol_mult
        spread_bps = cfg.spread_bps

        total_bps = slip_bps + spread_bps
        slippage_per_share = raw_price * total_bps / 10_000

        # Adverse fill: BUY pays more, SELL receives less
        if side == "BUY":
            fill_price = raw_price + slippage_per_share
        else:
            fill_price = raw_price - slippage_per_share

        commission = max(cfg.commission_min, cfg.commission_per_share * qty)
        slippage_cost = abs(fill_price - raw_price) * qty
        total_cost = commission + slippage_cost
        cost_bps = total_cost / (raw_price * qty) * 10_000 if raw_price > 0 and qty > 0 else 0.0

        return FillResult(
            side=side,
            qty=qty,
            raw_price=round(raw_price, 4),
            fill_price=round(fill_price, 4),
            commission=round(commission, 4),
            slippage_cost=round(slippage_cost, 4),
            total_cost=round(total_cost, 4),
            cost_bps=round(cost_bps, 2),
        )
