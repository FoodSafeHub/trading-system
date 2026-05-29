from __future__ import annotations

"""
Backtest cost model — Phase 0 scaffolding.

A `CostModel` worsens fills (slippage + half-spread) and charges commission /
taxes so backtests can report NET rather than gross results. It is plumbed into
`run_backtest` as an optional argument that DEFAULTS TO None (zero cost), so no
existing result moves until a caller explicitly opts in (Phase 2).

The all-zero instance is an exact identity: `apply_buy(p) == p`,
`apply_sell(p) == p`, and every commission/tax is 0.0. The engine relies on
this for byte-identical behaviour when cost_model is None or ZERO.

Units:
  *_bps fields are basis points (1 bp = 0.01%).
  commission_per_share is currency per share (US-style).
  commission_bps is a % of notional (India-style brokerage).
  taxes_bps is charged on the SELL side only (STT/stamp/GST/exchange, or US
  SEC/TAF), expressed in bps of notional.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    slippage_bps: float = 0.0          # adverse price move on every fill
    half_spread_bps: float = 0.0       # bid/ask half-spread, added to slippage
    commission_per_share: float = 0.0  # US-style per-share commission
    commission_bps: float = 0.0        # India-style % of notional
    min_commission: float = 0.0        # floor applied to per-trade commission
    taxes_bps: float = 0.0             # SELL-side taxes (STT/stamp/GST/exch)
    adv_cap_pct: float = 0.0           # 0 = off; else cap size to % of ADV$
    impact_coeff: float = 0.0          # extra slippage = impact_coeff*(size/ADV$)

    # ── Fill-price adjustments ────────────────────────────────────────────────
    def _slip_frac(self) -> float:
        return (self.slippage_bps + self.half_spread_bps) / 10_000.0

    def apply_buy(self, price: float) -> float:
        """A buy fills WORSE (higher) by the slippage+half-spread fraction."""
        return price * (1.0 + self._slip_frac())

    def apply_sell(self, price: float) -> float:
        """A sell fills WORSE (lower) by the slippage+half-spread fraction."""
        return price * (1.0 - self._slip_frac())

    # ── Commission / taxes ────────────────────────────────────────────────────
    def _commission(self, qty: float, notional: float) -> float:
        comm = abs(qty) * self.commission_per_share
        comm += abs(notional) * (self.commission_bps / 10_000.0)
        if self.min_commission > 0.0 and comm > 0.0:
            comm = max(comm, self.min_commission)
        return comm

    def entry_commission(self, qty: float, notional: float) -> float:
        return self._commission(qty, notional)

    def exit_commission(self, qty: float, notional: float) -> float:
        taxes = abs(notional) * (self.taxes_bps / 10_000.0)
        return self._commission(qty, notional) + taxes

    # ── Liquidity ─────────────────────────────────────────────────────────────
    def is_liquid(self, order_notional: float, adv_dollar: float) -> bool:
        """True when the order is within the ADV cap (cap off => always True)."""
        if self.adv_cap_pct <= 0.0 or adv_dollar <= 0.0:
            return True
        return order_notional <= adv_dollar * (self.adv_cap_pct / 100.0)


# ── Presets (declared now; NOT wired into any route in Phase 0) ────────────────
ZERO = CostModel()

US_DEFAULT = CostModel(
    slippage_bps=2.0,
    half_spread_bps=1.0,
    commission_per_share=0.005,
    min_commission=1.0,
    taxes_bps=0.1,
)

INDIA_DEFAULT = CostModel(
    slippage_bps=15.0,
    half_spread_bps=5.0,
    commission_bps=3.0,
    min_commission=0.0,
    taxes_bps=10.0,
)
