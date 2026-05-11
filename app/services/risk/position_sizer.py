from __future__ import annotations

"""
Position Sizer — calculates share quantity based on fixed fractional risk.

Formula:
    risk_per_trade  = account_value * risk_pct_per_trade
    stop_distance   = entry_price - stop_price
    shares          = risk_per_trade / stop_distance

Caps applied (in order):
  1. Max position size in USD (from settings / override)
  2. Max account risk open (sum of all open position risks)
  3. Minimum shares threshold (skip if result is too small to bother)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizeResult:
    symbol: str
    entry_price: float
    stop_price: float
    shares: float                   # recommended quantity (fractional)
    position_value: float           # entry_price * shares
    risk_amount: float              # dollars at risk on this trade
    risk_pct_of_account: float      # % of account risked
    stop_distance: float            # entry - stop in dollars
    stop_distance_pct: float        # stop distance as % of entry
    capped: bool                    # True if a cap was applied
    cap_reason: str                 # which cap fired
    viable: bool                    # False if trade should be skipped
    skip_reason: str                # why it was skipped


def calculate_position_size(
    symbol: str,
    entry_price: float,
    stop_price: float,
    account_value: float,
    risk_pct_per_trade: float = 0.01,       # 1% of account per trade
    max_position_size_usd: float = 10_000.0,
    max_account_risk_pct: float = 0.06,     # never risk more than 6% total open
    current_open_risk_usd: float = 0.0,     # sum of risk already committed in open positions
    min_shares: float = 0.001,
) -> SizeResult:
    """
    Returns how many shares to buy given entry, stop, and account size.

    Example:
        account = $50,000, risk 1% = $500
        entry = $100, stop = $95  → stop_distance = $5
        shares = $500 / $5 = 100 shares  → position = $10,000
    """
    if entry_price <= 0:
        return _skip(symbol, entry_price, stop_price, "entry price is zero")

    if stop_price >= entry_price:
        return _skip(symbol, entry_price, stop_price,
                     f"stop ${stop_price:.2f} is above entry ${entry_price:.2f} — invalid")

    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return _skip(symbol, entry_price, stop_price, "stop distance is zero")

    stop_distance_pct = stop_distance / entry_price * 100

    # Stop too tight (< 0.1%) or too wide (> 20%) — likely bad data
    if stop_distance_pct < 0.1:
        return _skip(symbol, entry_price, stop_price,
                     f"stop too tight ({stop_distance_pct:.2f}% — likely bad signal)")
    if stop_distance_pct > 20:
        return _skip(symbol, entry_price, stop_price,
                     f"stop too wide ({stop_distance_pct:.1f}% — risk too large)")

    risk_per_trade = account_value * risk_pct_per_trade
    shares = risk_per_trade / stop_distance
    position_value = shares * entry_price
    capped = False
    cap_reason = ""

    # Cap 1: max position size in USD
    if position_value > max_position_size_usd:
        shares = max_position_size_usd / entry_price
        position_value = shares * entry_price
        risk_amount = shares * stop_distance
        capped = True
        cap_reason = f"capped at max position size ${max_position_size_usd:,.0f}"

    risk_amount = shares * stop_distance

    # Cap 2: total open account risk
    max_open_risk_usd = account_value * max_account_risk_pct
    if current_open_risk_usd + risk_amount > max_open_risk_usd:
        remaining_risk = max_open_risk_usd - current_open_risk_usd
        if remaining_risk <= 0:
            return _skip(symbol, entry_price, stop_price,
                         f"max account risk {max_account_risk_pct*100:.0f}% already committed "
                         f"(${current_open_risk_usd:,.0f} open)")
        shares = remaining_risk / stop_distance
        position_value = shares * entry_price
        risk_amount = shares * stop_distance
        capped = True
        cap_reason = f"capped by max open risk {max_account_risk_pct*100:.0f}% of account"

    # Minimum shares check
    if shares < min_shares:
        return _skip(symbol, entry_price, stop_price,
                     f"position too small ({shares:.4f} shares < minimum {min_shares})")

    return SizeResult(
        symbol=symbol,
        entry_price=round(entry_price, 4),
        stop_price=round(stop_price, 4),
        shares=round(shares, 4),
        position_value=round(position_value, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct_of_account=round(risk_amount / account_value * 100, 3),
        stop_distance=round(stop_distance, 4),
        stop_distance_pct=round(stop_distance_pct, 2),
        capped=capped,
        cap_reason=cap_reason,
        viable=True,
        skip_reason="",
    )


def _skip(symbol: str, entry: float, stop: float, reason: str) -> SizeResult:
    return SizeResult(
        symbol=symbol,
        entry_price=entry,
        stop_price=stop,
        shares=0.0,
        position_value=0.0,
        risk_amount=0.0,
        risk_pct_of_account=0.0,
        stop_distance=max(0.0, entry - stop),
        stop_distance_pct=0.0,
        capped=False,
        cap_reason="",
        viable=False,
        skip_reason=reason,
    )
