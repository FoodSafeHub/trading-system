"""
The max-position-size limit must CLAMP a BUY down to fit, not reject it.

Regression: HLT (5-share cap × ~$332 = $1,660, cash-trimmed to 4 sh = $1,332)
was hard-BLOCKED by max_position_size_usd=$1,000 and never traded. It should
instead shrink to the largest whole-share qty that fits ($996 = 3 sh).
"""
from __future__ import annotations

import pytest

from app.services.execution.service import ExecutionService


class _Broker:
    def __init__(self, name):
        self.name = name


def _svc(broker_name="schwab"):
    return ExecutionService(_Broker(broker_name))


def test_quantize_floors_to_whole_shares_for_live_broker():
    svc = _svc("schwab")
    # 1000 / 332.24 = 3.01 → 3 whole shares (never rounds up past the limit).
    assert svc._quantize_qty(1000 / 332.24) == 3.0


def test_quantize_zero_when_under_one_share():
    svc = _svc("schwab")
    # A stock pricier than the whole limit → 0 shares fit → order will block.
    assert svc._quantize_qty(0.75) == 0.0


def test_quantize_allows_fractions_for_paper():
    svc = _svc("paper")
    q = svc._quantize_qty(1000 / 332.24)
    assert 3.0 < q < 3.02  # paper keeps the fraction


def test_hlt_scenario_fits_under_limit():
    # The clamp target for HLT: floor(max_usd / price) shares, value < limit.
    svc = _svc("schwab")
    price, max_usd = 332.24, 1000.0
    fitted = svc._quantize_qty(max_usd / price)
    assert fitted == 3.0
    assert fitted * price < max_usd  # $996.72 < $1000
