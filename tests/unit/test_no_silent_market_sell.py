"""
Repository-wide guard against silent SELL MARKET orders in live execution paths.

The BNY incident (2026-06-04) traced back to TWO escape hatches that bypassed
the tight-trail Approach C:
  1. scanner_service._auto_trade_top placed a MARKET SELL directly when the
     scan auto-trade toggle was on and a SELL candidate was top-ranked.
  2. ExecutionService.tighten_trail_on_sell silently fell back to a MARKET
     SELL when broker placement of the TRAILING_STOP raised.

Both are now closed. This test prevents new ones by AST-walking every
production module and failing when a literal
    OrderRequest(side="SELL", order_type="MARKET")
appears outside an explicit allowlist.

Allowlist criteria:
  - Backtest / simulator modules (no live broker effect).
  - Broker driver internals (those are translation layers, not signal-to-order).

If a future commit adds a SELL MARKET line in a NON-allowlisted live module,
this test will fail with the file:line so the reviewer can choose whether
that path should use tighten_trail_on_sell instead.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICES = REPO_ROOT / "app" / "services"


# Modules where a literal SELL MARKET is INTENTIONAL and reviewed.
# Keep this list SHORT; every addition needs explicit justification.
ALLOWLIST_SUFFIXES = (
    # Backtest engines simulate fills; no real broker effect.
    "backtest/engine.py",
    "backtest/perplexity_engine.py",
    "backtest/portfolio_engine.py",
    "backtest/consensus_engine.py",
    "strategy/daytrading/execution/fill_simulator.py",
    # Broker driver translation layers (just naming the side they ultimately submit).
    "broker/alpaca_broker.py",
    # Tests aren't scanned (tests/ dir excluded below) but be explicit.
)


def _is_sell_market_order_request(node: ast.AST) -> bool:
    """True when a node is `OrderRequest(side='SELL', order_type='MARKET', ...)`.
    Conservative: only fires on LITERAL constants — a dynamic `side=direction`
    binding is not flagged."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    fn_name = (
        fn.id if isinstance(fn, ast.Name)
        else fn.attr if isinstance(fn, ast.Attribute)
        else None
    )
    if fn_name != "OrderRequest":
        return False
    side = order_type = None
    for kw in node.keywords:
        if kw.arg == "side" and isinstance(kw.value, ast.Constant):
            side = kw.value.value
        elif kw.arg == "order_type" and isinstance(kw.value, ast.Constant):
            order_type = kw.value.value
    return side == "SELL" and order_type == "MARKET"


def _scan_module(path: Path) -> list[int]:
    """Return line numbers of any SELL MARKET OrderRequest in the module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[int] = []
    for node in ast.walk(tree):
        if _is_sell_market_order_request(node):
            hits.append(node.lineno)
    return hits


def _all_service_modules() -> list[Path]:
    return [p for p in SERVICES.rglob("*.py") if "__pycache__" not in p.parts]


# ── Contract ────────────────────────────────────────────────────────────────


def test_no_silent_sell_market_orders_in_live_modules():
    """Walk every app/services/**/*.py and assert no literal SELL/MARKET
    OrderRequest construction exists outside the allowlist."""
    violations: list[tuple[Path, list[int]]] = []
    for path in _all_service_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.endswith(suffix) for suffix in ALLOWLIST_SUFFIXES):
            continue
        hits = _scan_module(path)
        if hits:
            violations.append((path, hits))

    if violations:
        lines = ["Found literal SELL/MARKET OrderRequest in live execution path(s):"]
        for path, hits in violations:
            rel = path.relative_to(REPO_ROOT).as_posix()
            for ln in hits:
                lines.append(f"  {rel}:{ln}")
        lines.append("")
        lines.append(
            "If this is intentional (e.g. an audited fallback), add the file "
            "suffix to ALLOWLIST_SUFFIXES in this test with a comment. "
            "Otherwise, route through ExecutionService.tighten_trail_on_sell() "
            "so the SELL becomes a TRAILING_STOP instead of a market dump."
        )
        raise AssertionError("\n".join(lines))


def test_allowlist_entries_actually_exist():
    """The allowlist should not rot. If a file is renamed/moved, the
    suffix becomes a dead entry and the guard silently loses coverage."""
    missing = []
    for suffix in ALLOWLIST_SUFFIXES:
        candidates = list(REPO_ROOT.rglob(suffix.split("/")[-1]))
        matched = any(
            c.relative_to(REPO_ROOT).as_posix().endswith(suffix)
            for c in candidates
        )
        if not matched:
            missing.append(suffix)
    assert not missing, (
        f"Allowlist entries refer to files that no longer exist: {missing}. "
        f"Remove them or update the paths."
    )
