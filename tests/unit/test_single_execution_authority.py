"""
Single-execution-authority invariant.

The system has ONE order-placement path: `app/services/strategy/scheduler.py`
(the auto-scheduler, every 15 min). All other code surfaces -- the scanner,
the manual `/strategy/run` endpoint, the dashboard -- are DISCOVERY ONLY:
they record signals to the DB, the scheduler decides whether to act.

This invariant prevents the architectural drift that caused the BNY incident
(2026-06-04): scanner_service had its own auto-trade path that bypassed the
scheduler's assignment-cap and tight-trail policy. When the user toggled
"Auto-trade top match" on the Scanner page, the scanner placed a MARKET
SELL out-of-band.

The test fails when a new module starts calling:
    - ExecutionService.execute / svc.execute
    - ExecutionService.tighten_trail_on_sell
    - broker.place_order

...outside the allowlist.

If a future commit legitimately needs to place orders from a new module,
add that module to ALLOWLIST_SUFFIXES with a comment explaining why
(an audited path; a user-initiated manual order route; etc.).
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Only scan high-level orchestration modules. Broker drivers (passthroughs)
# and DB helpers (SQLAlchemy .execute) are outside the scope of this guard.
SCAN_ROOTS = (
    REPO_ROOT / "app" / "services" / "strategy",
    REPO_ROOT / "app" / "services" / "scanner",
    REPO_ROOT / "app" / "api" / "routes",
)


# Modules where calling the broker / ExecutionService is INTENTIONAL.
# Paths relative to the repo root.
ALLOWLIST = frozenset({
    # THE single execution authority for strategy-driven orders.
    "app/services/strategy/scheduler.py",
    # User-driven manual order entry (POST /orders backing the dashboard's
    # "Place Order" form). Source is force-stamped "manual" so the audit
    # trail is clear; this is NOT a strategy signal path.
    "app/api/routes/orders.py",
    # Day-trading subsystem has its own dedicated execution path
    # (single_stock_trader / autotrader manager). It is not subject to the
    # perplexity / swing scheduler invariant -- different system.
    "app/services/strategy/daytrading/autotrader/single_stock_trader.py",
    "app/services/strategy/daytrading/autotrader/manager.py",
    "app/services/strategy/daytrading/runner.py",
    "app/services/strategy/daytrading/autotrader/native_entry.py",
    # Day-trading per-symbol position manager (uses self.broker.place_order
    # for chandelier exits inside its own loop, audited separately).
    "app/services/strategy/daytrading/autotrader/position_manager.py",
    "app/services/strategy/daytrading/autotrader/exit_manager.py",
})

# AST: any of these call patterns counts as a placement.
PLACEMENT_METHODS = {"tighten_trail_on_sell", "place_order"}
# `execute` is also a placement method on ExecutionService, but the name is
# too generic (SQLAlchemy sessions, asyncio loops, futures all have it).
# We catch the specific shape `<X>.execute(<OrderRequest-like first arg>)`
# or by allowlisting receiver names.
PLACEMENT_RECEIVERS_FOR_EXECUTE = {"svc", "exec_svc", "execution_service"}


def _is_placement_call(node: ast.AST) -> bool:
    """True if a node is `*.tighten_trail_on_sell(...)`, `*.place_order(...)`,
    or `(svc|exec_svc|execution_service).execute(...)`."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr in PLACEMENT_METHODS:
        return True
    if fn.attr == "execute":
        # Only count `.execute(...)` when the receiver name is one of the
        # known ExecutionService bindings -- avoids SQLAlchemy false positives.
        recv = fn.value
        recv_name = (
            recv.id if isinstance(recv, ast.Name)
            else recv.attr if isinstance(recv, ast.Attribute)
            else None
        )
        if recv_name in PLACEMENT_RECEIVERS_FOR_EXECUTE:
            return True
    return False


def _scan_module(path: Path) -> list[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[int] = []
    for node in ast.walk(tree):
        if _is_placement_call(node):
            hits.append(node.lineno)
    return hits


def _scanned_modules() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        out.extend(
            p for p in root.rglob("*.py")
            if "__pycache__" not in p.parts
        )
    return out


# ── Contract ────────────────────────────────────────────────────────────────


def test_only_scheduler_and_manual_route_may_place_orders():
    """AST-walk every orchestration module (strategy/, scanner/, api/routes/)
    and assert no module outside the allowlist contains a placement call.
    New side-paths fail with file:line so the reviewer can choose:
      (a) route the work through the scheduler, or
      (b) explicitly add the module to ALLOWLIST with a comment."""
    violations: list[tuple[Path, list[int]]] = []
    for path in _scanned_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        hits = _scan_module(path)
        if hits:
            violations.append((path, hits))

    if violations:
        lines = [
            "Found placement call(s) outside the single-execution-authority "
            "allowlist. The auto-scheduler is the only path that should "
            "place strategy-driven orders:"
        ]
        for path, hits in violations:
            rel = path.relative_to(REPO_ROOT).as_posix()
            for ln in hits:
                lines.append(f"  {rel}:{ln}")
        lines += [
            "",
            "If this call is intentional (e.g. an audited new execution path "
            "or a user-driven manual endpoint), add the path to ALLOWLIST "
            "with a comment. Otherwise, route the underlying signal through "
            "the scheduler so caps + tight-trail policy are honoured.",
        ]
        raise AssertionError("\n".join(lines))


def test_allowlist_entries_actually_exist():
    """Catch rot: every allowlist path must point to a real file."""
    missing = [
        rel for rel in ALLOWLIST
        if not (REPO_ROOT / rel).is_file()
    ]
    assert not missing, (
        f"Allowlist entries point to files that no longer exist: {missing}. "
        f"Update or remove them."
    )
