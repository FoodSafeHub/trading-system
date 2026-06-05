"""
Routing tests for SELL signals in the scheduler.

These are intentionally STRUCTURAL tests: they assert that every SELL-direction
branch in `_run_cycle` ends in a call to `tighten_trail_on_sell` followed by
`continue` before reaching the generic MARKET order block. The reason they're
structural is that `_run_cycle` is a 400-line async function bound to the DB,
the broker, and an event loop — a runtime unit test would need to mock all of
them, and the bug we're guarding against is *exactly* the kind a runtime mock
can hide: a code path that exists but is never reached by a happy-path mock.

What broke before:
    The consensus SELL path computed qty=held then DROPPED THROUGH to the
    generic `OrderRequest(order_type="MARKET")` block at the bottom of the
    loop body. The assigned SELL path had a `continue` to skip that block;
    consensus did not. Result: on BNY (2026-06-04 14:16 ET) the scheduler
    placed a MARKET sell instead of a TRAILING_STOP.

These tests inspect the AST of `_run_cycle` to make sure both SELL branches
have a `tighten_trail_on_sell(...)` call followed by a `continue` statement
before any generic `OrderRequest` could be built. Adding a new SELL branch
that forgets the `continue` will fail these tests.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from app.services.strategy import scheduler as sched


# ── AST helpers ──────────────────────────────────────────────────────────────


def _find_run_cycle() -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(sched._run_cycle))
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_run_cycle"),
        None,
    )
    assert fn is not None, "_run_cycle not found in scheduler.py"
    return fn


def _is_tighten_trail_call(node: ast.AST) -> bool:
    """True if a node is `... tighten_trail_on_sell(...)`."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr == "tighten_trail_on_sell":
        return True
    return False


def _is_market_sell_order_request(node: ast.AST) -> bool:
    """True if a node looks like `OrderRequest(side='SELL', order_type='MARKET', ...)`."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if not (isinstance(fn, ast.Name) and fn.id == "OrderRequest"):
        return False
    side = order_type = None
    for kw in node.keywords:
        if kw.arg == "side":
            if isinstance(kw.value, ast.Constant):
                side = kw.value.value
            elif isinstance(kw.value, ast.Name):
                side = kw.value.id  # dynamic side variable
        if kw.arg == "order_type" and isinstance(kw.value, ast.Constant):
            order_type = kw.value.value
    # If side is a variable named "direction" we can't statically know — treat
    # as risky and let the dedicated trail-then-continue test cover it.
    return side == "SELL" and order_type == "MARKET"


def _walk_sell_branches(fn: ast.FunctionDef):
    """Yield every `if direction == "SELL"` / `if action == "SELL"` block."""
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "SELL"
            ):
                yield node


# ── Contract assertions ─────────────────────────────────────────────────────


def test_every_sell_branch_calls_tighten_trail_on_sell():
    """For each SELL block in _run_cycle, there must be at least one call to
    `tighten_trail_on_sell(...)` inside it. The assigned-strategy path and
    the consensus path are both expected to."""
    fn = _find_run_cycle()
    sell_branches = list(_walk_sell_branches(fn))
    assert sell_branches, (
        "no `if direction == 'SELL'` branches found — has the routing been "
        "refactored? Update this test."
    )
    branches_missing = []
    for branch in sell_branches:
        has_trail = any(_is_tighten_trail_call(n) for n in ast.walk(branch))
        if not has_trail:
            branches_missing.append(branch.lineno)
    assert not branches_missing, (
        f"SELL branch(es) at line(s) {branches_missing} do NOT call "
        f"tighten_trail_on_sell. They would fall through to a MARKET sell "
        f"(this was the BNY bug)."
    )


def test_every_sell_branch_continues_before_generic_market_block():
    """Each SELL branch must reach a top-level `continue` before any
    OrderRequest(order_type='MARKET') could be evaluated. We approximate
    this by checking: a `continue` appears in the branch's direct body
    AFTER the tighten_trail_on_sell call."""
    fn = _find_run_cycle()
    branches_missing = []
    for branch in _walk_sell_branches(fn):
        # Walk the branch's `body` top-to-bottom. Find the tighten_trail call
        # in any nested scope, then check that the same branch also ends in
        # a `continue` at one of its top-level statements.
        has_trail = any(_is_tighten_trail_call(n) for n in ast.walk(branch))
        if not has_trail:
            continue  # other test handles this case
        # `continue` must appear in the branch's direct body (so loop iteration
        # actually skips the generic block).
        def _has_top_level_continue(node) -> bool:
            for stmt in node.body:
                if isinstance(stmt, ast.Continue):
                    return True
                # If/Try blocks count: their `continue` short-circuits the
                # whole iteration just the same.
                if isinstance(stmt, (ast.If, ast.Try, ast.With)):
                    if _has_top_level_continue(stmt):
                        return True
                    for handler in getattr(stmt, "orelse", []) or []:
                        if isinstance(handler, ast.Continue):
                            return True
                    for handler in getattr(stmt, "handlers", []) or []:
                        if _has_top_level_continue(handler):
                            return True
                    for handler in getattr(stmt, "finalbody", []) or []:
                        if isinstance(handler, ast.Continue):
                            return True
            return False
        if not _has_top_level_continue(branch):
            branches_missing.append(branch.lineno)
    assert not branches_missing, (
        f"SELL branch(es) at line(s) {branches_missing} call tighten_trail "
        f"but do NOT `continue` before the generic MARKET order block — they "
        f"would place BOTH a trail and a market sell. This is the regression "
        f"vector that hit BNY."
    )


def test_no_market_sell_constant_in_module_top_level():
    """Belt-and-suspenders: the module must not contain a literal
    `OrderRequest(side='SELL', order_type='MARKET')` anywhere. Both
    actual order constructions use `side=direction` (a variable), and
    the only path that reaches them filters out SELL via continue."""
    src = textwrap.dedent(inspect.getsource(sched))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if _is_market_sell_order_request(node):
            raise AssertionError(
                f"Literal SELL MARKET OrderRequest found at line {node.lineno}. "
                f"All SELL orders should go through tighten_trail_on_sell."
            )
