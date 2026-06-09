"""
Symbol Deployment Policy — per-symbol live-trading gates.

States
------
enabled          : cleared for live auto-trading with no extra conditions
monitor_only     : runs in scans and backtests; auto-trader is blocked
regime_dependent : live trading allowed only when regime condition is met
disabled         : excluded from all live and scan flows

Each policy record carries:
  - status        : one of the four states above
  - reason        : one-line human explanation shown in UI
  - wf_verdict    : latest walk-forward label ("robust" / "marginal" / "unprofitable" / …)
  - wf_score      : 0–100 consistency score from the walk-forward report
  - override_live : if True, auto-trader ignores the policy (manual override)
  - allowed_regimes: for regime_dependent only — list of regime strings where live is ok

The policy dict is the single source of truth.  Change entries here to change
platform behavior across runner, auto-trader, and UI with no strategy edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Policy states ──────────────────────────────────────────────────────────────
ENABLED          = "enabled"
MONITOR_ONLY     = "monitor_only"
REGIME_DEPENDENT = "regime_dependent"
DISABLED         = "disabled"

_VALID_STATES = {ENABLED, MONITOR_ONLY, REGIME_DEPENDENT, DISABLED}


@dataclass
class SymbolPolicy:
    symbol:          str
    status:          str           # ENABLED | MONITOR_ONLY | REGIME_DEPENDENT | DISABLED
    reason:          str           # shown in UI
    wf_verdict:      str = ""      # e.g. "robust", "marginal", "unprofitable"
    wf_score:        float = 0.0   # 0–100 consistency score
    override_live:   bool = False  # manual override: bypasses status check for live
    allowed_regimes: list[str] = field(default_factory=list)  # regime_dependent only

    def allows_live(self, regime: str | None = None) -> bool:
        """Return True if live auto-trading is permitted right now."""
        if self.override_live:
            return True
        if self.status == ENABLED:
            return True
        if self.status == REGIME_DEPENDENT:
            return regime in self.allowed_regimes if self.allowed_regimes else False
        return False   # MONITOR_ONLY and DISABLED

    def allows_scan(self) -> bool:
        """Return True if the symbol should appear in scans and compare-all."""
        return self.status != DISABLED

    def badge(self) -> str:
        """Short badge string for UI display."""
        _badges = {
            ENABLED:          "✅ LIVE",
            MONITOR_ONLY:     "👁 MONITOR",
            REGIME_DEPENDENT: "🔀 REGIME",
            DISABLED:         "🚫 DISABLED",
        }
        return _badges.get(self.status, self.status.upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "status":          self.status,
            "reason":          self.reason,
            "wf_verdict":      self.wf_verdict,
            "wf_score":        self.wf_score,
            "override_live":   self.override_live,
            "allowed_regimes": self.allowed_regimes,
            "badge":           self.badge(),
            "allows_scan":     self.allows_scan(),
        }


# ── Master policy table ────────────────────────────────────────────────────────
# Edit entries here to change deployment behavior.
# Keys are uppercase ticker strings.
_POLICY: dict[str, SymbolPolicy] = {
    "AAPL": SymbolPolicy(
        symbol="AAPL",
        status=ENABLED,
        reason="Walk-forward validated: OOS PF 3.60 across all 3 windows, WFE 2.21, 75% win rate.",
        wf_verdict="robust",
        wf_score=100.0,
    ),
    "NVDA": SymbolPolicy(
        symbol="NVDA",
        status=ENABLED,
        reason=(
            "High-volatility name. Signals enabled for manual review; "
            "IS window marginal (PF 0.15 in April sell-off). "
            "Monitor position sizing carefully."
        ),
        wf_verdict="marginal",
        wf_score=69.0,
    ),
    "SPY": SymbolPolicy(
        symbol="SPY",
        status=ENABLED,
        reason=(
            "Profitable in trending windows. Signals enabled in all regimes "
            "for manual review; auto-trader most effective in BULL_OPEN/BEAR_OPEN."
        ),
        wf_verdict="marginal",
        wf_score=73.4,
        allowed_regimes=["BULL_OPEN", "BEAR_OPEN"],
    ),
    "TSLA": SymbolPolicy(
        symbol="TSLA",
        status=ENABLED,
        reason=(
            "High volatility — signals enabled for manual review. "
            "Walk-forward showed insufficient OOS trades; use signals as research, "
            "not auto-trade without further validation."
        ),
        wf_verdict="insufficient_data",
        wf_score=0.0,
    ),
    # All other symbols: ENABLED by default so any ticker generates signals.
    # The policy table controls auto-trader deployment, not signal visibility.
}

_DEFAULT_POLICY = SymbolPolicy(
    symbol="UNKNOWN",
    status=ENABLED,
    reason=(
        "No walk-forward validation on record. Signals enabled for manual review. "
        "Validate via backtest before enabling auto-trader."
    ),
    wf_verdict="unvalidated",
    wf_score=0.0,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_policy(symbol: str) -> SymbolPolicy:
    """Return the deployment policy for a symbol (uppercase). Falls back to default."""
    return _POLICY.get(symbol.upper(), _DEFAULT_POLICY)


def set_policy(symbol: str, policy: SymbolPolicy) -> None:
    """Update the in-memory policy for a symbol at runtime (persists until restart)."""
    _POLICY[symbol.upper()] = policy


def set_override(symbol: str, override: bool) -> None:
    """Toggle the manual live override for a symbol without changing other fields."""
    p = get_policy(symbol)
    p.override_live = override
    _POLICY[symbol.upper()] = p


def allows_live(symbol: str, regime: str | None = None) -> tuple[bool, str]:
    """
    Primary gate function.  Returns (allowed: bool, reason: str).
    reason is human-readable — pass it to the UI or log.
    """
    p = get_policy(symbol)
    ok = p.allows_live(regime)
    if ok:
        return True, f"{symbol} policy={p.status} — live trading permitted"
    if p.override_live:
        return True, f"{symbol} manual override active"
    if p.status == DISABLED:
        return False, f"{symbol} DISABLED: {p.reason}"
    if p.status == MONITOR_ONLY:
        return False, f"{symbol} MONITOR_ONLY: {p.reason}"
    if p.status == REGIME_DEPENDENT:
        return False, (
            f"{symbol} REGIME_DEPENDENT: live allowed only in "
            f"{p.allowed_regimes} — current regime={regime}"
        )
    return False, f"{symbol} policy={p.status}: not permitted"


def allows_scan(symbol: str) -> tuple[bool, str]:
    """Gate for scan / compare-all flows."""
    p = get_policy(symbol)
    ok = p.allows_scan()
    if ok:
        return True, f"{symbol} included in scan (status={p.status})"
    return False, f"{symbol} excluded from scan (DISABLED)"


def all_policies() -> list[dict[str, Any]]:
    """Return all defined policies as a list of dicts (for UI tables)."""
    return [p.to_dict() for p in sorted(_POLICY.values(), key=lambda x: x.symbol)]
