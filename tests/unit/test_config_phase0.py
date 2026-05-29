"""Phase 0 config scaffolding must be present and default-OFF."""
from app.config import get_settings


def test_phase0_flags_default_off():
    s = get_settings()
    assert s.use_unified_perplexity is False
    assert s.backtest_costs_enabled is False
