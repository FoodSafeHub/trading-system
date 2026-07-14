"""Phase 1: Perplexity adapter is OFF by default; ON swaps to rule-backed."""
import app.config as cfg
from app.services.strategy.perplexity import runner
from app.services.strategy.perplexity import adapter as adapter_mod
from app.services.strategy.models import StrategySignal


def test_default_off_returns_bespoke_list():
    # No flag flip -> the exact bespoke module list (byte-identical behaviour).
    # The list grew from 12 with the India swing/advanced additions — assert
    # identity and a floor rather than pinning the exact count, so adding a
    # bespoke strategy doesn't break an unrelated adapter test.
    assert runner.get_perplexity_strategies() is runner.PERPLEXITY_STRATEGIES
    assert len(runner.PERPLEXITY_STRATEGIES) >= 12


def test_flag_on_returns_rule_backed_adapters(monkeypatch):
    class S:
        use_unified_perplexity = True
    monkeypatch.setattr(cfg, "get_settings", lambda: S())
    str000 = runner.get_perplexity_strategies()
    assert len(str000) == 6
    assert all(s.__class__.__name__ == "RuleBackedPerplexityStrategy" for s in str000)
    assert [s.name for s in str000][0] == "Unified_RSI2_Reversion"


def test_adapter_maps_strategysignal_to_perplexitysignal(monkeypatch):
    import pandas as pd
    adp = adapter_mod.RuleBackedPerplexityStrategy("trend_pullback", "Unified_Trend_Pullback")
    fake = StrategySignal(symbol="X", direction="BUY", strength=0.5, price_at_signal=50.0,
                          indicators={"k": 1}, strategy_name="trend_pullback",
                          stop_price=48.0, target_price=55.0, confidence=0.7)
    monkeypatch.setattr(adapter_mod, "evaluate_strategy", lambda *a, **k: fake)
    df = pd.DataFrame({"Close": [50.0, 50.0]})
    sig = adp.run("X", df)
    assert sig.direction == "BUY"
    assert sig.entry_price == 50.0 and sig.stop_price == 48.0 and sig.target_price == 55.0
    assert sig.confidence == 0.7 and sig.strategy_name == "Unified_Trend_Pullback"


def test_adapter_hold_when_rule_holds(monkeypatch):
    import pandas as pd
    adp = adapter_mod.RuleBackedPerplexityStrategy("rsi2_reversion", "Unified_RSI2_Reversion")
    fake = StrategySignal(symbol="X", direction="HOLD", strategy_name="rsi2_reversion")
    monkeypatch.setattr(adapter_mod, "evaluate_strategy", lambda *a, **k: fake)
    sig = adp.run("X", pd.DataFrame({"Close": [10.0]}))
    assert sig.direction == "HOLD"
