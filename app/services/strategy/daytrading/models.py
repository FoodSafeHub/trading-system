from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DayTradeSignal:
    symbol: str
    strategy: str
    direction: str          # "BUY" | "SELL" | "HOLD"
    timeframe: str          # "5m" | "15m"
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float       # 0.0–1.0
    reason: str
    regime: str             # "PRE_MARKET" | "BULL_OPEN" | "BEAR_OPEN" | "CHOPPY"
    indicators: dict = field(default_factory=dict)
    r_multiple: float = 0.0
    risk_reward: float = 0.0
    time_in_force: str = "DAY"
    signal_time: str = ""

    def __post_init__(self) -> None:
        if self.entry_price != self.stop_price:
            risk = abs(self.entry_price - self.stop_price)
            reward = abs(self.target_price - self.entry_price)
            self.r_multiple = round(reward / risk, 2) if risk else 0.0
            self.risk_reward = self.r_multiple
