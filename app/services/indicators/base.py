from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


@dataclass
class IndicatorResult:
    name: str
    values: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latest(self) -> float | None:
        """Most recent non-NaN value."""
        valid = self.values.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None
