from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List

import pandas as pd

from app.db import SessionLocal
from app.models.signals import Signal
from app.models.strategy_runs import StrategyRun
from app.services.strategy.models import StrategyConfig, StrategySignal
from app.services.strategy.rules import evaluate_strategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Runs a configured strategy against price data, persists signals and run records.
    """

    def run(
        self,
        config: StrategyConfig,
        prices: pd.Series,
    ) -> List[StrategySignal]:
        """
        Evaluate a strategy config against price data.
        Persists a StrategyRun and any resulting Signal records.
        Returns list of StrategySignal objects.
        """
        signals: List[StrategySignal] = []
        run_id: int | None = None

        with SessionLocal() as db:
            run = StrategyRun(
                strategy_name=config.name,
                symbol=config.symbol,
                started_at=datetime.now(tz=timezone.utc),
                status="running",
                parameters_json=json.dumps(config.params),
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            run_id = run.id

        try:
            signal = evaluate_strategy(config.type, config.symbol, prices, config.params)
            signal.strategy_name = config.name
            signals.append(signal)

            with SessionLocal() as db:
                db_signal = Signal(
                    strategy_run_id=run_id,
                    strategy_name=config.name,
                    symbol=config.symbol,
                    direction=signal.direction,
                    strength=signal.strength,
                    price_at_signal=signal.price_at_signal,
                    indicators_json=json.dumps(signal.indicators),
                    created_at=datetime.now(tz=timezone.utc),
                )
                db.add(db_signal)

                run_obj = db.query(StrategyRun).filter_by(id=run_id).first()
                if run_obj:
                    run_obj.status = "completed"
                    run_obj.completed_at = datetime.now(tz=timezone.utc)
                    run_obj.signals_generated = 1
                db.commit()

            logger.info(
                f"Strategy [{config.name}] {config.symbol} → {signal.direction} "
                f"(price={signal.price_at_signal or 0:.4f})"
            )

        except Exception as exc:
            logger.exception("Strategy [%s] failed: %s", config.name, exc)
            with SessionLocal() as db:
                run_obj = db.query(StrategyRun).filter_by(id=run_id).first()
                if run_obj:
                    run_obj.status = "error"
                    run_obj.completed_at = datetime.now(tz=timezone.utc)
                    run_obj.error_message = str(exc)
                db.commit()

        return signals


def load_strategies_from_config(config_path: str = "strategies.json") -> List[StrategyConfig]:
    """
    Load strategy configs from a JSON file.
    If the file doesn't exist, return the built-in example strategy.
    """
    import os

    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = json.load(f)
        return [StrategyConfig(**s) for s in raw]

    # Built-in example strategy (SMA/RSI on SPY)
    return [
        StrategyConfig(
            name="SPY_SMA_RSI",
            symbol="SPY",
            type="sma_rsi",
            params={
                "sma_fast": 10,
                "sma_slow": 30,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
            },
            enabled=True,
        )
    ]
