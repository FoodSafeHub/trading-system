#!/usr/bin/env python
"""
Manually trigger one strategy evaluation cycle.
Usage: python scripts/run_cycle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import init_db
from app.services.strategy.scheduler import run_once
from app.utils.logging import configure_logging
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    init_db()
    print("Running one strategy cycle...")
    run_once()
    print("Done.")
