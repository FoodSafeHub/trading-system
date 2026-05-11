#!/usr/bin/env python
"""Initialize the SQLite database and create all tables."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import init_db
from app.utils.logging import configure_logging

configure_logging()
init_db()
print("Database initialized.")
