"""Legacy sidebar shim.

Navigation is now built by the ``streamlit_app.py`` entrypoint using
``st.navigation`` (real grouped sections, icons, brand header, utilities).
Every page still calls ``render_sidebar()`` near the top, so this is kept as a
no-op to avoid editing all 14 page files — the entrypoint already renders the
full sidebar before the page script runs.
"""
from __future__ import annotations


def render_sidebar() -> None:
    """No-op. The real sidebar is built in ``streamlit_app.py``."""
    return None
