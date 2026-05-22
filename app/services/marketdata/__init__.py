"""India market-data providers.

Upstox is the India quote + historical-bar source while orders execute on
Zerodha (free Upstox data API avoids Zerodha's paid data add-on). Each public
function returns empty / None on any failure so callers can fall back to the
existing provider chain (Twelve Data → Webull → yfinance) or to no-quote mode.
"""
