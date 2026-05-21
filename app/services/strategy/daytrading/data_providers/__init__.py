"""Market-data providers for the day-trading pipeline.

Provider routing for intraday bars (1m / 5m / 15m / 1d):

    Twelve Data (primary)  →  Webull (first fallback)  →  yfinance (second fallback)

Each provider returns an empty DataFrame on any failure (auth, rate-limit,
network, parse). Callers should treat empty == "try the next provider".

The chosen provider is attached to the returned DataFrame as
``df.attrs["source"]`` (one of: "twelvedata" | "webull" | "yfinance") and
logged at INFO level so a tail of trading.log makes it obvious which
provider actually served each request.
"""
