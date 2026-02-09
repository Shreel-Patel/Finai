import os
import re

import yfinance as yf
import pandas as pd
from pathlib import Path

# Project root (works on Render and locally)
_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PRICE_DIR = _ROOT / "data" / "raw" / "price"

# yfinance uses different symbols for some assets; we still save as {ticker}.csv for the pipeline
YF_SYMBOL_ALIASES = {
    "XAU-USD": "GC=F",   # Gold: use gold futures
    "XAG-USD": "SI=F",   # Silver: use silver futures
    "DXY": "DX-Y.NYB",   # US Dollar Index (ICE)
    "DOLLARINDEX": "DX-Y.NYB",
    "DOLLAR-INDEX": "DX-Y.NYB",
}

# Yahoo Finance forex: BASEQUOTE=X (e.g. EURUSD=X = EUR/USD). User may pass EUR-USD, EUR/USD, EURUSD.
# Major pairs and common alternates (e.g. EUR=X is USD/EUR; we prefer EURUSD=X for EUR/USD).
FOREX_YF_SYMBOLS = {
    "EUR-USD": "EURUSD=X", "EURUSD": "EURUSD=X", "EUR/USD": "EURUSD=X",
    "GBP-USD": "GBPUSD=X", "GBPUSD": "GBPUSD=X", "GBP/USD": "GBPUSD=X",
    "USD-JPY": "USDJPY=X", "USDJPY": "USDJPY=X", "USD/JPY": "USDJPY=X",
    "AUD-USD": "AUDUSD=X", "AUDUSD": "AUDUSD=X", "AUD/USD": "AUDUSD=X",
    "USD-CHF": "USDCHF=X", "USDCHF": "USDCHF=X", "USD/CHF": "USDCHF=X",
    "USD-CAD": "USDCAD=X", "USDCAD": "USDCAD=X", "USD/CAD": "USDCAD=X",
    "NZD-USD": "NZDUSD=X", "NZDUSD": "NZDUSD=X", "NZD/USD": "NZDUSD=X",
    "EUR-JPY": "EURJPY=X", "EURJPY": "EURJPY=X", "EUR/JPY": "EURJPY=X",
    "GBP-JPY": "GBPJPY=X", "GBPJPY": "GBPJPY=X", "GBP/JPY": "GBPJPY=X",
    "EUR-GBP": "EURGBP=X", "EURGBP": "EURGBP=X", "EUR/GBP": "EURGBP=X",
    "USD-CNY": "CNY=X",  "USDCNY": "CNY=X", "USD/CNY": "CNY=X",
    "USD-MXN": "MXN=X",  "USDMXN": "MXN=X", "USD/MXN": "MXN=X",
    "USD-INR": "INR=X",  "USDINR": "INR=X", "USD/INR": "INR=X",
}

def _normalize_forex_ticker(ticker: str) -> str | None:
    """Return Yahoo forex symbol if ticker looks like a forex pair, else None."""
    t = ticker.upper().strip()
    if t in FOREX_YF_SYMBOLS:
        return FOREX_YF_SYMBOLS[t]
    # Generic: BASE-QUOTE or BASEQUOTE -> BASEQUOTE=X (only for known 3-letter codes)
    m = re.match(r"^([A-Z]{3})[-/]?([A-Z]{3})$", t.replace(" ", ""))
    if m:
        base, quote = m.group(1), m.group(2)
        generic = f"{base}{quote}=X"
        return generic
    return None


def _yf_symbol_for_ticker(ticker: str) -> str:
    """Resolve ticker to the symbol yfinance expects. Save file still as original ticker."""
    t = ticker.upper().strip()
    if t in YF_SYMBOL_ALIASES:
        return YF_SYMBOL_ALIASES[t]
    forex = _normalize_forex_ticker(ticker)
    if forex:
        return forex
    return ticker


def fetch_price(ticker="MSFT", start="2025-05-01"):
    # Use Yahoo symbol for download; save under original ticker so pipeline finds it
    yf_symbol = _yf_symbol_for_ticker(ticker)
    try:
        df = yf.download(yf_symbol, start=start, progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        print(f"[!] Download failed for {ticker} (yf symbol: {yf_symbol}): {e}")
        return

    if df is None or df.empty or len(df) < 2:
        print(f"[!] No or insufficient price data for {ticker} (yf symbol: {yf_symbol})")
        return

    df = df.copy()
    df.reset_index(inplace=True)

    # Flatten MultiIndex columns so CSV has one header row
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).strip() for c in df.columns.get_level_values(0)]

    # Date column
    if "Date" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "Date"})

    # Forex (and some indices) often have no Volume; require OHLC
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = float("nan")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    # Coerce numeric (yfinance sometimes returns object)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Close is missing so technicals don't break
    df = df.dropna(subset=["Close"], how="all")
    if len(df) < 2:
        print(f"[!] Not enough rows with valid Close for {ticker}")
        return

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    filename = f"{ticker}.csv"
    out_path = RAW_PRICE_DIR / filename
    RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[✓] Saved price data for {ticker}, rows={len(df)}")


if __name__ == "__main__":
    fetch_price()
