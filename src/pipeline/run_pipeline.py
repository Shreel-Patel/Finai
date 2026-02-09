from datetime import datetime, timedelta
from pathlib import Path

from src.data_collection.fetch_price import fetch_price
from src.data_collection.fetch_reddit import fetch_reddit_for_ticker
from src.data_collection.fetch_news import fetch_and_append_news
from src.sentiment.aggregate import aggregate_multiple_sources
from src.sentiment.vader_social import score_multiple_sources
from src.sentiment.finbert_news import score_news

from data.features.build_dataset import build_dataset

# Project root (works on Render and locally)
_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FINAL = _ROOT / "data" / "final"

# Daily data: last 6 months
DAILY_START_DAYS = 183


def ensure_data(ticker: str):
    """
    End-to-end pipeline for a single ticker.
    Uses daily price for last 6 months only. Direction only (UP/DOWN/HOLD).
    Safe to call multiple times (idempotent).
    """
    print(f"[PIPELINE] Running full pipeline for {ticker} (daily, last 6 months)")

    # 1️⃣ PRICE: daily, last 6 months
    start = (datetime.now() - timedelta(days=DAILY_START_DAYS)).strftime("%Y-%m-%d")
    fetch_price(ticker=ticker, start=start)

    # 2️⃣ RAW TEXT DATA
    fetch_reddit_for_ticker(ticker=ticker)
    fetch_and_append_news(ticker=ticker)

    # 3️⃣ SENTIMENT SCORING
    score_news(ticker=ticker)
    score_multiple_sources(ticker=ticker)

    # 4️⃣ AGGREGATE DAILY SENTIMENT
    aggregate_multiple_sources(ticker=ticker)

    # 5️⃣ FINAL DATASET
    build_dataset(ticker=ticker)
