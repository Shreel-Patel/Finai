import asyncio
import pandas as pd
from pathlib import Path
from twscrape import API, gather
from twscrape.logger import set_log_level
import datetime

DATA_PATH = Path("data/raw/twitter")


async def fetch_with_twscrape(ticker="MSFT", limit=500):
    """Fetch tweets using twscrape"""
    api = API()

    # You need to add accounts first (run this once to set up)
    # await api.pool.add_account("username", "password", "email", "email_password")
    # await api.pool.login_all()

    query = f"${ticker} lang:en"
    rows = []

    async for tweet in api.search(query, limit=limit):
        rows.append({
            "ticker": ticker,
            "date": tweet.date,
            "content": tweet.rawContent,
            "likes": tweet.likeCount,
            "retweets": tweet.retweetCount,
            "user": tweet.user.username,
            "hashtags": ", ".join([h for h in tweet.hashtags]) if tweet.hashtags else ""
        })

    if rows:
        df = pd.DataFrame(rows)
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATA_PATH / f"{ticker}_twscrape.parquet")
        print(f"[✓] Saved {len(df)} tweets")
        return df

    return None


# Run the async function
if __name__ == "__main__":
    set_log_level("DEBUG")
    asyncio.run(fetch_with_twscrape(ticker="AAPL", limit=100))