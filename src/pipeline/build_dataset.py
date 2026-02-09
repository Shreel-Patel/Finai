"""
Build final dataset: merge price + technicals + Reddit sentiment + news sentiment.
Writes data/final/{ticker}.csv for the unified predictor.
"""
from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PRICE_DIR = _ROOT / "data" / "raw" / "price"
SENTIMENT_DIR = _ROOT / "data" / "features" / "sentiment"
FINAL_DIR = _ROOT / "data" / "final"


def _add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe. Modifies in place, returns df."""
    if df.empty or len(df) < 2:
        return df
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # EMA
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    df["rsi_14"] = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1 + rs)))

    # Bollinger Bands (20, 2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = np.where(mid == 0, 0, (df["bb_upper"] - df["bb_lower"]) / mid)

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_14_mean_20"] = df["atr_14"].rolling(20).mean()

    # Volume z-score (20)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_z"] = np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0)
    df["volume_z"] = df["volume_z"].fillna(0)

    # ADX 14
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = df["atr_14"]
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / np.where(atr == 0, 1e-10, atr)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / np.where(atr == 0, 1e-10, atr)
    dx = 100 * (plus_di - minus_di).abs() / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di)
    df["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    # Stochastic %K, %D (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / np.where(high14 - low14 == 0, 1e-10, high14 - low14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


def build_dataset(ticker: str) -> None:
    """
    Load price, add technicals, merge Reddit and news sentiment, add target/target_return, save to data/final.
    """
    price_path = RAW_PRICE_DIR / f"{ticker}.csv"
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    df = pd.read_csv(price_path)
    if "Date" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"Not enough rows in {price_path}")

    df = _add_technicals(df)

    # Reddit sentiment: data/features/sentiment/reddit_{ticker}.csv -> date, compound [, weighted_sentiment]
    reddit_path = SENTIMENT_DIR / f"reddit_{ticker}.csv"
    if reddit_path.exists():
        reddit = pd.read_csv(reddit_path)
        date_col = "date" if "date" in reddit.columns else reddit.columns[0]
        reddit[date_col] = pd.to_datetime(reddit[date_col]).dt.normalize()
        if "compound" not in reddit.columns and "sentiment_mean" in reddit.columns:
            reddit["compound"] = reddit["sentiment_mean"]
        reddit = reddit.rename(columns={date_col: "Date"})
        reddit_cols = ["Date", "compound"]
        if "weighted_sentiment" in reddit.columns:
            reddit_cols.append("weighted_sentiment")
        reddit = reddit[[c for c in reddit_cols if c in reddit.columns]]
        df = df.merge(reddit, on="Date", how="left")
        df["compound"] = df["compound"].fillna(0)
        df["sent_3d"] = df["compound"].rolling(3).mean().fillna(0)
        df["sent_7d"] = df["compound"].rolling(7).mean().fillna(0)
        df["sent_14d"] = df["compound"].rolling(14).mean().fillna(0)
        df["sent_delta"] = df["sent_3d"] - df["sent_14d"]
    else:
        df["compound"] = 0
        df["sent_3d"] = df["sent_7d"] = df["sent_14d"] = df["sent_delta"] = 0

    # News sentiment: data/features/sentiment/news_{ticker}_daily.csv -> date, sentiment_mean (or compound)
    news_path = SENTIMENT_DIR / f"news_{ticker}_daily.csv"
    if news_path.exists():
        news = pd.read_csv(news_path)
        date_col = "date" if "date" in news.columns else news.columns[0]
        news[date_col] = pd.to_datetime(news[date_col]).dt.normalize()
        sent_col = "sentiment_mean" if "sentiment_mean" in news.columns else "compound"
        if sent_col not in news.columns:
            sent_col = [c for c in news.columns if "sent" in c.lower() or c == "compound"][:1]
            sent_col = sent_col[0] if sent_col else None
        if sent_col:
            news = news.rename(columns={date_col: "Date", sent_col: "news_sent"})[["Date", "news_sent"]]
            df = df.merge(news, on="Date", how="left")
    if "news_sent" not in df.columns:
        df["news_sent"] = 0
    df["news_sent"] = df["news_sent"].fillna(0)

    # Targets
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["target_return"] = (df["Close"].shift(-1) / df["Close"]) - 1.0
    df = df.dropna(subset=["target"])

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_DIR / f"{ticker}.csv"
    df.to_csv(out_path, index=False)
    print(f"[✓] Built dataset for {ticker}: {len(df)} rows -> {out_path}")
