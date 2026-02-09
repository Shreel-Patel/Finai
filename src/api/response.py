import pandas as pd
from pathlib import Path
from typing import Optional

# Project root (works on Render and locally)
_ROOT = Path(__file__).resolve().parent.parent.parent
NEWS_PATH = _ROOT / "data" / "processed" / "news"
REDDIT_PATH = _ROOT / "data" / "reddit"


def _sentiment_label(pos: float, neg: float) -> str:
    net = pos - neg
    if net > 0.2:
        return "positive"
    if net < -0.2:
        return "negative"
    return "neutral"


def get_news_response(ticker: str, limit: int = 15):
    """Return recent news with FinBERT sentiment (pos, neg, neu, label) for the ticker."""
    path = NEWS_PATH / f"{ticker}.csv"
    if not path.exists():
        return {"type": "news", "ticker": ticker, "articles": [], "summary": "No news data available for this ticker."}

    df = pd.read_csv(path)
    if "pubDate" not in df.columns:
        return {"type": "news", "ticker": ticker, "articles": [], "summary": "News data has no dates."}

    df = df.sort_values("pubDate", ascending=False).head(limit)
    cols = ["title", "summary", "provider", "url", "pubDate"]
    if "pos" in df.columns and "neg" in df.columns:
        df = df.assign(
            sentiment_net=df["pos"] - df["neg"],
            sentiment_label=df.apply(lambda r: _sentiment_label(r["pos"], r["neg"]), axis=1)
        )
        cols = ["title", "summary", "provider", "url", "pubDate", "pos", "neg", "neu", "sentiment_label"]
    latest = df[cols].copy()
    latest["pubDate"] = latest["pubDate"].astype(str)
    articles = latest.to_dict(orient="records")

    # Brief summary: average sentiment over recent articles
    if "pos" in df.columns and "neg" in df.columns:
        avg_net = (df["pos"] - df["neg"]).mean()
        summary = f"Recent news sentiment for {ticker}: {'positive' if avg_net > 0.1 else 'negative' if avg_net < -0.1 else 'neutral'} (avg score {round(avg_net, 3)})."
    else:
        summary = f"Showing {len(articles)} recent headlines for {ticker}."

    return {
        "type": "news",
        "ticker": ticker,
        "articles": articles,
        "summary": summary,
    }


def _sentiment_weights(article_count: int, post_count: int) -> tuple[float, float]:
    """Weights based on how many articles vs posts we have. The source with more items gets higher weight."""
    total = article_count + post_count
    if total <= 0:
        return 0.5, 0.5
    return article_count / total, post_count / total


def get_sentiment_response(ticker: str):
    """
    Combined sentiment with weights by volume: whichever source has more items (articles vs posts)
    gets higher weight. avg_sentiment = weight_news * news_avg + weight_reddit * reddit_avg.
    """
    news_path = NEWS_PATH / f"{ticker}.csv"
    reddit_path = REDDIT_PATH / f"{ticker}.csv"

    news_avg: Optional[float] = None
    reddit_avg: Optional[float] = None
    article_count = 0
    post_count = 0

    if news_path.exists():
        df_news = pd.read_csv(news_path)
        if "pos" in df_news.columns and "neg" in df_news.columns:
            news_avg = float((df_news["pos"] - df_news["neg"]).mean())
            article_count = len(df_news)

    if reddit_path.exists():
        reddit = pd.read_csv(reddit_path)
        if "sentiment" in reddit.columns:
            reddit_avg = float(reddit["sentiment"].mean())
            post_count = len(reddit)

    # No data at all
    if news_avg is None and reddit_avg is None:
        return {
            "type": "sentiment",
            "ticker": ticker,
            "avg_sentiment": None,
            "post_count": 0,
            "message": "No news or Reddit data for this ticker.",
        }

    weight_news, weight_reddit = _sentiment_weights(article_count, post_count)

    # Both sources
    if news_avg is not None and reddit_avg is not None:
        avg_sentiment = round(weight_news * news_avg + weight_reddit * reddit_avg, 3)
        return {
            "type": "sentiment",
            "ticker": ticker,
            "avg_sentiment": avg_sentiment,
            "news_sentiment": round(news_avg, 3),
            "reddit_sentiment": round(reddit_avg, 3),
            "weights": {"news": round(weight_news, 3), "reddit": round(weight_reddit, 3)},
            "article_count": article_count,
            "post_count": post_count,
        }
    if news_avg is not None:
        return {
            "type": "sentiment",
            "ticker": ticker,
            "avg_sentiment": round(news_avg, 3),
            "news_sentiment": round(news_avg, 3),
            "reddit_sentiment": None,
            "weights": {"news": 1.0, "reddit": 0.0},
            "article_count": article_count,
            "post_count": 0,
        }
    # Reddit only
    return {
        "type": "sentiment",
        "ticker": ticker,
        "avg_sentiment": round(reddit_avg, 3),
        "news_sentiment": None,
        "reddit_sentiment": round(reddit_avg, 3),
        "weights": {"news": 0.0, "reddit": 1.0},
        "article_count": 0,
        "post_count": post_count,
    }


def _safe_float(row, key: str, default=None):
    try:
        if key not in row or pd.isna(row.get(key)):
            return default
        return float(row[key])
    except (TypeError, ValueError):
        return default


def build_technical_snapshot(row) -> dict:
    """Build a technical analysis snapshot from a single row (latest bar). Used for API and full_analysis."""
    close_col = "Close" if "Close" in row else "close"
    price = _safe_float(row, close_col, 0)
    rsi = _safe_float(row, "rsi_14")
    ema_20 = _safe_float(row, "ema_20")
    ema_50 = _safe_float(row, "ema_50")
    atr = _safe_float(row, "atr_14")
    atr_mean = _safe_float(row, "atr_14_mean_20")
    trend = "bullish" if (ema_20 is not None and ema_50 is not None and ema_20 > ema_50) else "bearish"
    volatility = "high" if (atr is not None and atr_mean is not None and atr > atr_mean) else "normal"

    macd = _safe_float(row, "macd")
    macd_sig = _safe_float(row, "macd_signal")
    macd_signal = "bullish" if (macd is not None and macd_sig is not None and macd > macd_sig) else "bearish"

    adx = _safe_float(row, "adx_14")
    trend_strength = "strong" if adx is not None and adx > 25 else ("weak" if adx is not None and adx < 20 else "moderate")

    bb_upper = _safe_float(row, "bb_upper", price * 1.1 if price else None)
    bb_lower = _safe_float(row, "bb_lower", price * 0.9 if price else None)
    if price and bb_upper is not None and bb_lower is not None:
        if price >= bb_upper * 0.99:
            bb_position = "upper_band"
        elif price <= bb_lower * 1.01:
            bb_position = "lower_band"
        else:
            bb_position = "middle"
    else:
        bb_position = None

    stoch_k = _safe_float(row, "stoch_k")
    stoch_d = _safe_float(row, "stoch_d")

    return {
        "price": price,
        "rsi": rsi,
        "trend": trend,
        "volatility": volatility,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "macd_signal": macd_signal,
        "adx": adx,
        "trend_strength": trend_strength,
        "bb_position": bb_position,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }


def get_technical_response(df: pd.DataFrame):
    """Return technical analysis for the latest row. Rich snapshot for UI."""
    latest = df.iloc[-1]
    out = build_technical_snapshot(latest)
    out["type"] = "technical"
    return out


def get_movement_response(expected_move: float, low: float, high: float):
    return {
        "type": "movement",
        "expected_move_pct": round(expected_move * 100, 2),
        "expected_range": {"low": low, "high": high},
    }
