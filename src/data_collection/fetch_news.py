import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import re
import hashlib
import xml.etree.ElementTree as ET
import requests

from src.utils.ticker_aliases import get_ticker_aliases

DATA_PATH = Path("C:\\Users\\SHREEL\\PycharmProjects\\FINAI\\data\\news")

# RSS feeds for Bloomberg, Reuters, and FXStreet. Items are filtered by ticker/aliases.
BLOOMBERG_RSS = "https://feeds.bloomberg.com/markets/news.rss"
REUTERS_RSS = "https://www.reutersagency.com/feed/?best-topics=business-finance"
REUTERS_MARKETS_RSS = "https://www.reuters.com/markets/rss/"
FXSTREET_NEWS_RSS = "https://www.fxstreet.com/news/feed"
FXSTREET_ANALYSIS_RSS = "https://www.fxstreet.com/rss/analysis"
FXSTREET_CRYPTO_RSS = "https://www.fxstreet.com/rss/crypto"
FXSTREET_STOCKS_RSS = "https://www.fxstreet.com/rss/stocks"


def _norm_text(s):
    """Strip and collapse whitespace for matching."""
    return (s or "").strip().lower()


def _matches_ticker(text: str, ticker: str) -> bool:
    """True if text contains ticker or any of its aliases (e.g. Bitcoin, BTC for BTC-USD)."""
    if not text:
        return False
    aliases = get_ticker_aliases(ticker)
    t = _norm_text(text)
    base = ticker.upper().split("-")[0] if "-" in ticker else ticker.upper()
    # Require word-boundary style match to avoid false positives
    for a in aliases:
        if not a or len(str(a)) < 2:
            continue
        a_clean = _norm_text(str(a))
        if a_clean in t:
            return True
    if base in t:
        return True
    return False


def _item_text(item, *tag_candidates) -> str:
    """Get text from first matching child (RSS 2.0 or Atom)."""
    atom_ns = "{http://www.w3.org/2005/Atom}"
    for tag in tag_candidates:
        el = item.find(tag)
        if el is None and not tag.startswith("{"):
            el = item.find(atom_ns + tag)
        if el is not None and (el.text or "").strip():
            return (el.text or "").strip()
    return ""


def _item_link(item) -> str:
    """Get link URL from RSS item or Atom entry."""
    el = item.find("link")
    if el is not None and (el.text or "").strip():
        return (el.text or "").strip()
    el = item.find("{http://www.w3.org/2005/Atom}link")
    if el is not None and el.get("href"):
        return (el.get("href") or "").strip()
    return ""


def _fetch_rss_for_ticker(url: str, provider: str, ticker: str, limit: int = 50) -> list:
    """Fetch RSS from url, filter items that mention ticker or aliases, return list of article dicts."""
    try:
        r = requests.get(url, headers={"User-Agent": "FINAI-News/1.0"}, timeout=15)
        r.raise_for_status()
    except Exception:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    # RSS 2.0: channel/item; Atom: feed/entry
    items = list(root.findall(".//item")) or list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
    if not items:
        items = list(root.findall("channel/item")) or list(root.findall("item"))
    articles = []
    for item in items:
        title = _item_text(item, "title")
        desc = _item_text(item, "description", "summary")
        if not _matches_ticker(title + " " + desc, ticker):
            continue
        link_url = _item_link(item)
        pub_date = _item_text(item, "pubDate", "published") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        summary = re.sub(r"<[^>]+>", " ", desc).strip()[:2000] if desc else ""
        uid = link_url or hashlib.md5((title + pub_date).encode()).hexdigest()
        articles.append({
            "id": uid,
            "title": title,
            "summary": summary,
            "pubDate": pub_date,
            "provider": provider,
            "url": link_url,
            "fetched_date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": ticker,
        })
        if len(articles) >= limit:
            break
    return articles[:limit]


def fetch_bloomberg_news(ticker: str, limit: int = 30) -> list:
    """Fetch Bloomberg markets RSS and return articles that mention the ticker or its aliases."""
    return _fetch_rss_for_ticker(BLOOMBERG_RSS, "Bloomberg", ticker, limit=limit)


def fetch_reuters_news(ticker: str, limit: int = 30) -> list:
    """Fetch Reuters business/markets RSS and return articles that mention the ticker or its aliases."""
    out = _fetch_rss_for_ticker(REUTERS_RSS, "Reuters", ticker, limit=limit)
    if len(out) < limit:
        extra = _fetch_rss_for_ticker(REUTERS_MARKETS_RSS, "Reuters", ticker, limit=limit - len(out))
        seen = {a["id"] for a in out}
        for a in extra:
            if a["id"] not in seen:
                out.append(a)
                seen.add(a["id"])
    return out[:limit]


def fetch_fxstreet_news(ticker: str, limit: int = 30) -> list:
    """Fetch FXStreet forex/crypto/stocks RSS and return articles that mention the ticker or its aliases."""
    out = []
    seen = set()
    for url, provider_label in [
        (FXSTREET_NEWS_RSS, "FXStreet"),
        (FXSTREET_ANALYSIS_RSS, "FXStreet"),
        (FXSTREET_CRYPTO_RSS, "FXStreet"),
        (FXSTREET_STOCKS_RSS, "FXStreet"),
    ]:
        if len(out) >= limit:
            break
        batch = _fetch_rss_for_ticker(url, provider_label, ticker, limit=limit - len(out))
        for a in batch:
            if a["id"] not in seen:
                out.append(a)
                seen.add(a["id"])
    return out[:limit]


def fetch_yfinance_news(ticker: str) -> list:
    """Fetch news from yfinance for the ticker. Returns list of article dicts."""
    stock = yf.Ticker(ticker)
    news = getattr(stock, "news", None) or []
    extracted = []
    for article in news:
        content = article.get("content") or article
        title = (content.get("title") or article.get("title") or "").strip()
        summary = (content.get("summary") or "").strip()
        pub = content.get("pubDate") or article.get("published") or ""
        provider = (content.get("provider") or {}).get("displayName", "Yahoo Finance") if isinstance(content.get("provider"), dict) else "Yahoo Finance"
        url = (content.get("canonicalUrl") or {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else article.get("link", "")
        uid = article.get("id") or url or hashlib.md5((title + str(pub)).encode()).hexdigest()
        extracted.append({
            "id": str(uid),
            "title": title,
            "summary": summary,
            "pubDate": str(pub),
            "provider": provider,
            "url": url,
            "fetched_date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": ticker,
        })
    return extracted


def fetch_and_append_news(ticker: str):
    """
    Fetch news from yfinance, Bloomberg, Reuters, and FXStreet; merge and append to the ticker's news file.
    RSS sources (Bloomberg, Reuters, FXStreet) include only items that mention the ticker or its aliases
    (e.g. Bitcoin, btc for BTC-USD; EUR, dollar for EUR-USD).
    """
    ticker = (ticker or "").strip().upper()
    all_articles = []

    # 1) yfinance
    yf_list = fetch_yfinance_news(ticker)
    if yf_list:
        all_articles.extend(yf_list)
        print(f"   Yahoo Finance: {len(yf_list)} articles")
    else:
        print("   Yahoo Finance: no news")

    # 2) Bloomberg RSS (filtered by ticker/aliases)
    try:
        bloomberg_list = fetch_bloomberg_news(ticker)
        if bloomberg_list:
            all_articles.extend(bloomberg_list)
            print(f"   Bloomberg: {len(bloomberg_list)} articles")
    except Exception as e:
        print(f"   Bloomberg: skip ({e})")

    # 3) Reuters RSS (filtered by ticker/aliases)
    try:
        reuters_list = fetch_reuters_news(ticker)
        if reuters_list:
            all_articles.extend(reuters_list)
            print(f"   Reuters: {len(reuters_list)} articles")
    except Exception as e:
        print(f"   Reuters: skip ({e})")

    # 4) FXStreet RSS (forex, crypto, stocks; filtered by ticker/aliases)
    try:
        fxstreet_list = fetch_fxstreet_news(ticker)
        if fxstreet_list:
            all_articles.extend(fxstreet_list)
            print(f"   FXStreet: {len(fxstreet_list)} articles")
    except Exception as e:
        print(f"   FXStreet: skip ({e})")

    if not all_articles:
        print("No new articles from any source.")
        history_file = DATA_PATH / f"{ticker}.csv"
        if history_file.exists():
            return pd.read_csv(history_file)
        return None

    df_new = pd.DataFrame(all_articles)
    df_new["pubDate"] = pd.to_datetime(df_new["pubDate"], errors="coerce")

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    history_file = DATA_PATH / f"{ticker}.csv"

    if history_file.exists():
        df_old = pd.read_csv(history_file)
        df_old["pubDate"] = pd.to_datetime(df_old["pubDate"], errors="coerce")
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
    else:
        df_combined = df_new

    df_combined.to_csv(history_file, index=False)
    print(f"Saved {len(df_new)} new articles, total: {len(df_combined)} articles")
    return df_combined


if __name__ == "__main__":
    df = fetch_and_append_news("BTC-USD")
    if df is not None:
        print("\nLatest articles:")
        print(df[["pubDate", "title", "provider"]].head(10))
