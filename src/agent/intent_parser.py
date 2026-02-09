import subprocess
import re
import json

VALID_INTENTS = {"full_analysis", "news_only", "sentiment_only", "technical_only", "movement_only"}

# Words that look like tickers (1–5 letters) but are common query/English words – never use as ticker
TICKER_BLOCKLIST = frozenset({
    "I", "A", "AI", "IT", "USA", "NEWS", "OF", "THE", "FOR", "AND", "OR", "TO", "IN", "ON", "AT", "BY", "AS", "IS",
    "RSI", "EMA", "MACD", "GET", "SEE", "GIVE", "SHOW", "NEED", "WANT", "RUN", "OUT", "ALL", "CAN", "HOW", "WHO",
    "NOW", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "USE", "USD", "API","GOING"
    # Question/auxiliary words so "how will AAPL perform" → AAPL not WILL
    "WILL", "WELL", "WHEN", "WHAT", "WHERE", "WHICH", "WHY", "WITH", "FROM", "THIS", "THAT", "THEM", "THAN", "THEN",
    "TODAY", "TOMORROW", "DOES", "DOING", "DONE", "BEEN", "BEAR", "BULL", "LONG", "SHORT", "BUY", "SELL",
    "SOON", "JUST", "MORE", "MOST", "MUCH", "MANY", "SOME", "ALSO", "ONLY", "VERY", "HERE", "THERE", "OVER", "INTO",
})


def safe_json_extract(text: str):
    """
    Extract the first JSON object found in a string.
    """
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# Known symbols: prefer these over other 1–5 letter words when both appear (e.g. "how will AAPL perform" → AAPL)
KNOWN_TICKER_LIKE = frozenset({
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT", "JNJ", "PG", "MA", "HD",
    "DIS", "NFLX", "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TMO", "ABBV", "LLY", "UNH", "XOM", "CVX",
    "BRK", "BAC", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "AXP", "GS", "MRK", "BMY", "AMGN", "GILD", "REGN",
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "UNI", "ATOM", "LTC", "BCH", "XAU", "XAG",
})

# Patterns: crypto/commodity pair (e.g. BTC-USD, XAU-USD) and forex pair (e.g. EUR-USD, USD-JPY)
TICKER_PAIR_PATTERN = re.compile(r"\b([A-Z]{2,5})-(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|CNY|MXN|INR)\b", re.IGNORECASE)
FOREX_PAIR_PATTERN = re.compile(r"\b([A-Z]{3})[-/]([A-Z]{3})\b", re.IGNORECASE)


def _ticker_from_message(message: str) -> str | None:
    """
    Extract ticker from the user message:
    - First look for explicit pair (e.g. BTC-USD, XAU-USD).
    - Then collect words that look like tickers (1–5 letters), exclude blocklist.
    - Prefer a word that is in KNOWN_TICKER_LIKE (e.g. AAPL over WILL/TODAY); else return the LAST candidate.
    So "how will AAPL perform today" → AAPL (blocklist drops WILL/TODAY; AAPL is only candidate).
    """
    # Explicit pair: crypto/commodity (BTC-USD) or forex (EUR-USD, USD/JPY)
    pair = TICKER_PAIR_PATTERN.search(message)
    if pair:
        return pair.group(0).upper()
    forex = FOREX_PAIR_PATTERN.search(message)
    if forex:
        return f"{forex.group(1).upper()}-{forex.group(2).upper()}"

    words = message.strip().split()
    candidates = []
    for w in words:
        w_clean = w.upper().strip(".,?!")
        if re.match(r"^[A-Z]{1,5}$", w_clean) and w_clean not in TICKER_BLOCKLIST:
            candidates.append(w_clean)

    if not candidates:
        return None
    # Prefer known ticker-like symbol if present
    for c in candidates:
        if c in KNOWN_TICKER_LIKE:
            return c
    return candidates[-1]


def _has_intent_keywords(message: str) -> bool:
    """True if the message explicitly asks for a specific intent (news, sentiment, technical, movement)."""
    msg_lower = message.strip().lower()
    keywords = (
        "news", "headline", "sentiment", "technical", "rsi", "ema", "macd",
        "movement", "price move", "trend",
    )
    return any(k in msg_lower for k in keywords)


def _is_buy_sell_recommendation(message: str) -> bool:
    """True if the user is asking whether to buy/sell (so we always run full_analysis + prediction)."""
    msg_lower = message.strip().lower()
    phrases = (
        "should i buy", "should i sell", "buy or sell", "sell or buy",
        "recommendation", "recommend ", "predict", "prediction",
        "buy/sell", "sell/buy", "should i invest", "worth buying", "worth selling",
    )
    return any(p in msg_lower for p in phrases)


def _fallback_intent(message: str) -> dict:
    """
    When Ollama is unavailable or returns invalid JSON, infer intent and ticker from text.
    - Buy/sell recommendation questions -> full_analysis.
    - Look for intent keywords (news, sentiment, technical, movement).
    - Ticker: from message (last ticker-like word, not in blocklist), else AAPL.
    """
    msg_lower = message.strip().lower()
    intent = "full_analysis"
    if _is_buy_sell_recommendation(message):
        intent = "full_analysis"
    elif "news" in msg_lower or "headline" in msg_lower:
        intent = "news_only"
    elif "sentiment" in msg_lower:
        intent = "sentiment_only"
    elif "technical" in msg_lower or "rsi" in msg_lower or "ema" in msg_lower or "macd" in msg_lower:
        intent = "technical_only"
    elif "movement" in msg_lower or "price move" in msg_lower or "trend" in msg_lower:
        intent = "movement_only"

    ticker = _ticker_from_message(message) or "AAPL"
    return {"intent": intent, "ticker": ticker}


def parse_user_intent(message: str) -> dict:
    """
    Parse user message into intent and ticker. Uses Ollama if available, else keyword fallback.
    Returns dict with keys: intent (str), ticker (str). Intent is one of VALID_INTENTS.
    """
    if not message or not message.strip():
        return {"intent": "full_analysis", "ticker": "AAPL"}

    prompt = f"""
You are an intent parser for a stock market AI.

Extract:
- intent (one of: full_analysis, news_only, sentiment_only, technical_only, movement_only)
- ticker (single stock or crypto ticker, and the ticker should be from yfinance e.g. AAPL, MSFT, BTC-USD)

Return STRICT JSON only. No explanation. No markdown.

User message:
"{message.strip()}"

JSON format:
{{"intent": "...", "ticker": "..."}}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma:2b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=15,
        )
        parsed = safe_json_extract(result.stdout)
        if parsed and isinstance(parsed, dict):
            intent = (parsed.get("intent") or "full_analysis").strip().lower()
            if intent not in VALID_INTENTS:
                intent = "full_analysis"
            # When user asks "should I buy/sell X", always run full analysis and prediction
            if _is_buy_sell_recommendation(message):
                intent = "full_analysis"
            # When user only types a ticker (e.g. "AAPL"), always give full report
            elif not _has_intent_keywords(message):
                intent = "full_analysis"
            ticker = (parsed.get("ticker") or "AAPL").strip().upper()
            if not ticker:
                ticker = "AAPL"
            # Prefer ticker from user message so spelling is preserved (e.g. RIVN not RVN/RINV)
            ticker_from_msg = _ticker_from_message(message)
            if ticker_from_msg:
                ticker = ticker_from_msg
            return {"intent": intent, "ticker": ticker}
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return _fallback_intent(message)
