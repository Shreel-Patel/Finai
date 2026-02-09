import os
import re
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
from pydantic import BaseModel

from src.agent.intent_parser import parse_user_intent
from src.agent.llm_explainer import generate_explanation

from src.pipeline.run_pipeline import ensure_data

from src.models.config import get_features_for_df
from src.models.predictor import train_unified_predictor

from src.agent.agent import run_agent
from src.agent.chat_agent import run_chat_agent
from src.api.response import get_news_response, get_sentiment_response, get_technical_response, get_movement_response, build_technical_snapshot, _safe_float

app = FastAPI(title="AI Market Agent")
from fastapi.middleware.cors import CORSMiddleware

_cors_origins = os.environ.get("CORS_ORIGINS", "*")
if _cors_origins.strip() != "*":
    _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
else:
    _cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project root: src/api/main.py -> FINAI
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final"
DATA_PATH = str(_DATA_DIR)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


def _build_reply_single(result: dict) -> str:
    """Build a short natural-language reply from a single analysis result."""
    ticker = result.get("ticker", "")
    rec = result.get("recommendation", "")
    analysis = result.get("analysis", "")
    if isinstance(analysis, dict):
        parts = (analysis.get("market_summary") or []) + (analysis.get("ai_reasoning") or [])
        analysis = " ".join(parts) if parts else ""
    if rec:
        reply = rec
        if analysis and isinstance(analysis, str) and analysis.strip():
            reply += " " + analysis.strip()
        return reply
    return f"Analysis for {ticker}: " + (str(analysis)[:300] if analysis else "No summary available.")


def _build_reply_compare(left: dict, right: dict) -> str:
    """Build a short reply comparing two analyses."""
    a, b = left.get("ticker", ""), right.get("ticker", "")
    ra, rb = left.get("recommendation", ""), right.get("recommendation", "")
    pa, pb = left.get("prob_up"), right.get("prob_up")
    parts = [f"**{a}**: {ra}"]
    if pa is not None:
        parts[-1] += f" (P(up) {round(pa * 100, 1)}%)"
    parts.append(f"**{b}**: {rb}")
    if pb is not None:
        parts[-1] += f" (P(up) {round(pb * 100, 1)}%)"
    return " ".join(parts)


def _parse_compare_message(message: str) -> tuple[str | None, str | None]:
    """If message asks to compare two tickers, return (ticker_a, ticker_b); else (None, None)."""
    msg = message.strip()
    # "compare AAPL and MSFT", "compare AAPL vs MSFT", "AAPL vs MSFT", "AAPL and MSFT"
    m = re.search(r"compare\s+([A-Za-z0-9\-\.]+)\s+(?:and|vs\.?)\s+([A-Za-z0-9\-\.]+)", msg, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip().upper()
    m = re.search(r"([A-Za-z0-9\-\.]+)\s+vs\.?\s+([A-Za-z0-9\-\.]+)", msg, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip().upper()
    m = re.search(r"([A-Za-z0-9\-\.]+)\s+and\s+([A-Za-z0-9\-\.]+)", msg, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip().upper()
    return None, None


def _chart_data(ticker: str, days: int = 180) -> dict:
    """Return time series (dates, closes) for chart. Used by route and by agent tools."""
    ticker = ticker.upper().strip()
    csv_path = _DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        return {"error": f"No data for {ticker}."}
    df = pd.read_csv(csv_path).dropna()
    date_col = "date" if "date" in df.columns else "Date"
    if date_col not in df.columns:
        date_col = df.columns[0]
    df = df.tail(min(365, max(7, days)))
    dates = df[date_col].astype(str).tolist()
    closes = df["Close"].astype(float).tolist()
    return {"ticker": ticker, "dates": dates, "closes": closes}


def _list_tickers() -> dict:
    """Return list of supported tickers. Used by route and by agent tools."""
    if not _DATA_DIR.exists():
        return {"tickers": []}
    tickers = [f.stem for f in _DATA_DIR.glob("*.csv")]
    return {"tickers": sorted(tickers)}


@app.get("/tickers")
def list_tickers():
    """Return list of supported tickers (from data/final)."""
    return _list_tickers()


@app.get("/chart/{ticker}")
def chart_data(ticker: str, days: int = Query(180, ge=7, le=365)):
    """Return time series (dates, closes) for price chart. Last N days from final dataset."""
    out = _chart_data(ticker, days)
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/query")
def query(q: str = Query(..., min_length=1)):
    """
    Natural-language query: parses intent and ticker, returns intent-specific data.
    Examples: "AAPL", "news sentiment of AAPL", "technical analysis MSFT", "sentiment for TSLA"
    """
    parsed = parse_user_intent(q)
    intent = parsed.get("intent", "full_analysis")
    ticker = (parsed.get("ticker") or "AAPL").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Could not extract a ticker from your query. Try e.g. 'AAPL' or 'news sentiment of MSFT'.")

    if intent == "news_only":
        ensure_data(ticker)
        news_data = get_news_response(ticker)
        return {"intent": "news_only", "ticker": ticker, **news_data}

    if intent == "sentiment_only":
        ensure_data(ticker)
        sentiment_data = get_sentiment_response(ticker)
        return {"intent": "sentiment_only", "ticker": ticker, **sentiment_data}

    if intent == "technical_only":
        ensure_data(ticker)
        df = _load_sorted_final_df(ticker)
        tech_data = get_technical_response(df)
        return {"intent": "technical_only", "ticker": ticker, **tech_data}

    if intent == "movement_only":
        ensure_data(ticker)
        df = _load_sorted_final_df(ticker)
        if len(df) < 2:
            return {"intent": "movement_only", "ticker": ticker, "message": "Not enough data for movement."}
        latest = df.iloc[-1]
        atr = _safe_float(latest, "atr_14", 0) or 0
        close = _safe_float(latest, "Close", 0) or _safe_float(latest, "close", 0) or 0
        move_data = get_movement_response(atr / close if close else 0, close - atr, close + atr)
        return {"intent": "movement_only", "ticker": ticker, **move_data}

    # full_analysis: same as /analyze
    return _run_full_analysis(ticker)


def _load_sorted_final_df(ticker: str) -> pd.DataFrame:
    """Load final CSV for ticker, sort by Date so last row is the latest bar. Drops invalid dates."""
    csv_path = _DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"No data for {ticker}.")
    df = pd.read_csv(csv_path).dropna()
    sort_col = "Date" if "Date" in df.columns else "date"
    if sort_col in df.columns:
        sort_key = pd.to_datetime(df[sort_col], errors="coerce")
        df = df.loc[sort_key.notna()].copy()
        df = df.assign(_sort_ts=sort_key.loc[df.index])
        df = df.sort_values("_sort_ts").drop(columns=["_sort_ts"]).reset_index(drop=True)
    return df


def _format_date(row) -> str:
    """Format date from analysis row for API response (Date or date column)."""
    for col in ("Date", "date"):
        if col in row and pd.notna(row.get(col)):
            v = row[col]
            if hasattr(v, "strftime"):
                return v.strftime("%Y-%m-%d")
            return str(v)
    return ""


def _run_full_analysis(ticker: str):
    ensure_data(ticker)
    df = _load_sorted_final_df(ticker)
    if len(df) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough rows for {ticker} to run prediction. Need at least 2 rows after dropna.",
        )
    features = get_features_for_df(df)
    if not features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature columns for {ticker}. Rebuild the dataset with technicals + sentiment.",
        )
    # Unified predictor: direction only (UP/DOWN/HOLD), no magnitude
    predictor = train_unified_predictor(df[:-1], features=features)
    latest = df.iloc[-1]
    out = predictor.predict(latest)
    prob = out["prob_up"]
    if prob is None or (isinstance(prob, float) and (prob != prob or prob < 0 or prob > 1)):
        prob = 0.5
    prob = float(prob)
    row = latest.copy()
    row["prob_up"] = float(prob)
    decision = run_agent(row)
    close = _safe_float(row, "Close", 0) or _safe_float(row, "close", 0)
    ema_20 = _safe_float(row, "ema_20")
    ema_50 = _safe_float(row, "ema_50")
    atr = _safe_float(row, "atr_14")
    atr_mean = _safe_float(row, "atr_14_mean_20")
    market_snapshot = {
        "price": close,
        "rsi": _safe_float(row, "rsi_14"),
        "trend": "bullish" if (ema_20 is not None and ema_50 is not None and ema_20 > ema_50) else "bearish",
        "sentiment_delta": _safe_float(row, "sent_delta") or 0,
        "volatility": "high" if (atr is not None and atr_mean is not None and atr > atr_mean) else "normal",
    }
    explanation = generate_explanation(decision, market_snapshot)
    # Clear one-line recommendation so "Should I buy/sell RIVN" gets an obvious answer
    signal = out["signal"]
    if signal == "buy":
        recommendation = f"Consider buying {ticker}. Probability of price going up: {round(prob * 100, 1)}%."
    elif signal == "sell":
        recommendation = f"Consider selling or avoiding {ticker}. Probability of price going up: {round(prob * 100, 1)}%."
    else:
        recommendation = f"Hold {ticker} — no strong buy/sell signal. Probability of price going up: {round(prob * 100, 1)}%."
    result = {
        "intent": "full_analysis",
        "ticker": ticker,
        "recommendation": recommendation,
        "signal": signal,
        "confidence": out["confidence"],
        "prob_up": round(prob, 3),
        "date": _format_date(row),
        "data_horizon": str(row.get("data_horizon", "daily_6m")),
        **decision,
        "analysis": explanation,
        "technical": build_technical_snapshot(row),
    }
    return result


def _run_sentiment(ticker: str) -> dict:
    """Return sentiment data for ticker. Used by agent tools."""
    try:
        ensure_data(ticker)
        data = get_sentiment_response(ticker)
        return {"intent": "sentiment_only", "ticker": ticker.upper(), **data}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


def _run_news(ticker: str) -> dict:
    """Return news data for ticker. Used by agent tools."""
    try:
        ensure_data(ticker)
        data = get_news_response(ticker)
        return {"intent": "news_only", "ticker": ticker.upper(), **data}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


def _run_technical(ticker: str) -> dict:
    """Return technical snapshot for ticker. Used by agent tools."""
    try:
        ensure_data(ticker)
        df = _load_sorted_final_df(ticker)
        data = get_technical_response(df)
        return {"intent": "technical_only", "ticker": ticker.upper(), **data}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


def _run_movement(ticker: str) -> dict:
    """Return movement/ATR range for ticker. Used by agent tools."""
    try:
        ensure_data(ticker)
        df = _load_sorted_final_df(ticker)
        if len(df) < 2:
            return {"intent": "movement_only", "ticker": ticker.upper(), "error": "Not enough data."}
        latest = df.iloc[-1]
        atr = _safe_float(latest, "atr_14", 0) or 0
        close = _safe_float(latest, "Close", 0) or _safe_float(latest, "close", 0) or 0
        move_data = get_movement_response(atr / close if close else 0, close - atr, close + atr)
        return {"intent": "movement_only", "ticker": ticker.upper(), **move_data}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


def _run_compare(ticker_a: str, ticker_b: str) -> dict:
    """Return compare data (left, right). Used by agent tools."""
    try:
        left = _run_full_analysis(ticker_a)
        right = _run_full_analysis(ticker_b)
        left.pop("intent", None)
        right.pop("intent", None)
        return {"left": left, "right": right}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


class _ToolRunner:
    """Provides tool implementations for the chat agent (Phase 2)."""

    def run_analysis(self, ticker: str) -> dict:
        try:
            r = _run_full_analysis(ticker.upper())
            r.pop("intent", None)
            return r
        except HTTPException as e:
            return {"error": e.detail}
        except Exception as e:
            return {"error": str(e)}

    def get_chart(self, ticker: str, days: int = 180) -> dict:
        return _chart_data(ticker, days)

    def list_tickers(self) -> dict:
        return _list_tickers()

    def compare_tickers(self, ticker_a: str, ticker_b: str) -> dict:
        return _run_compare(ticker_a, ticker_b)

    def get_sentiment(self, ticker: str) -> dict:
        return _run_sentiment(ticker)

    def get_news(self, ticker: str) -> dict:
        return _run_news(ticker)

    def get_technical(self, ticker: str) -> dict:
        return _run_technical(ticker)

    def get_movement(self, ticker: str) -> dict:
        return _run_movement(ticker)


_tool_runner = _ToolRunner()


@app.get("/analyze")
def analyze(ticker: str = Query("AAPL")):
    """Full analysis for a ticker (same as query with 'full_analysis' intent)."""
    result = _run_full_analysis(ticker.upper())
    # Keep backward compatibility: omit intent if caller only expects analyze shape
    result.pop("intent", None)
    return result


@app.post("/chat")
def chat(body: ChatRequest):
    """
    Conversational agent: send a message, get a reply and optional structured data.
    Phase 2: tries LLM with tool calling first (Ollama); falls back to intent-based flow.
    """
    message = (body.message or "").strip()
    if not message:
        return {"reply": "Send a ticker or a question, e.g. 'AAPL', 'How's MSFT?', or 'compare AAPL and NVDA'.", "data": None}

    # Phase 2: try LLM agent with tools (Ollama)
    try:
        agent_result = run_chat_agent(message, _tool_runner)
        if agent_result is not None:
            reply, data = agent_result
            return {"reply": reply or "Here’s what I found.", "data": data}
    except Exception:
        pass  # fall back to intent-based flow

    # Fallback: intent-based flow (compare, then single-ticker / intent)
    # Compare: "compare X and Y" or "X vs Y"
    ticker_a, ticker_b = _parse_compare_message(message)
    if ticker_a and ticker_b and ticker_a != ticker_b:
        try:
            left = _run_full_analysis(ticker_a)
            right = _run_full_analysis(ticker_b)
            reply = _build_reply_compare(left, right)
            # Normalize for frontend (same shape as /query)
            left.pop("intent", None)
            right.pop("intent", None)
            return {"reply": reply, "data": {"type": "compare", "left": left, "right": right}}
        except HTTPException as e:
            return {"reply": f"Could not compare: {e.detail}.", "data": None}
        except Exception as e:
            return {"reply": f"Something went wrong: {str(e)}", "data": None}

    # Single-ticker or intent-based query
    try:
        parsed = parse_user_intent(message)
        intent = parsed.get("intent", "full_analysis")
        ticker = (parsed.get("ticker") or "AAPL").strip().upper()
        if not ticker:
            return {"reply": "I couldn't find a ticker in that. Try e.g. 'AAPL' or 'news for MSFT'.", "data": None}

        if intent == "news_only":
            ensure_data(ticker)
            news_data = get_news_response(ticker)
            result = {"intent": "news_only", "ticker": ticker, **news_data}
            summary = result.get("summary", "") or (f"News for {ticker}: " + str(result.get("articles", []))[:200])
            return {"reply": summary, "data": result}

        if intent == "sentiment_only":
            ensure_data(ticker)
            sentiment_data = get_sentiment_response(ticker)
            result = {"intent": "sentiment_only", "ticker": ticker, **sentiment_data}
            avg = result.get("avg_sentiment")
            if isinstance(avg, (int, float)):
                label = "positive" if avg > 0.2 else ("negative" if avg < -0.2 else "neutral")
                parts = [f"Sentiment for {ticker}: **{label.capitalize()}** (avg {avg:.2f})."]
            else:
                parts = [f"Sentiment for {ticker}."]
            if result.get("news_sentiment") is not None:
                parts.append(f"News: {result['news_sentiment']:.2f}.")
            if result.get("reddit_sentiment") is not None:
                parts.append(f"Reddit: {result['reddit_sentiment']:.2f}.")
            if result.get("article_count") is not None:
                parts.append(f"{result['article_count']} articles.")
            if result.get("post_count") is not None:
                parts.append(f"{result['post_count']} posts.")
            reply = " ".join(parts)
            return {"reply": reply, "data": result}

        if intent == "technical_only":
            ensure_data(ticker)
            df = _load_sorted_final_df(ticker)
            tech_data = get_technical_response(df)
            result = {"intent": "technical_only", "ticker": ticker, **tech_data}
            return {"reply": f"Technical view for {ticker}: " + str(tech_data)[:300], "data": result}

        if intent == "movement_only":
            ensure_data(ticker)
            df = _load_sorted_final_df(ticker)
            if len(df) < 2:
                return {"reply": f"Not enough data for movement on {ticker}.", "data": None}
            latest = df.iloc[-1]
            atr = float(latest.get("atr_14", 0))
            close = float(latest["Close"])
            move_data = get_movement_response(atr / close if close else 0, close - atr, close + atr)
            result = {"intent": "movement_only", "ticker": ticker, **move_data}
            return {"reply": str(move_data)[:300], "data": result}

        # full_analysis
        result = _run_full_analysis(ticker)
        reply = _build_reply_single(result)
        result.pop("intent", None)
        return {"reply": reply, "data": {"type": "analysis", **result}}
    except HTTPException as e:
        return {"reply": f"Sorry, I couldn't do that: {e.detail}", "data": None}
    except Exception as e:
        return {"reply": f"Something went wrong: {str(e)}", "data": None}