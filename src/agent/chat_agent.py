"""
Phase 2: LLM agent with tool calling. Uses Ollama to decide which tools to call
and generates a natural-language reply. Falls back to None if Ollama is unavailable
or tool calling fails (caller should use existing intent-based flow).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

# Use your Ollama model (gemma:2b). Override with env FINAI_CHAT_MODEL if needed.
# If the model doesn't support tool calling, chat falls back to intent-based flow.
DEFAULT_MODEL = os.environ.get("FINAI_CHAT_MODEL", "gemma:2b")


def _execute_tool(name: str, arguments: dict, tools_map: dict[str, Callable[..., Any]]) -> Any:
    """Execute a single tool by name with given arguments."""
    fn = tools_map.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = fn(**arguments)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_chat_agent(
    message: str,
    runner: Any,
    model: str = DEFAULT_MODEL,
    max_tool_rounds: int = 3,
) -> tuple[str, dict | None] | None:
    """
    Run the LLM agent with tools. runner must have:
      run_analysis(ticker) -> dict
      get_chart(ticker, days=180) -> dict
      list_tickers() -> dict
      compare_tickers(ticker_a, ticker_b) -> dict with left, right
      get_sentiment(ticker) -> dict
      get_news(ticker) -> dict
      get_technical(ticker) -> dict
      get_movement(ticker) -> dict
    Returns (reply, data) or None if agent unavailable / failed.
    """
    if ollama is None:
        return None

    def get_analysis(ticker: str) -> dict:
        """Get full analysis for a stock ticker (recommendation, probability, signal, technicals)."""
        return runner.run_analysis(ticker)

    def get_chart(ticker: str, days: int = 180) -> dict:
        """Get price chart data for a ticker (dates and closes). days: 7-365, default 180."""
        return runner.get_chart(ticker, days=min(365, max(7, days)))

    def list_tickers() -> dict:
        """List supported ticker symbols (from available data)."""
        return runner.list_tickers()

    def compare_tickers(ticker_a: str, ticker_b: str) -> dict:
        """Compare two tickers side by side (full analysis for each)."""
        return runner.compare_tickers(ticker_a.strip().upper(), ticker_b.strip().upper())

    def get_sentiment(ticker: str) -> dict:
        """Get sentiment for a ticker (news + Reddit, weighted average)."""
        return runner.get_sentiment(ticker)

    def get_news(ticker: str) -> dict:
        """Get recent news headlines and sentiment for a ticker."""
        return runner.get_news(ticker)

    def get_technical(ticker: str) -> dict:
        """Get technical analysis snapshot (RSI, EMA, MACD, etc.) for a ticker."""
        return runner.get_technical(ticker)

    def get_movement(ticker: str) -> dict:
        """Get expected price movement (ATR-based range) for a ticker."""
        return runner.get_movement(ticker)

    tools = [get_analysis, get_chart, list_tickers, compare_tickers, get_sentiment, get_news, get_technical, get_movement]
    tools_map = {f.__name__: f for f in tools}

    messages = [
        {
            "role": "system",
            "content": "You are a helpful market analyst. Use the tools to fetch data when the user asks about a ticker, comparison, sentiment, news, technicals, or movement. Reply in 1-3 short sentences unless they ask for more. Always call the appropriate tool(s) first, then summarize the result.",
        },
        {"role": "user", "content": message},
    ]

    collected_data: dict | None = None
    reply_text = ""

    for _ in range(max_tool_rounds):
        try:
            response = ollama.chat(model=model, messages=messages, tools=tools)
        except Exception:
            return None

        # Handle both dict and object response (ollama package)
        msg = getattr(response, "message", None) or response.get("message") if isinstance(response, dict) else {}
        if hasattr(msg, "content"):
            msg = {"content": getattr(msg, "content", ""), "tool_calls": getattr(msg, "tool_calls", None)}
        if not isinstance(msg, dict):
            msg = {}
        content = (msg.get("content") or "").strip()
        raw_tc = msg.get("tool_calls")
        tool_calls = list(raw_tc) if raw_tc and hasattr(raw_tc, "__iter__") and not isinstance(raw_tc, str) else (raw_tc if isinstance(raw_tc, list) else [])

        if content:
            reply_text = content

        if not tool_calls:
            break

        # Append assistant message with tool_calls
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if not isinstance(fn, dict) and fn is not None:
                fn = {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "{}")}
            fn = fn or {}
            name = fn.get("name") or ""
            args_str = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str if isinstance(args_str, dict) else {})
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(name, args, tools_map)
            # Keep first analysis/compare for frontend
            if collected_data is None and name == "get_analysis" and isinstance(result, dict) and "error" not in result:
                collected_data = {"type": "analysis", **result}
            elif collected_data is None and name == "compare_tickers" and isinstance(result, dict) and "left" in result:
                collected_data = {"type": "compare", "left": result.get("left"), "right": result.get("right")}
            elif collected_data is None and name == "get_sentiment" and isinstance(result, dict):
                collected_data = {"intent": "sentiment_only", "ticker": result.get("ticker"), **result}
            elif collected_data is None and name == "get_news" and isinstance(result, dict):
                collected_data = {"intent": "news_only", "ticker": result.get("ticker"), **result}
            elif collected_data is None and name == "get_technical" and isinstance(result, dict):
                collected_data = {"intent": "technical_only", "ticker": result.get("ticker"), **result}
            elif collected_data is None and name == "get_movement" and isinstance(result, dict):
                collected_data = {"intent": "movement_only", "ticker": result.get("ticker"), **result}

            content_str = json.dumps(result) if not isinstance(result, str) else result
            if len(content_str) > 4000:
                content_str = content_str[:4000] + '..." (truncated)'
            messages.append({
                "role": "tool",
                "name": name,
                "content": content_str,
            })

    if not reply_text and collected_data:
        reply_text = "Here’s the data you asked for."
    if not reply_text:
        reply_text = "I couldn’t complete that. Try asking for a ticker (e.g. AAPL) or 'compare AAPL and NVDA'."

    return (reply_text, collected_data)
