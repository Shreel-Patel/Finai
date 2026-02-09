def _safe_get(row, key, default=0):
    """Safe float from row; use default if missing or NaN."""
    try:
        v = row.get(key, default)
        return float(v) if v is not None and (v == v) else default  # NaN check
    except (TypeError, ValueError):
        return default


def build_state(row):
    """
    Build agent state from a single row. Uses multiple technical factors:
    trend (EMAs + ADX), momentum (RSI, MACD, Stochastic), Bollinger position,
    volume, volatility, sentiment. No single indicator drives the signal.
    """
    state = {}

    # ---- Probability (from model) ----
    state["prob_up"] = _safe_get(row, "prob_up", 0.5)

    # ---- Trend: multi-factor (not just EMA 20 vs 50) ----
    close = _safe_get(row, "Close")
    ema_9 = _safe_get(row, "ema_9") if "ema_9" in row else None
    ema_20 = _safe_get(row, "ema_20")
    ema_50 = _safe_get(row, "ema_50")
    # Short-term: EMA 9 vs EMA 20 (if available); else medium-term
    medium_bull = ema_20 > ema_50
    short_bull = (ema_9 > ema_20) if ema_9 is not None and ema_20 else medium_bull
    state["trend"] = "bullish" if medium_bull else "bearish"
    state["trend_short"] = "bullish" if short_bull else "bearish"
    # ADX: trend strength (only trust trend when ADX > 20)
    adx = _safe_get(row, "adx_14", 0)
    state["trend_strength"] = "strong" if adx > 25 else ("weak" if adx < 20 else "moderate")
    state["adx"] = adx

    # ---- Momentum: RSI ----
    rsi = _safe_get(row, "rsi_14", 50)
    if rsi < 30:
        state["momentum"] = "oversold"
    elif rsi > 70:
        state["momentum"] = "overbought"
    else:
        state["momentum"] = "neutral"
    state["rsi"] = rsi

    # ---- MACD: line vs signal + histogram ----
    macd = _safe_get(row, "macd")
    macd_sig = _safe_get(row, "macd_signal")
    macd_hist = _safe_get(row, "macd_histogram", macd - macd_sig)
    state["macd_signal"] = "bullish" if macd > macd_sig else "bearish"
    state["macd_strength"] = abs(macd - macd_sig)
    state["macd_histogram_bullish"] = macd_hist > 0

    # ---- Stochastic: oversold/overbought ----
    stoch_k = _safe_get(row, "stoch_k", 50)
    state["stoch_oversold"] = stoch_k < 20
    state["stoch_overbought"] = stoch_k > 80

    # ---- Bollinger Band position ----
    bb_upper = _safe_get(row, "bb_upper", close * 1.1)
    bb_lower = _safe_get(row, "bb_lower", close * 0.9)
    if close >= bb_upper * 0.99:
        state["bb_position"] = "upper_band"
    elif close <= bb_lower * 1.01:
        state["bb_position"] = "lower_band"
    else:
        state["bb_position"] = "middle_range"

    # ---- Volatility ----
    atr = _safe_get(row, "atr_14")
    atr_mean = _safe_get(row, "atr_14_mean_20", atr)
    state["high_volatility"] = atr > atr_mean if atr_mean else False

    # ---- Sentiment ----
    if _safe_get(row, "sent_delta", 0) > 0:
        state["sentiment"] = "improving"
    elif _safe_get(row, "sent_delta", 0) < 0:
        state["sentiment"] = "deteriorating"
    else:
        state["sentiment"] = "flat"
    comp = abs(_safe_get(row, "compound", 0))
    state["sentiment_strength"] = "strong" if comp > 0.5 else ("moderate" if comp > 0.2 else "weak")

    # ---- Volume ----
    vz = _safe_get(row, "volume_z", 0)
    if vz > 2:
        state["volume"] = "very_high"
    elif vz > 1:
        state["volume"] = "high"
    elif vz < -1:
        state["volume"] = "low"
    elif vz < -2:
        state["volume"] = "very_low"
    else:
        state["volume"] = "normal"

    # ---- Composite technical score (multi-factor; not just EMA or MACD) ----
    score = 0
    if state["trend"] == "bullish":
        score += 1
    else:
        score -= 1
    if state["macd_signal"] == "bullish":
        score += 1
    else:
        score -= 1
    if rsi < 30:
        score += 1  # oversold = potential bounce
    elif rsi > 70:
        score -= 1  # overbought = risk
    elif rsi > 50:
        score += 0.5
    else:
        score -= 0.5
    if state["bb_position"] == "lower_band":
        score += 1
    elif state["bb_position"] == "upper_band":
        score -= 1
    if state["trend_strength"] == "strong" and state["trend"] == "bullish":
        score += 0.5
    elif state["trend_strength"] == "strong" and state["trend"] == "bearish":
        score -= 0.5
    if state["stoch_oversold"]:
        score += 0.5
    elif state["stoch_overbought"]:
        score -= 0.5
    state["technical_score"] = round(score, 1)

    # Market regime: derived from trend + strength
    if state["trend_strength"] == "strong":
        state["market_regime"] = state["trend"]
    else:
        state["market_regime"] = "choppy"

    # Legacy: bullish_score 0-1 for compatibility
    state["bullish_score"] = (score + 5) / 10  # rough map to 0-1
    state["bullish_score"] = max(0, min(1, state["bullish_score"]))

    return state