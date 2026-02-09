def decide(state):
    reasons = []
    risks = []
    confirmations = 0
    warnings = 0

    prob = state["prob_up"]

    if prob > 0.6:
        bias = "Bullish"
    elif prob < 0.4:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # ====== TREND (multi-factor: EMAs + strength, not just one crossover) ======
    if state["trend"] == "bullish":
        confirmations += 1
        if state.get("trend_strength") == "strong":
            reasons.append("Uptrend with strong trend strength (ADX)")
        else:
            reasons.append("Price trend is bullish (EMAs aligned)")
    else:
        warnings += 1
        if state.get("trend_strength") == "strong":
            risks.append("Downtrend with strong trend strength (ADX)")
        else:
            risks.append("Price trend is bearish")

    # ====== MOMENTUM: RSI ======
    if state["momentum"] == "oversold":
        confirmations += 1
        reasons.append("RSI indicates oversold conditions - potential reversal")
    elif state["momentum"] == "overbought":
        warnings += 1
        risks.append("RSI indicates overbought conditions - risk of pullback")
    else:
        reasons.append("RSI in neutral range")

    # ====== STOCHASTIC (additional momentum, not just RSI/MACD) ======
    if state.get("stoch_oversold"):
        confirmations += 1
        reasons.append("Stochastic oversold - possible bounce")
    elif state.get("stoch_overbought"):
        warnings += 1
        risks.append("Stochastic overbought - caution")

    # ====== SENTIMENT ANALYSIS ======
    if state["sentiment"] == "improving":
        confirmations += 1
        reasons.append("Market sentiment is improving")
    elif state["sentiment"] == "deteriorating":
        warnings += 1
        risks.append("Market sentiment deteriorating")

    # Add sentiment strength if available
    if "sentiment_strength" in state:
        if state["sentiment_strength"] == "strong":
            weight = 2 if state["sentiment"] == "improving" else -2
            confirmations += abs(weight) if weight > 0 else 0
            warnings += abs(weight) if weight < 0 else 0
            reasons.append(f"Strong {state['sentiment']} sentiment")

    # ====== VOLATILITY ASSESSMENT ======
    if state["high_volatility"]:
        warnings += 2  # Higher weight for volatility
        risks.append("High volatility environment - increased risk")
    else:
        reasons.append("Volatility at normal levels")

    # ====== MACD (one of several momentum signals) ======
    if "macd_signal" in state:
        if state["macd_signal"] == "bullish":
            confirmations += 1
            reasons.append("MACD shows bullish momentum")
        else:
            warnings += 1
            risks.append("MACD shows bearish momentum")
        if state.get("macd_strength", 0) > 0.5:
            if state["macd_signal"] == "bullish":
                confirmations += 1
                reasons.append("Strong MACD histogram support")
            else:
                warnings += 1
                risks.append("Strong MACD bearish histogram")

    # ====== VOLUME CONFIRMATION ======
    if "volume" in state:
        volume_analysis = {
            "very_high": (2, "Very high volume - strong conviction"),
            "high": (1, "Above average volume - good participation"),
            "low": (-1, "Low volume - weak conviction"),
            "very_low": (-2, "Very low volume - caution advised")
        }

        if state["volume"] in volume_analysis:
            weight, message = volume_analysis[state["volume"]]
            if weight > 0:
                confirmations += weight
                reasons.append(message)
            else:
                warnings += abs(weight)
                risks.append(message)

    # ====== BOLLINGER BANDS ANALYSIS ======
    if "bb_position" in state:
        bb_signals = {
            "upper_band": (-1, "Price at upper Bollinger Band - resistance"),
            "lower_band": (1, "Price at lower Bollinger Band - support"),
            "middle_range": (0, "Price in middle Bollinger Band range")
        }

        if state["bb_position"] in bb_signals:
            weight, message = bb_signals[state["bb_position"]]
            if weight > 0:
                confirmations += weight
                reasons.append(message)
            elif weight < 0:
                warnings += abs(weight)
                risks.append(message)

    # ====== COMBINED TECHNICAL SCORE (multi-factor, not just EMA/MACD) ======
    if "technical_score" in state:
        ts = state["technical_score"]
        if ts >= 3:
            confirmations += 2
            reasons.append("Strong alignment across trend, momentum, and bands")
        elif ts >= 1.5:
            confirmations += 1
            reasons.append("Technical score moderately bullish")
        elif ts <= -3:
            warnings += 2
            risks.append("Multiple technical indicators bearish")
        elif ts <= -1.5:
            warnings += 1
            risks.append("Technical score moderately bearish")

    # ====== MARKET REGIME (trend strength + direction) ======
    if "market_regime" in state:
        if state["market_regime"] == "bullish":
            confirmations += 1
            reasons.append("Market regime bullish (trend confirmed)")
        elif state["market_regime"] == "bearish":
            warnings += 1
            risks.append("Market regime bearish (trend confirmed)")
        elif state["market_regime"] == "choppy":
            reasons.append("Choppy market (ADX low) - trend signals less reliable")

    # ====== FINAL DECISION LOGIC ======
    # Calculate confidence based on multiple factors
    total_signals = confirmations + warnings
    signal_ratio = confirmations / max(total_signals, 1)  # Avoid division by zero

    # Base confidence on probability AND signal alignment
    if prob > 0.7 and signal_ratio > 0.7 and confirmations >= 4:
        confidence = "Very High"
        bias = "Strongly " + bias if bias != "Neutral" else bias
    elif prob > 0.6 and signal_ratio > 0.6 and confirmations >= 3:
        confidence = "High"
    elif prob > 0.55 or (0.4 <= signal_ratio <= 0.6):
        confidence = "Medium"
    elif prob < 0.45 and warnings > confirmations:
        confidence = "High" if bias == "Bearish" else "Low"
    else:
        confidence = "Low"

    # ====== ADD CONTEXTUAL WARNINGS ======
    # Override if conflicting signals
    if bias == "Bullish" and warnings > confirmations + 2:
        confidence = "Low"
        risks.append("Multiple warnings contradict bullish bias")

    if bias == "Bearish" and confirmations > warnings + 2:
        confidence = "Low"
        reasons.append("Multiple confirmations contradict bearish bias")

    # Add signal summary
    signal_summary = f"{confirmations} confirmations vs {warnings} warnings"
    if "signal_summary" not in state:
        reasons.append(signal_summary)

    # ====== SPECIAL CASES ======
    # Extreme caution for high volatility + overbought/oversold
    if state["high_volatility"] and state["momentum"] in ["overbought", "oversold"]:
        confidence = "Very Low"
        risks.append("Extreme conditions: High volatility with " + state["momentum"])

    # Strong conviction for aligned signals
    if (bias == "Bullish" and confirmations >= 5 and warnings <= 1) or \
            (bias == "Bearish" and warnings >= 5 and confirmations <= 1):
        confidence = "Very High"

    return {
        "bias": bias,
        "confidence": confidence,
        "confidence_score": round(signal_ratio, 2),
        "confirmations": confirmations,
        "warnings": warnings,
        "reasons": reasons,
        "risks": risks,
        "probability": prob
    }