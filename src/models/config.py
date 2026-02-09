"""
Model and feature config for buy/sell prediction.
Use this curated feature list instead of all numeric columns to avoid
leaking index/date and to get reproducible, interpretable models.
"""

TARGET_COLUMN = "target"
TARGET_RETURN_COLUMN = "target_return"

# Curated features: technicals + reddit + news sentiment. Excludes Date, date, Unnamed: 0, target*.
FEATURE_COLUMNS = [
    "Close", "High", "Low", "Open", "Volume",
    "rsi_14", "ema_9", "ema_20", "ema_50", "macd", "macd_signal", "macd_histogram",
    "bb_width", "bb_upper", "bb_lower", "atr_14", "atr_14_mean_20", "volume_z",
    "adx_14", "stoch_k", "stoch_d",
    "compound", "sent_3d", "sent_7d", "sent_14d", "sent_delta", "news_sent",
]

# Optional: add if present in dataset (e.g. from reddit weighted aggregation)
OPTIONAL_FEATURES = ["weighted_sentiment"]


def get_features_for_df(df):
    """Return feature list from dataframe, only including columns that exist."""
    import pandas as pd
    available = set(df.columns)
    features = [c for c in FEATURE_COLUMNS if c in available]
    for c in OPTIONAL_FEATURES:
        if c in available:
            features.append(c)
    return [c for c in features if c not in (TARGET_COLUMN, TARGET_RETURN_COLUMN)]
