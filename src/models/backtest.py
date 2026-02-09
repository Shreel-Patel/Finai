import numpy as np
import pandas as pd

from src.models.train_model import train_model
from src.models.config import get_features_for_df, TARGET_COLUMN


def _date_column(df):
    """Use 'date' or 'Date' whichever exists (prefer lowercase from merge)."""
    if "date" in df.columns:
        return "date"
    if "Date" in df.columns:
        return "Date"
    return None


def walk_forward_backtest(
    df,
    features=None,
    target=TARGET_COLUMN,
    train_size=60,
    threshold=0.55,
    model_fn=None,
):
    """
    Walk-forward backtest: at each step train on past data, predict next row.
    model_fn(train_df, test_df, features, target) -> prob_up (float).
    Default: train_model (LogisticRegression).
    """
    if features is None:
        features = get_features_for_df(df)
    if model_fn is None:
        def _default_model_fn(train, test, feats, tgt):
            model = train_model(train, feats, tgt)
            return model.predict_proba(test[feats])[0][1]
        model_fn = _default_model_fn

    date_col = _date_column(df)
    predictions = []

    for i in range(train_size, len(df) - 1):
        train = df.iloc[:i]
        test = df.iloc[i : i + 1]
        prob = model_fn(train, test, features, target)
        pred = int(prob > threshold)
        actual = test[target].values[0]
        date_val = test[date_col].values[0] if date_col else i
        predictions.append({
            "date": date_val,
            "prob_up": prob,
            "prediction": pred,
            "actual": actual,
        })

    return pd.DataFrame(predictions)
