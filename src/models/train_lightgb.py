"""
LightGBM classifier for buy/sell (P(up)) prediction.
Use as alternative or in ensemble with XGBoost.
"""
try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def train_lightgb(df, features, target, **kwargs):
    if lgb is None:
        raise ImportError("Install lightgbm: pip install lightgbm")

    default_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        metric="binary_logloss",
        random_state=42,
        verbosity=-1,
    )
    default_params.update(kwargs)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(df[features], df[target])
    return model
