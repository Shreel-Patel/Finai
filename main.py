"""
Run backtest with unified predictor (technicals + Reddit + news).
For the web API, use: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import pandas as pd
from src.models.backtest import walk_forward_backtest
from src.models.metrics import evaluate, evaluate_strategy, sweep_threshold
from src.models.config import get_features_for_df
from src.models.predictor import get_backtest_model_fn

_ROOT = Path(__file__).resolve().parent
DATA_PATH = _ROOT / "data" / "final"
TICKER = "MSFT"

df = pd.read_csv(DATA_PATH / f"{TICKER}.csv").dropna()
if "target" not in df.columns:
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
if "target_return" not in df.columns:
    df["target_return"] = (df["Close"].shift(-1) / df["Close"]) - 1.0

features = get_features_for_df(df)
# Unified predictor: technicals + Reddit sentiment + news sentiment
model_fn = get_backtest_model_fn(buy_threshold=0.55, sell_threshold=0.45)
bt = walk_forward_backtest(df, features=features, target="target", threshold=0.55, model_fn=model_fn)
print("Backtest (unified):", bt.tail())
print("Metrics:", evaluate(bt))
print("Strategy:", evaluate_strategy(bt, threshold=0.55))
print("Threshold sweep:", sweep_threshold(bt))