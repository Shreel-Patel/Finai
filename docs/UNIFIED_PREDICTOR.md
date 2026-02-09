# Unified Price Predictor (Technicals + Reddit + News)

This is the **recommended** predictor for your collected data: it uses **technicals**, **Reddit sentiment**, and **news sentiment** in one model to produce buy/sell/hold and predicted return.

---

## What it uses

| Source | Features |
|--------|----------|
| **Technicals** | Close, High, Low, Open, Volume, RSI, EMA 20/50, MACD, Bollinger width, ATR, volume_z |
| **Reddit sentiment** | compound, sent_3d, sent_7d, sent_14d, sent_delta (optional: weighted_sentiment) |
| **News sentiment** | news_sent (FinBERT: pos − neg) |

---

## How it works

1. **Direction (buy/sell)**  
   XGBoost classifier predicts P(price goes up next day) from all features above.

2. **Return (magnitude)**  
   XGBoost regressor predicts next-day simple return `(Close_next / Close - 1)` from the same features (optional; used if `target_return` exists in the dataset).

3. **Signal**  
   - **buy**: `prob_up ≥ 0.55`  
   - **sell**: `prob_up ≤ 0.45`  
   - **hold**: otherwise  

   Confidence is derived from how far `prob_up` is from 0.5 and, when available, from predicted return size.

---

## Usage

### API (`/analyze` or `/query` with full analysis)

- Trains the unified predictor on all but the last row, then predicts on the latest row.
- Response includes: `prob_up`, `signal` (buy/sell/hold), `confidence`, `pred_return` (if return target was built), plus your existing decision rules and explanation.

### Backtest (`main.py`)

```bash
python main.py
```

- Uses walk-forward backtest with the unified predictor (technicals + Reddit + news).
- Prints classification metrics, strategy Sharpe, max drawdown, and a threshold sweep (best threshold by Sharpe).

### Programmatic

```python
from src.models.predictor import train_unified_predictor
from src.models.config import get_features_for_df

df = ...  # your final CSV with target and optionally target_return
features = get_features_for_df(df)
predictor = train_unified_predictor(df[:-1], features=features)
out = predictor.predict(df.iloc[-1])
# out["prob_up"], out["signal"], out["confidence"], out["pred_return"]
```

---

## Building the dataset

`data/features/build_dataset.py` now:

- Merges **price** → **technicals** → **Reddit sentiment** → **news sentiment**.
- Creates **target** (next day up = 1, down = 0) and **target_return** (next-day simple return).
- Uses paths relative to project root.

Run the full pipeline for a ticker (price, Reddit, news, sentiment, then build_dataset) so the final CSV has all columns. Then the unified predictor uses technicals + Reddit + news automatically.

---

## Class balance and sell bias

If the model predicts **mostly sell** across tickers, the usual cause is **class imbalance**: the training data has more "down" days (target=0) than "up" days (target=1). XGBoost then tends to predict the majority class, so `prob_up` stays low and the signal is often "sell".

**What we do:** The predictor now sets **`scale_pos_weight`** automatically from the training labels: `scale_pos_weight = n_neg / n_pos`, clipped to `[0.25, 4.0]`. This upweights the "up" (positive) class when it is in the minority, so probabilities are better balanced.

**Optional tuning:** If you still see too many sells, you can:
- Widen the hold zone by using **symmetric thresholds** (e.g. buy ≥ 0.52, sell ≤ 0.48) so fewer days fall into sell.
- In backtest, use `sweep_threshold(bt)` to pick a threshold that maximizes your chosen metric (e.g. Sharpe) and pass that into `get_backtest_model_fn` or the API if you expose threshold as a parameter.
