# ML/DL Recommendations for Buy/Sell Prediction (FINAI)

This document suggests **machine learning and deep learning** choices to improve **buy/sell (or hold)** predictions in your project.

---

## 1. Current Setup (Summary)

- **Data**: Price (OHLCV), technicals (RSI, EMA 20/50, MACD, Bollinger, ATR, volume_z), sentiment (Reddit compound, news_sent, rolling windows: sent_3d, sent_7d, sent_14d, sent_delta).
- **Target**: Binary — next day up (1) vs down (0): `Close.shift(-1) > Close`.
- **Models**: Logistic Regression (`train_model.py`) and XGBoost (`train_xgb.py`).
- **Decision**: Rule-based layer (`decision_rules.py`) uses model probability + technical/sentiment state to output bias (Bullish/Bearish) and confidence.

---

## 2. Recommended Models (What to Use)

### Tier 1 — Tabular (best fit for your feature table)

| Model | Use case | Pros | Cons |
|-------|----------|------|------|
| **XGBoost** (already in project) | Default choice for buy/sell probability | Handles mixed features, robust, fast, feature importance | Needs tuning; can overfit on small data |
| **LightGBM** | Alternative or ensemble with XGB | Often better with many samples; fast; good with categoricals | Slightly more hyperparameters |
| **CatBoost** | When you add categoricals (e.g. sector, regime) | Handles categories natively; good default behavior | Heavier dependency |
| **Logistic Regression** | Baseline and interpretability | Simple, stable, coefficients interpretable | Linear; limited capacity |

**Recommendation**: Use **XGBoost or LightGBM** as the main classifier. Keep **Logistic Regression** as a baseline. Consider an **ensemble** (e.g. average probability of LR + XGB + LightGBM) for more stable signals.

### Tier 2 — Deep learning (when you have sequences)

| Model | Use case | Pros | Cons |
|-------|----------|------|------|
| **LSTM / GRU** | Sequence of last N days (technicals + sentiment) | Captures temporal structure; can learn momentum/regime | Needs more data; slower; more tuning |
| **1D CNN** | Short patterns (e.g. 5–20 days) | Fast; good at local patterns | Less long-range than LSTM |
| **Transformer (time series)** | If you go to longer history + many series | State-of-the-art potential | Heavy; needs a lot of data and compute |

**Recommendation**: Add a **simple LSTM (or GRU)** that takes a **rolling window** (e.g. 20–30 days) of your existing features. Use it as an **extra signal** (probability) to blend with XGBoost, or as a dedicated “sequence model” in an ensemble. Do not replace the tabular model with LSTM until you have enough history (e.g. 2+ years of daily data).

### Tier 3 — What to avoid (for this project)

- **Naive “predict next close” regression** as the only signal — direction (up/down) is usually more stable than point forecasts.
- **Very large transformers** unless you have a lot of tickers and long history.
- **Using raw price as only input** — your technicals + sentiment are the right kind of features.

---

## 3. Target Definition (Buy / Sell / Hold)

- **Current**: Binary — next day up (1) vs down (0).
- **Improvements**:
  1. **Threshold on move size**: e.g. only “up” if next-day return &gt; 0.2% to avoid noisy flat days.
  2. **Three-class**: Strong down (0) / Hold (1) / Strong up (2) using return percentiles (e.g. bottom 33% / middle / top 33%). Use a classifier with `num_class=3` and cross-entropy.
  3. **Multi-horizon**: Train separate models (or multi-output) for 1-day, 3-day, 5-day direction and combine (e.g. only “buy” if 1d and 3d both bullish).

**Recommendation**: Keep binary for now; add a **minimum move threshold** (e.g. 0.1–0.2%) so the target is “meaningful up/down” rather than “any up/down”. Optionally add a 3-class model later for “strong buy / hold / strong sell”.

---

## 4. Feature Engineering

- **Already good**: RSI, EMA 20/50, MACD, Bollinger, ATR, volume_z, sentiment rolling (sent_3d, sent_7d, sent_14d, sent_delta), news_sent.
- **Add**:
  - **Lags**: 1–3 day lags of RSI, returns, sentiment (e.g. `sent_3d_lag1`, `return_lag1`).
  - **Return**: 1-day (and optionally 3-day) log return; can replace raw Close in the model if you already use technicals.
  - **Volatility regime**: Rolling std of returns (e.g. 20d) or ATR/Close ratio as a feature.
  - **Time**: Day-of-week or “days to next earnings” if you have calendar data (optional).

Use a **curated feature list** (e.g. `src.models.config.FEATURE_COLUMNS` / `get_features_for_df`) so you don’t feed index, date, or target into the model.

---

## 5. Training and Validation

- **Walk-forward**: You already do walk-forward in `backtest.py`. Keep it; it avoids look-ahead bias.
- **Train size**: 60–252 days (about 3–12 months). Tune with a validation fold (e.g. last 20% of training window).
- **Purging**: Drop a few days between train and test (e.g. 1–2 days) to avoid overlap from rolling features.
- **Class imbalance**: If “up” is not ~50%, use `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost) or oversampling (e.g. SMOTE) with care.

---

## 6. Decision Threshold (Buy vs Sell)

- **Current**: Fixed threshold 0.55 for “up”.
- **Improvement**: Choose threshold by **strategy performance**, not accuracy:
  - For each threshold in [0.45, 0.50, 0.55, 0.60, 0.65], run a simple rule: if prob &gt; threshold → long; if prob &lt; (1−threshold) → short; else flat.
  - Compute **Sharpe ratio** (or PnL) of that strategy on a validation period.
  - Pick the threshold that maximizes Sharpe (or minimizes drawdown).

This gives a **trading-oriented** calibration (when to act vs when to stay neutral).

---

## 7. Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, **F1**, and **per-class** metrics (especially for 3-class). Use **log loss** for probability quality.
- **Trading**: Add **strategy Sharpe ratio** and **max drawdown** on a paper portfolio that trades on your signal (e.g. long 1 unit when prob &gt; 0.55, short when &lt; 0.45). These matter more than accuracy for “should I buy or sell”.

Your `metrics.py` is extended in code to include F1, optional threshold sweep, and strategy Sharpe/drawdown.

---

## 8. End-to-End Flow Suggestion

1. **Build dataset**: Same as now (price → technicals → sentiment merge → target). Optionally add lags and return.
2. **Features**: Use `config.get_features_for_df(df)` so only valid, curated columns are used.
3. **Train**:  
   - **Primary**: XGBoost (or LightGBM) for P(up).  
   - **Optional**: LSTM on last 20–30 days of the same features → second probability.  
   - **Optional**: Average the two probabilities (or use a small meta-learner) for final `prob_up`.
4. **Backtest**: Walk-forward; at each step predict `prob_up`; apply your chosen **threshold** (e.g. 0.52 from validation) to get buy/sell/hold.
5. **Decision layer**: Keep `decision_rules.py`: combine `prob_up` with trend, RSI, sentiment, volatility, MACD, volume as you already do. Optionally feed **ensemble** `prob_up` instead of a single model.
6. **API**: Continue returning bias, confidence, and explanation; optionally expose `prob_up` and threshold used.

---

## 9. Summary Table (What to Implement)

| Priority | Item | Action |
|----------|------|--------|
| High | Curated features | Use `config.FEATURE_COLUMNS` / `get_features_for_df` everywhere (backtest, API, main). |
| High | Threshold tuning | Sweep threshold by strategy Sharpe in backtest; use best in production. |
| High | Strategy metrics | Report Sharpe and max drawdown in backtest. |
| Medium | LightGBM | Add `train_lightgb.py`; use as alternative or in ensemble with XGB. |
| Medium | LSTM | Add simple LSTM on rolling window; blend probability with XGB. |
| Medium | Target refinement | Optional: minimum move size; optional 3-class (strong down / hold / strong up). |
| Low | More lags | Add 1–3 day lags of key features in `build_dataset`. |

This setup should give you **better-calibrated buy/sell signals** and a path to **deeper models (LSTM)** and **multi-class (strong buy/hold/strong sell)** when you’re ready.
