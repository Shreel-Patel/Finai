import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

def evaluate(df, zero_division=0):
    """Classification metrics for backtest predictions."""
    acc = accuracy_score(df["actual"], df["prediction"])
    prec = precision_score(df["actual"], df["prediction"], zero_division=zero_division)
    rec = recall_score(df["actual"], df["prediction"], zero_division=zero_division)
    f1 = f1_score(df["actual"], df["prediction"], zero_division=zero_division)
    hit_up = df[df["prediction"] == 1]["actual"].mean() if (df["prediction"] == 1).any() else 0.0
    hit_down = (
        (1 - df[df["prediction"] == 0]["actual"]).mean()
        if (df["prediction"] == 0).any()
        else 0.0
    )
    try:
        ll = log_loss(df["actual"], df["prob_up"])
    except Exception:
        ll = None
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "hit_rate": hit_up,
        "hit_rate_down": hit_down,
        "log_loss": ll,
    }


def strategy_returns(df, returns_series=None, threshold=0.55):
    """
    Strategy: long when prob_up > threshold, short when prob_up < (1-threshold).
    If returns_series is None, assumes 'actual' is next-day direction (1=up, 0=down)
    and builds synthetic returns as +1/-1 for display (normalize by your risk).
    For real PnL pass a Series of next-day returns aligned by index.
    """
    pred = df["prediction"].values if "prediction" in df.columns else (df["prob_up"].values > threshold).astype(int)
    prob = df["prob_up"].values
    long_signal = prob > threshold
    short_signal = prob < (1 - threshold)
    if returns_series is not None:
        ret = np.asarray(returns_series, dtype=float)
        ret = ret[: len(long_signal)]
        strategy_ret = np.where(long_signal[: len(ret)], ret, np.where(short_signal[: len(ret)], -ret, 0.0))
    else:
        # Synthetic: actual 1 -> +1, 0 -> -1 for direction
        actual = df["actual"].values
        strategy_ret = np.where(long_signal, 2 * actual - 1, np.where(short_signal, 1 - 2 * actual, 0.0))
    return pd.Series(strategy_ret, index=df.index[: len(strategy_ret)])


def sharpe_ratio(returns, periods_per_year=252, risk_free=0.0):
    """Annualized Sharpe. 'returns' is a 1d array or Series."""
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return 0.0
    excess = r.mean() - risk_free
    return (excess / r.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns):
    """Max drawdown from cumulative returns."""
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r) if np.any(r != 0) else np.ones_like(r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    return float(np.min(dd))


def evaluate_strategy(df, threshold=0.55):
    """
    Strategy metrics: synthetic daily returns from predictions, then Sharpe and max drawdown.
    For production, pass real next-day returns into strategy_returns and recompute.
    """
    ret = strategy_returns(df, returns_series=None, threshold=threshold)
    # Scale to small daily moves for more readable Sharpe (e.g. 1% per day)
    ret = ret * 0.01
    sharpe = sharpe_ratio(ret)
    dd = max_drawdown(ret)
    return {"strategy_sharpe": sharpe, "max_drawdown": dd}


def sweep_threshold(df, thresholds=None):
    """
    Sweep decision threshold by strategy Sharpe; return best threshold and metrics.
    Use this to pick threshold for buy/sell (e.g. in backtest or API).
    """
    if thresholds is None:
        thresholds = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    best_sharpe = -np.inf
    best_t = 0.55
    best_metrics = {}
    for t in thresholds:
        m = evaluate_strategy(df, threshold=t)
        if m["strategy_sharpe"] > best_sharpe:
            best_sharpe = m["strategy_sharpe"]
            best_t = t
            best_metrics = m
    return {"best_threshold": best_t, **best_metrics}
