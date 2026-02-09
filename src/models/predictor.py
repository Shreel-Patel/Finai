"""
Unified price predictor: technicals + Reddit sentiment + news sentiment.
Uses one XGBoost classifier for direction only (UP/DOWN/HOLD). No magnitude/return prediction.
Probabilities are calibrated (isotonic) so prob_up better reflects actual likelihood of UP.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.models.config import get_features_for_df, TARGET_COLUMN  # no TARGET_RETURN: direction only

# Fixed seed for reproducible CV splits (avoids alternating prob_up across runs)
CALIBRATION_CV_SEED = 42


# Default thresholds for signal
DEFAULT_BUY_THRESHOLD = 0.55
DEFAULT_SELL_THRESHOLD = 0.45

# Clip raw probabilities so we don't show overconfident 0% or 100%
PROB_CLIP_LOW = 0.02
PROB_CLIP_HIGH = 0.98


class UnifiedPricePredictor:
    """
    Predicts direction only (UP/DOWN/HOLD) using:
    - Technicals (RSI, EMA, MACD, Bollinger, ATR, volume)
    - Reddit sentiment (compound, sent_3d, sent_7d, sent_14d, sent_delta)
    - News sentiment (news_sent)
    """

    def __init__(
        self,
        buy_threshold=DEFAULT_BUY_THRESHOLD,
        sell_threshold=DEFAULT_SELL_THRESHOLD,
        xgb_classifier_kwargs=None,
        calibrate_proba=True,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.clf_kwargs = xgb_classifier_kwargs or {}
        self.calibrate_proba = calibrate_proba
        self._clf = None
        self._features = None

    def _default_clf_params(self):
        return dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            **self.clf_kwargs,
        )

    def fit(self, df, features=None, target_dir=TARGET_COLUMN):
        """Train on dataframe with technicals + reddit + news. Requires target (direction only)."""
        if features is None:
            features = get_features_for_df(df)
        self._features = [f for f in features if f in df.columns]
        X = df[self._features].fillna(0)

        # Direction classifier (with class balance + optional probability calibration)
        if target_dir in df.columns:
            y_dir = df[target_dir].values
            n_pos = int(np.sum(y_dir == 1))
            n_neg = int(np.sum(y_dir == 0))
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            scale_pos_weight = float(np.clip(scale_pos_weight, 0.25, 4.0))
            clf_params = self._default_clf_params()
            clf_params["scale_pos_weight"] = scale_pos_weight
            base_clf = XGBClassifier(**clf_params)

            # Calibrate probabilities so prob_up better matches actual P(up)
            # Use explicit StratifiedKFold with random_state so results are deterministic across runs
            n_samples = len(y_dir)
            if self.calibrate_proba and n_samples >= 30:
                n_splits = min(5, max(2, n_samples // 15))
                cv_splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=CALIBRATION_CV_SEED
                )
                self._clf = CalibratedClassifierCV(
                    base_clf, cv=cv_splitter, method="isotonic", ensemble=True
                )
                self._clf.fit(X, y_dir)
            else:
                self._clf = base_clf
                self._clf.fit(X, y_dir)

        return self

    def predict_proba_up(self, X):
        """P(price goes up next period). Calibrated and clipped for more accurate probabilities."""
        if self._clf is None:
            raise RuntimeError("Predictor not fitted or direction model missing.")
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        X = X[self._features].fillna(0)
        prob = self._clf.predict_proba(X)[:, 1]
        prob = np.clip(prob, PROB_CLIP_LOW, PROB_CLIP_HIGH)
        return prob

    def predict_signal(self, prob_up):
        """
        Map prob_up to signal and confidence. Direction only (no magnitude).
        Returns: signal ("buy" | "sell" | "hold"), confidence (0-1).
        """
        if prob_up >= self.buy_threshold:
            signal = "buy"
            confidence = min(1.0, (prob_up - self.buy_threshold) / (1 - self.buy_threshold) + 0.5)
        elif prob_up <= self.sell_threshold:
            signal = "sell"
            confidence = min(1.0, (self.sell_threshold - prob_up) / self.sell_threshold + 0.5)
        else:
            signal = "hold"
            confidence = 0.5 - abs(prob_up - 0.5)
        return signal, round(float(np.clip(confidence, 0, 1)), 3)

    def predict(self, row_or_df):
        """
        Single row or DataFrame. Returns dict:
        - prob_up: P(up)
        - signal: "buy" | "sell" | "hold"
        - confidence: 0-1
        (No magnitude/return prediction.)
        """
        if isinstance(row_or_df, pd.DataFrame):
            prob_up = self.predict_proba_up(row_or_df).ravel()
            out = []
            for i in range(len(row_or_df)):
                sig, conf = self.predict_signal(float(prob_up[i]))
                out.append({
                    "prob_up": float(prob_up[i]),
                    "signal": sig,
                    "confidence": conf,
                })
            return out
        # single row
        prob_up = float(self.predict_proba_up(row_or_df).ravel()[0])
        signal, confidence = self.predict_signal(prob_up)
        return {
            "prob_up": prob_up,
            "signal": signal,
            "confidence": confidence,
        }

    @property
    def features(self):
        return self._features


def train_unified_predictor(
    df,
    features=None,
    target_dir=TARGET_COLUMN,
    buy_threshold=DEFAULT_BUY_THRESHOLD,
    sell_threshold=DEFAULT_SELL_THRESHOLD,
    calibrate_proba=True,
):
    """
    Train and return UnifiedPricePredictor on df (direction only, no return prediction).
    calibrate_proba=True (default) uses isotonic calibration so prob_up is more accurate.
    """
    pred = UnifiedPricePredictor(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        calibrate_proba=calibrate_proba,
    )
    pred.fit(df, features=features, target_dir=target_dir)
    return pred


def get_backtest_model_fn(buy_threshold=0.55, sell_threshold=0.45):
    """
    Return a model_fn(train, test, features, target) -> prob_up for use with
    walk_forward_backtest. Uses UnifiedPricePredictor (technicals + reddit + news).
    """
    def _fn(train, test, features, target):
        pred = train_unified_predictor(
            train, features=features, target_dir=target,
            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
        )
        return pred.predict_proba_up(test).ravel()[0]
    return _fn
