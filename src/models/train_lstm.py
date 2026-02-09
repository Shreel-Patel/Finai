"""
Simple LSTM classifier for buy/sell using a rolling window of features.
Use as an extra signal to blend with XGBoost/LightGBM (ensemble).
Requires: pip install torch  (or tensorflow)
"""
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_sequences(df, features, seq_len=20):
    """Build (X_seq, y) from dataframe. X_seq shape: (n_samples, seq_len, n_features)."""
    X = df[features].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    # Replace NaN with 0 for simplicity
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(X)
    if n <= seq_len:
        return None, None
    X_seq = np.zeros((n - seq_len, seq_len, len(features)), dtype=np.float32)
    y_seq = y[seq_len:]
    for i in range(n - seq_len):
        X_seq[i] = X[i : i + seq_len]
    return X_seq, y_seq


if TORCH_AVAILABLE:

    class SeqClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)


def train_lstm(df, features, target, seq_len=20, epochs=30, lr=1e-3, device=None):
    """
    Train a simple LSTM on rolling windows. Returns a predictor callable:
    predictor(X_df_or_array) -> proba_up (float or array).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Install PyTorch: pip install torch")

    X_seq, y_seq = build_sequences(df, features, seq_len=seq_len)
    if X_seq is None or len(X_seq) < 20:
        raise ValueError("Not enough rows for sequence length; need at least seq_len + 20.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(features)
    model = SeqClassifier(input_size=n_features, hidden_size=32, num_layers=1, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_seq, device=device)
    y_t = torch.tensor(y_seq, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    def predictor(X_seq_array):
        """X_seq_array: (n, seq_len, n_features). Returns P(up) per sample."""
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_seq_array, dtype=torch.float32, device=device)
            logits = model(x)
            return torch.sigmoid(logits).cpu().numpy()

    return predictor, model, (features, seq_len)
