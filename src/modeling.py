"""
ARIMA/SARIMA and LSTM modeling utilities.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from tensorflow import keras
from tensorflow.keras import layers

# -------------------- ARIMA --------------------
def train_arima(series: pd.Series):
    """
    Auto-ARIMA on (log) price or returns series.
    Expects a 1-D pd.Series indexed by datetime.
    """
    s = series.dropna().astype(float)
    model = pm.auto_arima(
        s,
        seasonal=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
        max_p=5, max_q=5, max_d=2
    )
    return model

def forecast_arima(model, n_periods: int) -> pd.Series:
    fc, conf = model.predict(n_periods=n_periods, return_conf_int=True)
    idx = pd.RangeIndex(start=0, stop=n_periods)
    out = pd.Series(fc, index=idx, name="forecast")
    return out

# -------------------- LSTM --------------------
def make_sequences(arr: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i, :])
        y.append(arr[i, 0])  # predict first feature (e.g., scaled price)
    return np.array(X), np.array(y)

def build_lstm(n_features: int, lookback: int = 60, units: int = 64, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units//2),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def prepare_lstm_data(series: pd.Series, lookback: int = 60, split_date: Optional[str] = None):
    s = series.dropna().astype(float)
    df = pd.DataFrame({"target": s})
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    if split_date:
        split_idx = df.index.get_indexer([pd.to_datetime(split_date)], method="nearest")[0]
    else:
        split_idx = int(len(df)*0.8)
    train_scaled = scaled[:split_idx]
    test_scaled  = scaled[split_idx - lookback:]  # include lookback overlap
    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test, y_test   = make_sequences(test_scaled, lookback)
    return X_train, y_train, X_test, y_test, scaler, df.index[split_idx:]

