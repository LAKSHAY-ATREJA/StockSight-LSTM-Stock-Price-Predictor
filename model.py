"""
model.py — LSTM training and inference pipeline for StockSight.

The model is a stacked LSTM that takes a sliding window of 60 trading days
and predicts the next-day closing price. Ten features are used as inputs:
Close, Volume, High, Low, Open, 7-day MA, 21-day MA, RSI-14, daily price
change percentage, and 10-day rolling volatility.

Usage (standalone training):
    python model.py               # trains AAPL by default
    python model.py TSLA          # trains the specified ticker
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60
MODEL_DIR       = "saved_models"

# Feature columns must remain stable across train and predict
FEATURE_COLS = ["Close", "Volume", "High", "Low", "Open",
                "MA_7", "MA_21", "RSI", "Price_Change", "Volatility"]


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return a clean DataFrame."""
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df = df[["Close", "Volume", "High", "Low", "Open"]].dropna()
    if len(df) < SEQUENCE_LENGTH + 30:
        raise ValueError(
            f"Insufficient history for {ticker}: got {len(df)} rows, "
            f"need at least {SEQUENCE_LENGTH + 30}."
        )
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer technical indicator features and return a stable column-ordered DataFrame."""
    df = df.copy()
    df["MA_7"]         = df["Close"].rolling(window=7).mean()
    df["MA_21"]        = df["Close"].rolling(window=21).mean()
    df["RSI"]          = compute_rsi(df["Close"])
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"]   = df["Close"].rolling(window=10).std()
    df.dropna(inplace=True)
    # Reorder to match FEATURE_COLS exactly so the scaler indices are consistent
    return df[FEATURE_COLS]


def prepare_sequences(data: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """
    Convert a 2-D time-series array into supervised learning sequences.

    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)  — target is the Close price (column index 0)
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(input_shape: tuple) -> tf.keras.Model:
    """
    Stacked LSTM model:
        LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(32) -> Dense(1)
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train(ticker: str = "AAPL"):
    """
    Full training pipeline for one ticker.

    Saves:
        saved_models/<TICKER>_model.keras   — the trained Keras model
        saved_models/<TICKER>_scaler.pkl    — the fitted MinMaxScaler

    Returns:
        model, scaler, history, rmse, mae
    """
    logger.info("Fetching data for %s ...", ticker)
    df = fetch_stock_data(ticker)
    df = create_features(df)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = prepare_sequences(scaled)

    split    = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("Training samples: %d  |  Validation samples: %d", len(X_train), len(X_val))

    model = build_model((X_train.shape[1], X_train.shape[2]))
    es    = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    # Evaluate on the validation set
    preds_scaled = model.predict(X_val, verbose=0)

    n_features = scaled.shape[1]
    dummy_pred  = np.zeros((len(preds_scaled), n_features))
    dummy_pred[:, 0] = preds_scaled.ravel()
    preds = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_act  = np.zeros((len(y_val), n_features))
    dummy_act[:, 0] = y_val
    actuals = scaler.inverse_transform(dummy_act)[:, 0]

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae  = float(mean_absolute_error(actuals, preds))
    logger.info("RMSE: %.4f  |  MAE: %.4f", rmse, mae)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{ticker}_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    logger.info("Saved model and scaler to %s/", MODEL_DIR)

    return model, scaler, history, rmse, mae


def predict_next_days(ticker: str, days: int = 7) -> dict:
    """
    Load a pre-trained model and predict the next `days` closing prices.

    The prediction loop feeds each forecast back into the sequence as the
    new Close value while keeping all other features constant from the last
    known row. This is a standard autoregressive inference approach.

    Raises FileNotFoundError if no saved model exists for the ticker.
    """
    model_path  = f"{MODEL_DIR}/{ticker}_model.keras"
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model for {ticker}. "
            f"Train first: python model.py {ticker}  or  POST /api/train/{ticker}"
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file missing for {ticker}. Re-train the model."
        )

    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df     = fetch_stock_data(ticker, period="1y")
    df     = create_features(df)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data to build a {SEQUENCE_LENGTH}-step sequence for {ticker}."
        )

    scaled   = scaler.transform(df.values)
    sequence = scaled[-SEQUENCE_LENGTH:].copy()

    n_features   = scaled.shape[1]
    raw_preds    = []

    for _ in range(days):
        inp  = sequence.reshape(1, SEQUENCE_LENGTH, n_features)
        pred = float(model.predict(inp, verbose=0)[0, 0])
        raw_preds.append(pred)
        new_row    = sequence[-1].copy()
        new_row[0] = pred                          # update Close column
        sequence   = np.vstack([sequence[1:], new_row])

    # Inverse-transform predictions back to USD
    dummy = np.zeros((len(raw_preds), n_features))
    dummy[:, 0] = raw_preds
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]

    last_price = float(df["Close"].iloc[-1])
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq="B")

    change_pct = ((float(predicted_prices[-1]) - last_price) / last_price) * 100
    signal     = "BUY" if change_pct > 1 else ("SELL" if change_pct < -1 else "HOLD")

    return {
        "ticker":      ticker,
        "last_price":  round(last_price, 2),
        "predictions": [
            {"date": str(d.date()), "price": round(float(p), 2)}
            for d, p in zip(dates, predicted_prices)
        ],
        "change_pct":  round(change_pct, 2),
        "signal":      signal,
        "method":      "LSTM",
    }


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    train(ticker)
