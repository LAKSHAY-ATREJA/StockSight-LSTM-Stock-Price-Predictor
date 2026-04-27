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
import os

SEQUENCE_LENGTH = 60
MODEL_DIR = "saved_models"

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df = df[["Close", "Volume", "High", "Low", "Open"]].dropna()
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""
    df = df.copy()
    df["MA_7"]  = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["RSI"]   = compute_rsi(df["Close"])
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"]   = df["Close"].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def prepare_sequences(data: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """Convert time series into supervised learning sequences."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])          # Predict Close price (index 0)
    return np.array(X), np.array(y)

def build_model(input_shape: tuple) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train(ticker: str = "AAPL"):
    """Full training pipeline."""
    print(f"[INFO] Fetching data for {ticker}...")
    df = fetch_stock_data(ticker)
    df = create_features(df)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = prepare_sequences(scaled)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[INFO] Training samples: {len(X_train)} | Validation: {len(X_val)}")

    model = build_model((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # Evaluate
    preds_scaled = model.predict(X_val)
    # Inverse transform only the Close column
    dummy = np.zeros((len(preds_scaled), scaled.shape[1]))
    dummy[:, 0] = preds_scaled.ravel()
    preds = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_val
    actuals = scaler.inverse_transform(dummy)[:, 0]

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    print(f"[RESULT] RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{ticker}_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"[INFO] Model saved to {MODEL_DIR}/")

    return model, scaler, history, rmse, mae

def predict_next_days(ticker: str, days: int = 7):
    """Load saved model and predict next N days."""
    model_path  = f"{MODEL_DIR}/{ticker}_model.keras"
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model for {ticker}. Train first.")

    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df     = fetch_stock_data(ticker, period="1y")
    df     = create_features(df)
    scaled = scaler.transform(df.values)

    sequence = scaled[-SEQUENCE_LENGTH:]
    predictions = []

    for _ in range(days):
        inp  = sequence.reshape(1, SEQUENCE_LENGTH, scaled.shape[1])
        pred = model.predict(inp, verbose=0)[0, 0]
        predictions.append(pred)
        new_row       = sequence[-1].copy()
        new_row[0]    = pred
        sequence      = np.vstack([sequence[1:], new_row])

    # Inverse transform
    dummy = np.zeros((len(predictions), scaled.shape[1]))
    dummy[:, 0] = predictions
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]

    last_price = df["Close"].iloc[-1]
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq="B")

    return {
        "ticker": ticker,
        "last_price": round(float(last_price), 2),
        "predictions": [
            {"date": str(d.date()), "price": round(float(p), 2)}
            for d, p in zip(dates, predicted_prices)
        ]
    }

if __name__ == "__main__":
    train("AAPL")
