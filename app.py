from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info  = stock.info
    return {
        "name":          info.get("longName", ticker),
        "sector":        info.get("sector", "N/A"),
        "market_cap":    info.get("marketCap", 0),
        "pe_ratio":      info.get("trailingPE", 0),
        "52w_high":      info.get("fiftyTwoWeekHigh", 0),
        "52w_low":       info.get("fiftyTwoWeekLow", 0),
        "current_price": info.get("currentPrice", 0),
        "volume":        info.get("volume", 0),
    }

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def get_historical_data(ticker: str, period: str = "1y") -> dict:
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)[["Close", "Volume"]].dropna()
    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["RSI"]   = compute_rsi(df["Close"])
    df.dropna(inplace=True)

    return {
        "dates":   [str(d.date()) for d in df.index],
        "close":   df["Close"].round(2).tolist(),
        "volume":  df["Volume"].tolist(),
        "ma7":     df["MA_7"].round(2).tolist(),
        "ma21":    df["MA_21"].round(2).tolist(),
        "rsi":     df["RSI"].round(2).tolist(),
    }

def simple_forecast(ticker: str, days: int = 7) -> dict:
    """
    Lightweight forecast (no saved Keras model required).
    Uses exponential smoothing + trend extrapolation so the
    dashboard works out-of-the-box before any GPU training.
    """
    stock = yf.Ticker(ticker)
    df    = stock.history(period="6mo")[["Close"]].dropna()
    close = df["Close"].values

    # Linear trend on last 30 days
    recent = close[-30:]
    x      = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)

    # Exponential smoothing residual
    alpha    = 0.3
    smoothed = [recent[0]]
    for p in recent[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    last_smooth = smoothed[-1]

    predictions = []
    last_price  = float(close[-1])
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq="B")

    for i, d in enumerate(dates, start=1):
        trend_val = intercept + slope * (len(recent) + i - 1)
        noise     = np.random.normal(0, last_price * 0.005)   # ±0.5 % stochastic noise
        price     = 0.5 * trend_val + 0.5 * last_smooth + noise
        predictions.append({"date": str(d.date()), "price": round(float(price), 2)})

    change_pct = ((predictions[-1]["price"] - last_price) / last_price) * 100
    signal     = "BUY" if change_pct > 1 else ("SELL" if change_pct < -1 else "HOLD")

    return {
        "ticker":      ticker,
        "last_price":  round(last_price, 2),
        "predictions": predictions,
        "change_pct":  round(change_pct, 2),
        "signal":      signal,
        "method":      "Exponential Smoothing + Linear Trend",
    }

# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock/<ticker>")
def stock_info(ticker):
    try:
        return jsonify(get_stock_info(ticker.upper()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/<ticker>")
def history(ticker):
    period = request.args.get("period", "1y")
    try:
        return jsonify(get_historical_data(ticker.upper(), period))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/<ticker>")
def predict(ticker):
    days = int(request.args.get("days", 7))
    try:
        # Try full LSTM model first; fall back to simple forecast
        model_path = f"saved_models/{ticker.upper()}_model.keras"
        if os.path.exists(model_path):
            from model import predict_next_days
            result = predict_next_days(ticker.upper(), days)
        else:
            result = simple_forecast(ticker.upper(), days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train/<ticker>", methods=["POST"])
def train_model(ticker):
    try:
        from model import train
        _, _, _, rmse, mae = train(ticker.upper())
        return jsonify({"status": "success", "ticker": ticker.upper(),
                        "rmse": round(rmse, 2), "mae": round(mae, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
