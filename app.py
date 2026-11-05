import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# ── helpers ────────────────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> dict:
    """Best-effort company fundamentals with a historical-price fallback.

    Yahoo Finance's info endpoint can omit quoteType or fail independently
    of its OHLCV endpoint. Missing optional metadata is not evidence that a
    valid listed ticker does not exist.
    """
    stock = yf.Ticker(ticker)
    info = {}
    try:
        response = stock.info
        if isinstance(response, dict):
            info = response
    except Exception as exc:
        logger.warning("Fundamental metadata unavailable for %s: %s", ticker, type(exc).__name__)

    history = pd.DataFrame()
    try:
        history = stock.history(period="1y", auto_adjust=True)
    except Exception as exc:
        logger.warning("Historical metadata fallback unavailable for %s: %s", ticker, type(exc).__name__)

    def finite_number(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    price_history = pd.DataFrame()
    if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history:
        required = [col for col in ("Close", "High", "Low", "Volume") if col in history.columns]
        price_history = history[required].copy()
        price_history["Close"] = pd.to_numeric(price_history["Close"], errors="coerce")
        price_history = price_history.replace([np.inf, -np.inf], np.nan)
        price_history = price_history.dropna(subset=["Close"])
        price_history = price_history[price_history["Close"] > 0]

    has_metadata = any(info.get(field) for field in (
        "longName", "shortName", "currentPrice", "regularMarketPrice",
        "marketCap", "fiftyTwoWeekHigh"
    ))
    if price_history.empty and not has_metadata:
        raise ValueError(f"No market data is currently available for {ticker}. Please retry later.")

    def recent_value(column, operation, default):
        if column not in price_history or price_history.empty:
            return default
        values = pd.to_numeric(price_history[column], errors="coerce").dropna()
        if values.empty:
            return default
        return finite_number(operation(values.tail(252)))

    last_close = recent_value("Close", lambda values: values.iloc[-1], 0.0)
    latest_volume = recent_value("Volume", lambda values: values.iloc[-1], 0.0)
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "N/A",
        "market_cap": finite_number(info.get("marketCap")),
        "pe_ratio": finite_number(info.get("trailingPE")),
        "52w_high": finite_number(info.get("fiftyTwoWeekHigh")) or recent_value(
            "High", lambda values: values.max(), last_close
        ),
        "52w_low": finite_number(info.get("fiftyTwoWeekLow")) or recent_value(
            "Low", lambda values: values.min(), last_close
        ),
        "current_price": last_close or finite_number(
            info.get("currentPrice") or info.get("regularMarketPrice")
        ),
        "volume": latest_volume or finite_number(info.get("volume")),
    }


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def get_historical_data(ticker: str, period: str = "1y") -> dict:
    allowed_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in allowed_periods:
        period = "1y"

    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)[["Close", "Volume"]].dropna()

    if df.empty:
        raise ValueError(f"No historical data returned for ticker: {ticker}")

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
    Lightweight forecast that does not require a saved Keras model.
    Combines a linear trend fitted over the last 30 trading days with
    exponential smoothing to produce a plausible short-term estimate.
    This is the default predictor until an LSTM model is trained.
    """
    stock = yf.Ticker(ticker)
    df    = stock.history(period="6mo")[["Close"]].dropna()

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    close = df["Close"].values

    if len(close) < 30:
        raise ValueError(f"Insufficient history for ticker: {ticker} (need at least 30 trading days)")

    # Linear trend fitted on the most recent 30 trading days
    recent = close[-30:]
    x      = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)

    # Exponential smoothing of the same window
    alpha    = 0.3
    smoothed = [float(recent[0])]
    for p in recent[1:]:
        smoothed.append(alpha * float(p) + (1 - alpha) * smoothed[-1])
    last_smooth = smoothed[-1]

    last_price  = float(close[-1])
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq="B")

    predictions = []
    for i, d in enumerate(dates, start=1):
        trend_val = intercept + slope * (len(recent) + i - 1)
        noise     = np.random.normal(0, last_price * 0.005)
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
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol"}), 400
    try:
        return jsonify(get_stock_info(ticker))
    except ValueError as exc:
        logger.warning("stock_info unavailable for %s: %s", ticker, exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("stock_info unexpected error for %s", ticker)
        return jsonify({"error": "Market data is temporarily unavailable. Please retry."}), 503


@app.route("/api/history/<ticker>")
def history(ticker):
    ticker = ticker.upper().strip()
    period = request.args.get("period", "1y")
    if not ticker.isalpha() or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol"}), 400
    try:
        return jsonify(get_historical_data(ticker, period))
    except Exception as e:
        logger.warning("history error for %s: %s", ticker, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/<ticker>")
def predict(ticker):
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol"}), 400

    try:
        days = int(request.args.get("days", 7))
        if days < 1 or days > 30:
            return jsonify({"error": "days must be between 1 and 30"}), 400
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400

    try:
        model_path = f"saved_models/{ticker}_model.keras"
        if os.path.exists(model_path):
            logger.info("Using LSTM model for %s", ticker)
            from model import predict_next_days
            result = predict_next_days(ticker, days)
        else:
            logger.info("Using fallback forecast for %s", ticker)
            result = simple_forecast(ticker, days)
        return jsonify(result)
    except Exception as e:
        logger.warning("predict error for %s: %s", ticker, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/train/<ticker>", methods=["POST"])
def train_model(ticker):
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol"}), 400
    try:
        from model import train
        logger.info("Starting LSTM training for %s", ticker)
        _, _, _, rmse, mae = train(ticker)
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "rmse":   round(rmse, 4),
            "mae":    round(mae, 4),
        })
    except Exception as e:
        logger.error("train error for %s: %s", ticker, e)
        return jsonify({"error": str(e)}), 500


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)
