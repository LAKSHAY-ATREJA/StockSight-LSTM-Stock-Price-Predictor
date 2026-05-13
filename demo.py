"""
demo.py — Standalone demonstration for StockSight.

Runs the exponential-smoothing + linear-trend forecast for one or more
tickers without starting the Flask web server. If a saved LSTM model
exists for a ticker, it is used automatically.

Usage:
    python demo.py                   # forecast AAPL, TSLA, NVDA
    python demo.py MSFT GOOG         # forecast specific tickers
    python demo.py AAPL --train      # train LSTM for AAPL, then forecast
    python demo.py AAPL --days 14    # forecast 14 days ahead
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import yfinance as yf


# ── simple forecast (no TensorFlow required) ───────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def simple_forecast(ticker: str, days: int = 7) -> dict:
    """Exponential smoothing + linear trend forecast (no model required)."""
    stock = yf.Ticker(ticker)
    df    = stock.history(period="6mo")[["Close"]].dropna()

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    if len(df) < 30:
        raise ValueError(f"Insufficient history for {ticker}")

    close  = df["Close"].values
    recent = close[-30:]
    x      = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)

    alpha    = 0.3
    smoothed = [float(recent[0])]
    for p in recent[1:]:
        smoothed.append(alpha * float(p) + (1 - alpha) * smoothed[-1])
    last_smooth = smoothed[-1]

    last_price = float(close[-1])
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


# ── display helpers ────────────────────────────────────────────────────────────

def print_separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def print_forecast(result: dict) -> None:
    ticker     = result["ticker"]
    last_price = result["last_price"]
    method     = result.get("method", "Unknown")
    signal     = result.get("signal", "HOLD")
    change_pct = result.get("change_pct", 0.0)

    print_separator("=")
    print(f"  {ticker}   |   Last close: ${last_price:.2f}   |   Method: {method}")
    print_separator("=")
    print(f"  {'Day':<4}  {'Date':<12}  {'Forecast ($)':>13}  {'vs Last (%)'}")
    print_separator()

    for i, p in enumerate(result["predictions"], start=1):
        chg   = (p["price"] - last_price) / last_price * 100
        arrow = "+" if chg >= 0 else ""
        print(f"  {i:<4}  {p['date']:<12}  ${p['price']:>11.2f}  {arrow}{chg:.2f}%")

    print_separator()
    signal_line = f"  Signal: {signal}   |   7-day change: {change_pct:+.2f}%"
    print(signal_line)
    print()


def print_stock_summary(ticker: str) -> None:
    """Print current market stats from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        name    = info.get("longName", ticker)
        price   = info.get("currentPrice", 0) or 0
        cap     = info.get("marketCap", 0) or 0
        pe      = info.get("trailingPE", 0) or 0
        hi52    = info.get("fiftyTwoWeekHigh", 0) or 0
        lo52    = info.get("fiftyTwoWeekLow", 0) or 0

        cap_str = (
            f"${cap/1e12:.2f}T" if cap >= 1e12 else
            f"${cap/1e9:.2f}B"  if cap >= 1e9  else
            f"${cap/1e6:.2f}M"  if cap > 0     else "N/A"
        )

        print(f"\n  {name}")
        print(f"  Current Price : ${price:.2f}")
        print(f"  Market Cap    : {cap_str}")
        print(f"  P/E Ratio     : {pe:.1f}" if pe else "  P/E Ratio     : N/A")
        print(f"  52-week range : ${lo52:.2f} - ${hi52:.2f}")
    except Exception as e:
        print(f"  (Could not load market data: {e})")


# ── optional matplotlib chart ─────────────────────────────────────────────────

def plot_forecast(result: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ticker     = result["ticker"]
        last_price = result["last_price"]
        dates      = [pd.Timestamp(p["date"]) for p in result["predictions"]]
        prices     = [p["price"] for p in result["predictions"]]
        colors     = ["green" if p >= last_price else "crimson" for p in prices]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(dates, prices, color=colors, alpha=0.75, width=0.6, label="Forecast")
        ax.axhline(last_price, color="grey", linestyle="--", linewidth=1, label=f"Last close ${last_price:.2f}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_title(f"{ticker} — {len(dates)}-Day Price Forecast", fontsize=13, pad=12)
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = f"{ticker}_forecast.png"
        fig.savefig(out_path, dpi=120)
        print(f"  Chart saved to: {out_path}")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not installed; skipping chart)")
    except Exception as e:
        print(f"  (Chart generation failed: {e})")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="StockSight demo — forecast stock prices from the command line"
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=["AAPL", "TSLA", "NVDA"],
        help="Ticker symbols to forecast (default: AAPL TSLA NVDA)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train an LSTM model for each ticker before forecasting (requires TensorFlow)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of business days to forecast (default: 7, max: 30)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a bar chart of the forecast to <TICKER>_forecast.png",
    )
    args = parser.parse_args()

    days = max(1, min(args.days, 30))

    for raw_ticker in args.tickers:
        ticker = raw_ticker.upper().strip()

        print(f"\nProcessing {ticker} ...")
        print_stock_summary(ticker)

        # Optional LSTM training
        if args.train:
            print(f"\n  Training LSTM for {ticker} (this may take a few minutes)...")
            try:
                from model import train as lstm_train
                _, _, _, rmse, mae = lstm_train(ticker)
                print(f"  Training complete.  RMSE: ${rmse:.4f}   MAE: ${mae:.4f}")
            except Exception as e:
                print(f"  Training failed: {e}")
                print("  Falling back to exponential-smoothing forecast.")

        # Predict
        try:
            model_path = f"saved_models/{ticker}_model.keras"
            if os.path.exists(model_path):
                from model import predict_next_days
                result = predict_next_days(ticker, days)
            else:
                result = simple_forecast(ticker, days)
        except Exception as e:
            print(f"  Forecast failed for {ticker}: {e}")
            continue

        print_forecast(result)

        if args.plot:
            plot_forecast(result)


if __name__ == "__main__":
    main()
