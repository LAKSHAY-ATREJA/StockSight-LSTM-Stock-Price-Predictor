# StockSight — LSTM Stock Price Predictor

StockSight is a machine-learning application that predicts short-term stock closing prices using a stacked LSTM neural network. It exposes a REST API and an interactive browser dashboard where you can look up any ticker, view technical indicators, and see a 7-day price forecast with a Buy / Hold / Sell signal.

The app works immediately after installation: an exponential-smoothing + linear-trend fallback predictor is active out of the box. The LSTM model can be trained on demand for any ticker via a single API call or command, after which the richer model is used automatically.

---

## Model architecture

The LSTM model takes a 60-trading-day sliding window across 10 features and predicts the next day's closing price.

```
Input shape: (60 timesteps, 10 features)

LSTM(128 units, return_sequences=True)
Dropout(0.2)
LSTM(64 units)
Dropout(0.2)
Dense(32, activation="relu")
Dense(1)  -->  predicted closing price (USD)
```

**Features (all normalized with MinMaxScaler):**

| Index | Feature          | Description                                   |
|-------|------------------|-----------------------------------------------|
| 0     | Close            | Daily closing price (prediction target)       |
| 1     | Volume           | Number of shares traded                       |
| 2     | High             | Session high                                  |
| 3     | Low              | Session low                                   |
| 4     | Open             | Opening price                                 |
| 5     | MA_7             | 7-day simple moving average of Close          |
| 6     | MA_21            | 21-day simple moving average of Close         |
| 7     | RSI              | 14-period Relative Strength Index             |
| 8     | Price_Change     | Day-over-day percentage change in Close       |
| 9     | Volatility       | 10-day rolling standard deviation of Close    |

**Training configuration:**

- Historical data: 2 years via yfinance
- Train / validation split: 80 / 20 (chronological)
- Optimizer: Adam
- Loss function: Mean Squared Error
- Early stopping: patience = 10 epochs, restores best weights
- Max epochs: 50

---

## Features

- Stacked LSTM neural network trained per ticker on 2 years of price history
- Exponential smoothing fallback predictor that works with no training step
- Live market data fetched via yfinance (no API key required)
- Technical indicators: 7-day MA, 21-day MA, RSI-14, rolling volatility
- Buy / Hold / Sell signal derived from the 7-day forecast direction
- REST API for stock metadata, price history, forecasts, and model training
- Interactive dark-theme dashboard with Chart.js: price chart with moving averages, RSI panel, forecast bar chart, and prediction table
- "Train LSTM" button in the dashboard triggers training from the browser
- Standalone demo script for command-line forecasting without a running server
- Health check endpoint for deployment platforms

---

## Requirements

- Python 3.10 or later
- Internet connection (for yfinance data)
- No GPU is required; training runs on CPU in approximately 5-15 minutes depending on hardware

---

## Installation

```bash
git clone https://github.com/LAKSHAY-ATREJA/-StockSight-LSTM-Stock-Price-Predictor.git
cd -StockSight-LSTM-Stock-Price-Predictor

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the web application

```bash
python app.py
```

Open http://localhost:5000 in a browser. Type any ticker symbol (for example AAPL, TSLA, NVDA, MSFT) and click Analyse. The dashboard loads current market data, price history, RSI, and a 7-day forecast automatically.

To run with Gunicorn (recommended for production):

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

---

## Environment variables

All variables are optional. The app runs with defaults if none are set.

Copy `.env.example` to `.env` to configure:

```
FLASK_DEBUG=false    # Set to true for development auto-reload
PORT=5000            # Port the server listens on
```

---

## Training the LSTM model

The application uses the exponential-smoothing fallback until an LSTM model is trained. Once trained, the model is saved to `saved_models/` and loaded automatically for all subsequent predictions for that ticker.

**From the command line:**

```bash
python model.py           # trains AAPL by default
python model.py TSLA      # trains a specific ticker
```

**Via the API (while the server is running):**

```bash
curl -X POST http://localhost:5000/api/train/TSLA
```

Response:

```json
{
  "status": "success",
  "ticker": "TSLA",
  "rmse": 3.42,
  "mae": 2.61
}
```

**Via the dashboard:**

Click the "Train LSTM" button in the forecast panel. Training runs server-side and a result dialog appears when complete.

---

## Running the demo (no server required)

`demo.py` runs forecasts from the terminal without starting Flask.

```bash
# Forecast AAPL, TSLA, NVDA (defaults)
python demo.py

# Forecast specific tickers
python demo.py MSFT GOOG AMZN

# Forecast 14 days ahead
python demo.py AAPL --days 14

# Train an LSTM model first, then forecast
python demo.py AAPL --train

# Save forecast charts as PNG files
python demo.py AAPL TSLA --plot
```

Example output:

```
============================================================
  AAPL   |   Last close: $213.32   |   Method: Exponential Smoothing + Linear Trend
============================================================
  Day   Date          Forecast ($)    vs Last (%)
------------------------------------------------------------
  1     2025-10-14      $214.10        +0.37%
  2     2025-10-15      $214.88        +0.73%
  3     2025-10-16      $215.62        +1.08%
  4     2025-10-17      $216.31        +1.40%
  5     2025-10-20      $217.02        +1.74%
  6     2025-10-21      $217.69        +2.05%
  7     2025-10-22      $218.41        +2.39%
------------------------------------------------------------
  Signal: BUY   |   7-day change: +2.39%
```

---

## API reference

| Method | Endpoint                        | Description                                      |
|--------|---------------------------------|--------------------------------------------------|
| GET    | /api/stock/AAPL                 | Company name, sector, market cap, P/E, 52W range |
| GET    | /api/history/AAPL?period=1y     | OHLCV data with RSI and moving averages          |
| GET    | /api/predict/AAPL?days=7        | N-day price forecast with Buy/Hold/Sell signal   |
| POST   | /api/train/AAPL                 | Train LSTM model; returns RMSE and MAE           |
| GET    | /healthz                        | Health check; returns {"status": "ok"}           |

Valid period values for /api/history: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`.

The `days` parameter for /api/predict accepts integers from 1 to 30.

---

## Project structure

```
app.py               Flask application and REST API
model.py             LSTM model: data fetching, feature engineering, training, inference
demo.py              Standalone command-line forecasting demo
templates/
    index.html       Interactive browser dashboard (Chart.js, Vanilla JS)
requirements.txt     Python dependencies
.env.example         Environment variable template
render.yaml          Render deployment configuration
Procfile             Heroku / Railway process file
saved_models/        Auto-created after training; contains .keras and .pkl files
```

---

## Deployment

### Render

The repository includes a `render.yaml` file. Connect the GitHub repository in the Render dashboard and it will detect the configuration automatically.

### Railway / Heroku

The `Procfile` is compatible with both platforms:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

Set the `PORT` environment variable as required by the platform (Railway and Heroku inject it automatically).

---

## Typical results (AAPL, trained on 2 years of data)

| Metric             | Typical value |
|--------------------|---------------|
| Validation RMSE    | ~3.2 USD      |
| Validation MAE     | ~2.4 USD      |
| Direction accuracy | ~63-67%       |

Results vary by ticker and by the market conditions covered in the training window. Stocks with steadier trends (large-cap technology names) tend to produce lower RMSE than volatile small-cap or meme stocks.

---

## Tech stack

| Component      | Technology                       |
|----------------|----------------------------------|
| ML model       | TensorFlow 2 / Keras LSTM        |
| Data pipeline  | yfinance, Pandas, NumPy          |
| Preprocessing  | scikit-learn MinMaxScaler        |
| Backend        | Flask 3, Flask-CORS, Gunicorn    |
| Frontend       | Chart.js 4, Vanilla JS           |

---

## Disclaimer

This project is for educational and demonstration purposes only. The price forecasts are statistical estimates produced by a model trained on historical data; they do not constitute financial advice. Past price patterns do not guarantee future performance. Do not make investment decisions based solely on this tool.

---

## License

MIT. Built by Lakshay Atreja.
