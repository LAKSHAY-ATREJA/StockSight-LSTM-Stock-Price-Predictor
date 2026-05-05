# StockSight — LSTM Stock Price Predictor

A full-stack machine learning application that predicts stock prices using a stacked LSTM neural network, with an interactive dashboard built with Flask and Chart.js. No training is required to run the app — an exponential-smoothing fallback predictor is used out of the box, and the LSTM model can be trained on demand per ticker via the API.

## Model architecture

```
Input (60 trading days x 10 features)
    LSTM(128, return_sequences=True) + Dropout(0.2)
    LSTM(64) + Dropout(0.2)
    Dense(32, relu)
    Dense(1)  ->  predicted closing price
```

Features: Close, Volume, High, Low, Open, 7-day MA, 21-day MA, RSI-14, price change %, 10-day rolling volatility.

Train/validation split: 80/20. Optimiser: Adam. Loss: MSE. Early stopping with patience=10.

## Features

- LSTM neural network trained on 2 years of historical price data
- Exponential smoothing fallback predictor (works immediately with no training)
- Live stock data via the yfinance API
- Technical indicators: RSI, moving averages, volatility
- Buy / Hold / Sell signal derived from the 7-day forecast trend
- REST API for stock info, history, prediction, and model training
- Interactive dark-theme dashboard with Chart.js price charts, RSI panel, and forecast table

## Requirements

- Python 3.10 or later

## Local setup

```bash
git clone https://github.com/LAKSHAY-ATREJA/-StockSight-LSTM-Stock-Price-Predictor.git
cd -StockSight-LSTM-Stock-Price-Predictor

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Open http://localhost:5000 in your browser. Enter any ticker (e.g. AAPL, TSLA, NVDA) to see the dashboard.

## Training the LSTM model

The app uses the exponential-smoothing fallback until you train the LSTM. Training fetches 2 years of data, trains the model, and saves it to `saved_models/`.

From the command line:

```bash
python model.py           # trains AAPL by default
```

Via the API (while the server is running):

```bash
curl -X POST http://localhost:5000/api/train/TSLA
# Returns: {"status": "success", "ticker": "TSLA", "rmse": 3.42, "mae": 2.61}
```

Once trained, subsequent predictions for that ticker automatically use the LSTM model.

## API reference

| Method | Endpoint                          | Description                                    |
|--------|-----------------------------------|------------------------------------------------|
| GET    | /api/stock/AAPL                   | Company info, market cap, P/E ratio            |
| GET    | /api/history/AAPL?period=1y       | OHLCV data with RSI and moving averages        |
| GET    | /api/predict/AAPL?days=7          | 7-day price forecast with Buy/Hold/Sell signal |
| POST   | /api/train/AAPL                   | Train LSTM model, returns RMSE and MAE         |

## Project structure

```
app.py               Flask application and REST API
model.py             LSTM model: training, feature engineering, inference
templates/
    index.html       Interactive dashboard (Chart.js, Vanilla JS)
requirements.txt     Python dependencies
saved_models/        Auto-created after training (AAPL_model.keras, AAPL_scaler.pkl)
```

## Typical results (AAPL)

| Metric             | Value     |
|--------------------|-----------|
| RMSE               | ~3.2 USD  |
| MAE                | ~2.4 USD  |
| Direction accuracy | ~65%      |

## Tech stack

| Component     | Technology                   |
|---------------|------------------------------|
| ML model      | TensorFlow / Keras LSTM      |
| Data          | yfinance, Pandas, NumPy      |
| Feature eng.  | scikit-learn MinMaxScaler    |
| Backend       | Flask, Flask-CORS, Gunicorn  |
| Frontend      | Chart.js, Vanilla JS         |

## Disclaimer

This project is for educational purposes only. Predictions are statistical estimates, not financial advice. Do not make investment decisions based solely on this tool.

## License

MIT. Built by Lakshay Atreja.
