# 📈 StockSight – LSTM Stock Price Predictor

A full-stack machine learning application that predicts stock prices using an LSTM neural network, with a real-time interactive dashboard built with Flask and Chart.js.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

- **LSTM Neural Network** – Stacked LSTM with dropout, trained on 2 years of historical data
- **Technical Indicators** – RSI, 7-day & 21-day moving averages, volatility, price change %
- **Real-Time Data** – Live stock data via `yfinance` API
- **Interactive Dashboard** – Beautiful dark-theme UI with price charts, RSI, and forecast table
- **Buy / Hold / Sell Signal** – Derived from 7-day predicted trend
- **REST API** – Clean endpoints for stock info, history, prediction, and model training

---

## 🧠 Model Architecture

```
Input  →  LSTM(128, return_sequences=True)
        →  Dropout(0.2)
        →  LSTM(64)
        →  Dropout(0.2)
        →  Dense(32, relu)
        →  Dense(1)          ← predicted closing price
```

**Features used:** Close, Volume, High, Low, Open, MA-7, MA-21, RSI-14, Price Change %, 10-day Volatility

**Sequence length:** 60 trading days  
**Train / Val split:** 80 / 20  
**Optimizer:** Adam | **Loss:** MSE | **Early stopping:** patience=10

---

## 📁 Project Structure

```
stock-price-predictor/
├── app.py               # Flask application & REST API
├── model.py             # LSTM model — training & inference
├── requirements.txt
├── saved_models/        # Auto-created after training
│   ├── AAPL_model.keras
│   └── AAPL_scaler.pkl
└── templates/
    └── index.html       # Full-stack dashboard (Chart.js)
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
python app.py
# → Open http://localhost:5000
```

> **No training required to run!** The app uses an exponential-smoothing fallback predictor out of the box. Train the LSTM for better accuracy (see below).

---

## 🏋️ Train the LSTM Model

**Via the command line:**
```bash
python model.py           # trains AAPL by default
```

**Via the API:**
```bash
curl -X POST http://localhost:5000/api/train/TSLA
# Returns: {"status":"success","ticker":"TSLA","rmse":3.42,"mae":2.61}
```

Trained models are saved to `saved_models/` and automatically used for subsequent predictions.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stock/<ticker>` | Company info, market cap, P/E |
| GET | `/api/history/<ticker>?period=1y` | OHLCV + indicators |
| GET | `/api/predict/<ticker>?days=7` | 7-day price forecast |
| POST | `/api/train/<ticker>` | Train LSTM & return metrics |

---

## 📊 Example Results (AAPL)

| Metric | Value |
|--------|-------|
| RMSE   | ~3.2  |
| MAE    | ~2.4  |
| Direction Accuracy | ~65% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | TensorFlow / Keras LSTM |
| Data | yfinance, Pandas, NumPy |
| Feature Eng. | scikit-learn (MinMaxScaler) |
| Backend | Flask, Flask-CORS |
| Frontend | Chart.js, Vanilla JS |
| Deployment | Gunicorn (production) |

---

## 🔮 Future Improvements

- [ ] Sentiment analysis from financial news (FinBERT)
- [ ] Portfolio-level multi-ticker comparison
- [ ] Backtesting engine with Sharpe ratio
- [ ] WebSocket live price streaming
- [ ] Docker containerisation
- [ ] Deploy to Render / Railway / Fly.io

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Predictions are not financial advice. Do not make investment decisions based solely on model output.

---

## 📄 License

MIT © Lakshay Atreja
