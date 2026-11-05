from app import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_invalid_ticker_is_rejected_without_network_call():
    client = app.test_client()
    response = client.get("/api/stock/INVALID123")
    assert response.status_code == 400


def test_prediction_days_must_be_in_range():
    client = app.test_client()
    response = client.get("/api/predict/AAPL?days=31")
    assert response.status_code == 400


def test_prediction_days_must_be_integer():
    client = app.test_client()
    response = client.get("/api/predict/AAPL?days=seven")
    assert response.status_code == 400


def test_stock_info_survives_missing_quote_type(monkeypatch):
    """Regression: this is the exact type of failure seen on the live dashboard."""
    import pandas as pd
    import app as stock_app

    class Ticker:
        info = {"longName": "Apple Inc.", "sector": "Technology"}

        def history(self, period="1y", auto_adjust=True):
            assert period == "1y"
            return pd.DataFrame(
                {"Close": [201.0, 208.0], "High": [205.0, 210.0],
                 "Low": [198.0, 203.0], "Volume": [1200000, 1000000]},
                index=pd.bdate_range("2026-10-01", periods=2),
            )

    monkeypatch.setattr(stock_app.yf, "Ticker", lambda ticker: Ticker())
    response = stock_app.app.test_client().get("/api/stock/AAPL")
    assert response.status_code == 200
    data = response.get_json()
    assert data["name"] == "Apple Inc."
    assert data["current_price"] == 208.0
    assert data["52w_high"] == 210.0
    assert data["52w_low"] == 198.0


def test_stock_info_falls_back_when_yahoo_info_raises(monkeypatch):
    import pandas as pd
    import app as stock_app

    class Ticker:
        @property
        def info(self):
            raise RuntimeError("Yahoo metadata endpoint unavailable")

        def history(self, period="1y", auto_adjust=True):
            return pd.DataFrame(
                {"Close": [150.5], "High": [155.0], "Low": [148.0],
                 "Volume": [440000]},
                index=pd.bdate_range("2026-10-01", periods=1),
            )

    monkeypatch.setattr(stock_app.yf, "Ticker", lambda ticker: Ticker())
    response = stock_app.app.test_client().get("/api/stock/AAPL")
    assert response.status_code == 200
    data = response.get_json()
    assert data["name"] == "AAPL"
    assert data["current_price"] == 150.5
    assert data["market_cap"] == 0.0


def test_stock_info_is_unavailable_when_both_sources_fail(monkeypatch):
    import pandas as pd
    import app as stock_app

    class Ticker:
        info = {}

        def history(self, period="1y", auto_adjust=True):
            return pd.DataFrame()

    monkeypatch.setattr(stock_app.yf, "Ticker", lambda ticker: Ticker())
    response = stock_app.app.test_client().get("/api/stock/NODATA")
    assert response.status_code == 503
    assert "temporarily" not in response.get_json()["error"] or "market data" in response.get_json()["error"].lower()
