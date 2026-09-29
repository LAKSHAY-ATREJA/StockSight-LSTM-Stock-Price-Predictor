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
