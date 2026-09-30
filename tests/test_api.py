import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "threshold" in data
    assert data["threshold"] == 0.99


def test_predict_normal_transaction():
    sample = {f"V{i}": 0.0 for i in range(1, 29)}
    sample["Time"] = 0.0
    sample["Amount"] = 149.62

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == 0
    assert data["label"] == "LEGITIMATE"
    assert data["fraud_probability"] < 0.99


def test_predict_fraud_transaction():
    sample = {f"V{i}": 0.0 for i in range(1, 29)}
    sample["Time"] = 406.0
    sample["V4"] = 4.5
    sample["V11"] = 3.8
    sample["V12"] = -6.0
    sample["V14"] = -7.0
    sample["Amount"] = 500.0

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["label"] == "FRAUD"
    assert data["fraud_probability"] >= 0.99


def test_predict_missing_columns():
    # Only send partial data
    sample = {"Amount": 50.0}
    response = client.post("/predict", json=sample)
    assert response.status_code == 422
