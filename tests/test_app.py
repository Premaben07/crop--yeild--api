import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
meta = json.loads((Path(__file__).parent.parent / "features.json").read_text())
FEATURES = meta.get("features", meta) if isinstance(meta, dict) else meta


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Crop Yield" in r.text
    assert "Fetch Live Weather" in r.text


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "available_models" in data
    assert "random_forest" in data["available_models"]


def test_predict_rf_with_advisory():
    payload = {
        "Area": "India",
        "Item": "Rice, paddy",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 24.5,
        "model_name": "random_forest",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "advisory" in data
    assert "sensitivity" in data["advisory"]
    assert "recommendations" in data["advisory"]
    assert len(data["advisory"]["recommendations"]) > 0


def test_predict_xgboost():
    payload = {
        "Area": "United States",
        "Item": "Maize",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1150.0,
        "pesticides_tonnes": 3000.0,
        "avg_temp": 16.0,
        "model_name": "xgboost",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["model_used"] == "xgboost"


def test_predict_lightgbm():
    payload = {
        "Area": "France",
        "Item": "Wheat",
        "Year": 2012,
        "average_rain_fall_mm_per_year": 850.0,
        "pesticides_tonnes": 500.0,
        "avg_temp": 14.0,
        "model_name": "lightgbm",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["model_used"] == "lightgbm"


def test_predict_missing_feature():
    r = client.post("/predict", json={"features": {"Area": "India"}})
    assert r.status_code == 422


def test_analytics_importance():
    r = client.get("/analytics/importance")
    assert r.status_code == 200
    data = r.json()
    assert "feature_importance" in data
    assert len(data["feature_importance"]) > 0


def test_analytics_trends():
    r = client.get("/analytics/trends?area=India&item=Rice,%20paddy")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_sample_template():
    r = client.get("/sample-template")
    assert r.status_code == 200
    assert "Area,Item,Year" in r.text


def test_predict_batch():
    sample_csv = "Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp\nIndia,\"Rice, paddy\",2013,1083.0,2228.0,24.5\nUnited States,Maize,2013,1150.0,3500.0,15.2\n"
    files = {"file": ("test_farms.csv", io.BytesIO(sample_csv.encode("utf-8")), "text/csv")}
    r = client.post("/predict-batch?model_name=xgboost", files=files)
    assert r.status_code == 200
    assert "predicted_yield_hg_ha" in r.text
    assert "xgboost" in r.text


def test_weather_forecast():
    r = client.get("/weather/forecast?country=India")
    assert r.status_code == 200
    data = r.json()
    assert "temperature" in data
    assert "rainfall_annual_est" in data


def test_pdf_report():
    payload = {
        "Area": "India",
        "Item": "Rice, paddy",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 24.5,
    }
    r = client.post("/report/pdf", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 500
    assert r.content.startswith(b"%PDF")


def test_predict_with_financials():
    payload = {
        "Area": "India",
        "Item": "Rice, paddy",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 24.5,
        "market_price_per_ton": 400.0,
        "cost_per_hectare": 1000.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "financials" in data
    assert data["financials"]["price_per_ton"] == 400.0
    assert data["financials"]["gross_revenue"] > 0
    assert "roi_pct" in data["financials"]


def test_crop_leaderboard():
    payload = {
        "Area": "India",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 24.5,
        "model_name": "random_forest",
    }
    r = client.post("/recommend/leaderboard", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 10
    assert data[0]["rank"] == 1
    assert "predicted_yield_tons_ha" in data[0]
    assert "financials" in data[0]

