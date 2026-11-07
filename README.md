# AI Crop Yield Prediction Platform

An enterprise-ready machine-learning platform served as a REST API and interactive web dashboard with **FastAPI**, featuring multi-model comparison (**Random Forest**, **XGBoost**, **LightGBM**), interactive data analytics with **Chart.js**, batch CSV predictions, containerized with **Docker** & **Docker Compose**, and tested with **pytest**.

---

## Features

- **🌾 Real Kaggle Dataset**: Trained on 28,242 historical agricultural records from the FAO across 101 countries and 10 major crops.
- **⚡ Multi-Model Support**:
  - **Random Forest**: $R^2 = 0.9876$, $\text{MAE} = 3,466\text{ hg/ha}$ (Highest Accuracy)
  - **LightGBM**: $R^2 = 0.9568$ (Ultra-fast inference)
  - **XGBoost**: $R^2 = 0.9559$
- **📊 Visual Insights**: Feature importance bar chart and dynamic historical yield trend line chart (1990–2013).
- **📁 Batch CSV Predictions**: Upload CSV spreadsheets with hundreds of fields/farms and download instant predictions in CSV format.
- **🐳 Container Ready**: Complete `Dockerfile` and `docker-compose.yml` with Linux OpenMP runtime support.

---

## Run Locally

### Option 1: 1-Click Windows Scripts (Recommended)
- **`run.bat`**: Starts the FastAPI server and interactive web UI at `http://localhost:8000`.
- **`train.bat`**: Retrains the multi-model bundle and regenerates feature importance analytics.
- **`test.bat`**: Runs the complete pytest test suite.

### Option 2: Command Line (PowerShell / CMD)
```powershell
# 1. Install dependencies
py -3.14 -m pip install -r requirements.txt
py -3.14 -m pip install -r requirements-dev.txt

# 2. Train candidate models (Random Forest, XGBoost, LightGBM)
py -3.14 train.py

# 3. Launch the API server & Web Dashboard
py -3.14 -m uvicorn app:app --reload --port 8000

# 4. Run automated test suite
py -3.14 -m pytest -v
```

---

## Web Dashboard & API Endpoints

Once running, access:
- **Interactive Web App**: [http://localhost:8000](http://localhost:8000)
  - **Tab 1: Single Prediction**: Test yield predictions with model selector and country/crop dropdowns.
  - **Tab 2: Insights & Trends**: View dynamic feature importance and historical yield curves.
  - **Tab 3: Batch CSV Prediction**: Drag-and-drop CSV upload with instant downloadable predictions.
- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health & Metrics**: [http://localhost:8000/health](http://localhost:8000/health)

---

## API Request Examples

### Single Prediction (cURL / PowerShell)
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{
    "Area": "India",
    "Item": "Rice, paddy",
    "Year": 2013,
    "average_rain_fall_mm_per_year": 1485.0,
    "pesticides_tonnes": 121.0,
    "avg_temp": 24.5,
    "model_name": "random_forest"
  }'
```

**Response**:
```json
{
  "prediction": 25366.68,
  "prediction_hg_ha": 25366.68,
  "prediction_tons_per_ha": 2.537,
  "unit": "hg/ha (10,000 hg = 1 metric ton)",
  "model_used": "random_forest",
  "latency_ms": 32.5
}
```

### Batch CSV Prediction
```bash
curl -X POST http://localhost:8000/predict-batch?model_name=xgboost \
  -F "file=@data/sample_batch.csv" \
  --output predicted_results.csv
```

---

## Docker & Docker Compose

### Using Docker Compose
```bash
docker compose up --build
```

### Using Standard Docker CLI
```bash
docker build -t crop-yield-api .
docker run -p 8000:8000 crop-yield-api
```
