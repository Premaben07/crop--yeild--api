FROM python:3.11-slim

# Install OpenMP runtime required by XGBoost & LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, multi-model artifacts, analytics, and data
COPY app.py model.joblib features.json metrics.json analytics.json ./
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
