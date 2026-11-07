import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
FEATURES_PATH = BASE / "features.json"
METRICS_PATH = BASE / "metrics.json"
ANALYTICS_PATH = BASE / "analytics.json"
DATA_PATH = BASE / "data" / "crop_data.csv"
SAMPLE_BATCH_PATH = BASE / "data" / "sample_batch.csv"
TARGET = "yield"

DEFAULT_PRICES = {
    "Potatoes": 220.0,
    "Rice, paddy": 380.0,
    "Wheat": 260.0,
    "Maize": 210.0,
    "Soybeans": 450.0,
    "Cassava": 180.0,
    "Sweet potatoes": 240.0,
    "Yams": 310.0,
    "Sorghum": 220.0,
    "Plantains and others": 350.0,
}

DEFAULT_COSTS = {
    "Potatoes": 2400.0,
    "Rice, paddy": 950.0,
    "Wheat": 750.0,
    "Maize": 850.0,
    "Soybeans": 800.0,
    "Cassava": 700.0,
    "Sweet potatoes": 1100.0,
    "Yams": 1200.0,
    "Sorghum": 650.0,
    "Plantains and others": 1100.0,
}


def ensure_model_artifacts():
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        raw_meta = json.loads(FEATURES_PATH.read_text())
        meta = raw_meta if isinstance(raw_meta, dict) else {
            "features": raw_meta,
            "categorical": {},
            "numerical": raw_meta,
            "target": TARGET,
            "models": ["random_forest"],
        }
        loaded = joblib.load(MODEL_PATH)
        models = loaded if isinstance(loaded, dict) else {"random_forest": loaded}
        return models, meta

    import subprocess
    subprocess.run([sys.executable, str(BASE / "train.py")], cwd=BASE, check=True)
    raw_meta = json.loads(FEATURES_PATH.read_text())
    meta = raw_meta if isinstance(raw_meta, dict) else {"features": raw_meta, "categorical": {}, "numerical": raw_meta}
    loaded = joblib.load(MODEL_PATH)
    models = loaded if isinstance(loaded, dict) else {"random_forest": loaded}
    return models, meta


MODELS, METADATA = ensure_model_artifacts()
FEATURES = METADATA.get("features", [])
CATEGORICAL = METADATA.get("categorical", {})
NUMERICAL = METADATA.get("numerical", [])
AVAILABLE_MODELS = METADATA.get("models", list(MODELS.keys()))
METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
ANALYTICS = json.loads(ANALYTICS_PATH.read_text()) if ANALYTICS_PATH.exists() else {"feature_importance": []}

DF_HISTORICAL = pd.read_csv(DATA_PATH).dropna() if DATA_PATH.exists() else pd.DataFrame()

app = FastAPI(title="AI Crop Yield & Agronomic Intelligence Platform")


class PredictRequest(BaseModel):
    Area: Optional[str] = "India"
    Item: Optional[str] = "Rice, paddy"
    Year: Optional[int] = 2013
    average_rain_fall_mm_per_year: Optional[float] = 1485.0
    pesticides_tonnes: Optional[float] = 121.0
    avg_temp: Optional[float] = 24.5
    model_name: Optional[str] = "random_forest"
    market_price_per_ton: Optional[float] = None
    cost_per_hectare: Optional[float] = None
    features: Optional[dict[str, Any]] = None


class LeaderboardRequest(BaseModel):
    Area: Optional[str] = "India"
    Year: Optional[int] = 2013
    average_rain_fall_mm_per_year: Optional[float] = 1485.0
    pesticides_tonnes: Optional[float] = 121.0
    avg_temp: Optional[float] = 24.5
    model_name: Optional[str] = "random_forest"


def compute_financials(item: str, yield_tons: float, custom_price: Optional[float] = None, custom_cost: Optional[float] = None) -> dict:
    price_per_ton = custom_price if (custom_price is not None and custom_price > 0) else DEFAULT_PRICES.get(item, 250.0)
    cost_per_ha = custom_cost if (custom_cost is not None and custom_cost > 0) else DEFAULT_COSTS.get(item, 850.0)

    gross_revenue = round(yield_tons * price_per_ton, 2)
    net_profit = round(gross_revenue - cost_per_ha, 2)
    roi_pct = round((net_profit / cost_per_ha) * 100, 1) if cost_per_ha > 0 else 0.0
    break_even = round(cost_per_ha / price_per_ton, 2) if price_per_ton > 0 else 0.0

    return {
        "price_per_ton": price_per_ton,
        "cost_per_ha": cost_per_ha,
        "gross_revenue": gross_revenue,
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "break_even_tons_ha": break_even,
    }


def compute_advisory(selected_model, input_dict: dict, base_yield: float) -> dict:
    scenarios = [
        ("water_plus", "Rainfall +150 mm", {"average_rain_fall_mm_per_year": input_dict["average_rain_fall_mm_per_year"] + 150}),
        ("water_minus", "Rainfall -150 mm", {"average_rain_fall_mm_per_year": max(100.0, input_dict["average_rain_fall_mm_per_year"] - 150)}),
        ("temp_plus", "Temperature +2.0 °C", {"avg_temp": input_dict["avg_temp"] + 2.0}),
        ("pest_plus", "Pesticides +20%", {"pesticides_tonnes": input_dict["pesticides_tonnes"] * 1.2}),
    ]

    sensitivity = []
    for code, label, change in scenarios:
        row_dict = {**input_dict, **change}
        test_row = pd.DataFrame([{f: row_dict[f] for f in FEATURES}], columns=FEATURES)
        try:
            pred_val = float(selected_model.predict(test_row)[0])
            pct_change = round(((pred_val - base_yield) / base_yield) * 100, 2)
            sensitivity.append({
                "scenario": label,
                "predicted_yield_tons": round(pred_val / 10000.0, 3),
                "pct_change": pct_change,
            })
        except Exception:
            pass

    recommendations = []
    water_change = next((s["pct_change"] for s in sensitivity if "Rainfall +150" in s["scenario"]), 0)
    if water_change > 1.5:
        recommendations.append(f"💧 Irrigation Opportunity: Increasing effective water supply by 150 mm is projected to increase yield by +{water_change}%. Consider supplemental drip or furrow irrigation.")
    elif water_change < -1.5:
        recommendations.append(f"⚠️ Drainage Advisory: Soil saturation risk detected. Additional rainfall may diminish yields by {abs(water_change)}%. Ensure active plot drainage.")
    else:
        recommendations.append("💧 Hydration Status: Current annual rainfall appears near optimal balance for this crop type.")

    temp_change = next((s["pct_change"] for s in sensitivity if "Temperature +2" in s["scenario"]), 0)
    if temp_change < -1.0:
        recommendations.append(f"🌡️ Heat Vulnerability: A 2°C temperature rise reduces projected yield by {abs(temp_change)}%. Consider heat-tolerant cultivars or adjust planting date.")
    else:
        recommendations.append("🌡️ Thermal Suitability: The crop demonstrates strong thermal resilience under expected local temperature variance.")

    pest_change = next((s["pct_change"] for s in sensitivity if "Pesticides +20%" in s["scenario"]), 0)
    if pest_change < 1.0:
        recommendations.append("🌿 Eco-Efficiency: Increasing pesticide application yields negligible benefit. Maintain current dosages to minimize input costs and runoff.")
    else:
        recommendations.append(f"🛡️ Pest Protection: Targeted pest management could safeguard an estimated +{pest_change}% yield gain.")

    return {
        "sensitivity": sensitivity,
        "recommendations": recommendations,
    }


def generate_pdf_report(input_dict: dict, result: dict, advisory: dict, financials: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1b5e20"),
        alignment=0,
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2e7d32"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#263238"),
    )
    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#37474f"),
        leftIndent=15,
        spaceAfter=3,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("🌾 Crop Yield, Advisory & Financial Intelligence Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Model: {result.get('model_used', 'Random Forest').replace('_', ' ').title()}", body_style))
    story.append(Spacer(1, 10))

    # Parameters & Prediction Table
    story.append(Paragraph("1. Farm Parameters & Yield Prediction", h2_style))
    param_data = [
        ["Country / Area", input_dict.get("Area", "N/A"), "Target Crop", input_dict.get("Item", "N/A")],
        ["Year", str(input_dict.get("Year", "N/A")), "Avg Temperature", f"{input_dict.get('avg_temp', 'N/A')} °C"],
        ["Avg Rainfall", f"{input_dict.get('average_rain_fall_mm_per_year', 'N/A')} mm/yr", "Pesticide Usage", f"{input_dict.get('pesticides_tonnes', 'N/A')} tonnes"],
        ["Predicted Yield (Metric)", f"{result.get('prediction_tons_per_ha', 'N/A')} tons / ha", "Predicted Yield (hg/ha)", f"{result.get('prediction_hg_ha', 'N/A'):,} hg / ha"],
    ]
    t1 = Table(param_data, colWidths=[130, 140, 130, 140])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fbe7")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#c8e6c9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#263238")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Commercial Profitability Table
    story.append(Paragraph("2. Commercial Farm Profitability Projections", h2_style))
    fin_data = [
        ["Metric Market Price", f"${financials.get('price_per_ton', 0):,.2f} / ton", "Estimated Production Cost", f"${financials.get('cost_per_ha', 0):,.2f} / ha"],
        ["Projected Gross Revenue", f"${financials.get('gross_revenue', 0):,.2f} / ha", "Estimated Net Profit", f"${financials.get('net_profit', 0):,.2f} / ha"],
        ["Return on Investment (ROI)", f"{financials.get('roi_pct', 0)}%", "Break-Even Yield Threshold", f"{financials.get('break_even_tons_ha', 0)} tons / ha"],
    ]
    t_fin = Table(fin_data, colWidths=[130, 140, 130, 140])
    t_fin.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (2, 1), (3, 1), colors.HexColor("#dcedc8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1b5e20")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (2, 1), (3, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_fin)
    story.append(Spacer(1, 10))

    # Sensitivity / What-If Analysis
    story.append(Paragraph("3. What-If Sensitivity Analysis", h2_style))
    sens_rows = [["Scenario", "Projected Yield (tons/ha)", "Expected Impact (%)"]]
    for item in advisory.get("sensitivity", []):
        pct = item.get("pct_change", 0)
        sign = "+" if pct > 0 else ""
        sens_rows.append([item["scenario"], f"{item['predicted_yield_tons']} t/ha", f"{sign}{pct}%"])

    t2 = Table(sens_rows, colWidths=[200, 170, 170])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # Agronomic Recommendations
    story.append(Paragraph("4. Actionable Agronomic Recommendations", h2_style))
    for rec in advisory.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", bullet_style))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: Projections are derived from ML models trained on 28,000+ FAO historical records. Commodity prices and production costs are baselines subject to localized market dynamics.", ParagraphStyle("Footnote", parent=styles["Italic"], fontSize=8, textColor=colors.HexColor("#78909c"))))

    doc.build(story)
    return buffer.getvalue()


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = BASE / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found")

    html = template_path.read_text(encoding="utf-8")

    countries = CATEGORICAL.get("Area", [])
    crops = CATEGORICAL.get("Item", [])

    badges = "".join(
        f'<span class="badge">{m.replace("_", " ").upper()} R² {round(METRICS.get(m, {}).get("r2", 0.95)*100, 1)}%</span>'
        for m in AVAILABLE_MODELS
    )
    country_options = "".join(
        f'<option value="{c}" {"selected" if c=="India" else ""}>{c}</option>'
        for c in countries
    )
    crop_options = "".join(
        f'<option value="{crop}" {"selected" if "Rice" in crop else ""}>{crop}</option>'
        for crop in crops
    )
    model_options = "".join(
        f'<option value="{m}" {"selected" if m=="random_forest" else ""}>{m.replace("_", " ").title()} (R²: {METRICS.get(m, {}).get("r2", "N/A")})</option>'
        for m in AVAILABLE_MODELS
    )

    html = html.replace("<!--MODEL_BADGES-->", badges)
    html = html.replace("<!--COUNTRY_OPTIONS-->", country_options)
    html = html.replace("<!--CROP_OPTIONS-->", crop_options)
    html = html.replace("<!--MODEL_OPTIONS-->", model_options)
    html = html.replace("<!--IMPORTANCE_JSON-->", json.dumps(ANALYTICS.get("feature_importance", [])))
    html = html.replace("<!--DEFAULT_PRICES_JSON-->", json.dumps(DEFAULT_PRICES))
    html = html.replace("<!--DEFAULT_COSTS_JSON-->", json.dumps(DEFAULT_COSTS))

    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "Kaggle FAO Crop Yield Dataset (28,242 rows)",
        "available_models": AVAILABLE_MODELS,
        "metrics": METRICS,
        "features": FEATURES,
    }


@app.get("/weather/forecast")
def get_live_weather(country: str = Query("India", description="Country/Area name")):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={country}&count=1"
        geo_resp = httpx.get(geo_url, timeout=4.0)
        geo_data = geo_resp.json()

        if "results" in geo_data and geo_data["results"]:
            res = geo_data["results"][0]
            lat = res["latitude"]
            lon = res["longitude"]
            city = res.get("name", country)

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation&daily=temperature_2m_mean,precipitation_sum&timezone=auto"
            w_resp = httpx.get(weather_url, timeout=4.0)
            w_data = w_resp.json()

            current_temp = w_data.get("current", {}).get("temperature_2m", 25.0)
            daily_precip = w_data.get("daily", {}).get("precipitation_sum", [3.0])
            avg_daily = sum(daily_precip) / max(1, len(daily_precip))
            est_annual_rain = max(300.0, round(avg_daily * 365, 1))

            return {
                "country": country,
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "temperature": round(current_temp, 1),
                "rainfall_annual_est": est_annual_rain,
            }
    except Exception:
        pass

    defaults = {
        "India": (26.5, 1100.0),
        "United States": (16.0, 950.0),
        "France": (14.0, 850.0),
        "Brazil": (25.0, 1600.0),
    }
    temp, rain = defaults.get(country, (22.0, 1000.0))
    return {
        "country": country,
        "city": country,
        "latitude": 0.0,
        "longitude": 0.0,
        "temperature": temp,
        "rainfall_annual_est": rain,
        "source": "fallback",
    }


@app.get("/analytics/importance")
def get_feature_importance():
    return ANALYTICS


@app.get("/analytics/trends")
def get_crop_trends(
    area: str = Query("India", description="Country/Area name"),
    item: str = Query("Rice, paddy", description="Crop item name")
):
    if DF_HISTORICAL.empty:
        return []

    filtered = DF_HISTORICAL[(DF_HISTORICAL["Area"] == area) & (DF_HISTORICAL["Item"] == item)]
    if filtered.empty:
        return []

    grouped = filtered.groupby("Year", as_index=False).agg({
        "yield": "mean",
        "average_rain_fall_mm_per_year": "mean",
        "avg_temp": "mean",
        "pesticides_tonnes": "mean",
    }).sort_values("Year")
    grouped["yield"] = grouped["yield"].round(2)
    return grouped.to_dict(orient="records")


@app.get("/sample-template")
def get_sample_template():
    if not SAMPLE_BATCH_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample template not found")
    content = SAMPLE_BATCH_PATH.read_text()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crop_yield_batch_template.csv"}
    )


@app.post("/predict")
def predict(req: PredictRequest):
    data = req.model_dump()
    model_name = data.get("model_name") or "random_forest"
    if model_name not in MODELS:
        model_name = "random_forest"

    selected_model = MODELS[model_name]

    if data.get("features"):
        input_data = data["features"]
    else:
        input_data = {k: v for k, v in data.items() if k not in ("features", "model_name", "market_price_per_ton", "cost_per_hectare")}

    missing = [f for f in FEATURES if f not in input_data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    row = pd.DataFrame([{f: input_data[f] for f in FEATURES}], columns=FEATURES)

    t0 = time.time()
    try:
        raw_pred = float(selected_model.predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    latency_ms = round((time.time() - t0) * 1000, 2)

    tons_per_ha = round(raw_pred / 10000.0, 3)

    advisory = compute_advisory(selected_model, input_data, raw_pred)
    financials = compute_financials(
        input_data.get("Item", "Rice, paddy"),
        tons_per_ha,
        data.get("market_price_per_ton"),
        data.get("cost_per_hectare"),
    )

    return {
        "prediction": round(raw_pred, 2),
        "prediction_hg_ha": round(raw_pred, 2),
        "prediction_tons_per_ha": tons_per_ha,
        "unit": "hg/ha (10,000 hg = 1 metric ton)",
        "model_used": model_name,
        "latency_ms": latency_ms,
        "advisory": advisory,
        "financials": financials,
    }


@app.post("/recommend/leaderboard")
def recommend_leaderboard(req: LeaderboardRequest):
    """Simulate and rank all 10 major crops simultaneously for the given land and climate."""
    model_name = req.model_name or "random_forest"
    if model_name not in MODELS:
        model_name = "random_forest"
    selected_model = MODELS[model_name]

    crops = CATEGORICAL.get("Item", [])
    if not crops:
        return []

    # Build vectorized DataFrame for all crops
    rows = []
    for crop in crops:
        rows.append({
            "Area": req.Area,
            "Item": crop,
            "Year": req.Year,
            "average_rain_fall_mm_per_year": req.average_rain_fall_mm_per_year,
            "pesticides_tonnes": req.pesticides_tonnes,
            "avg_temp": req.avg_temp,
        })
    df_eval = pd.DataFrame(rows)[FEATURES]

    try:
        preds = selected_model.predict(df_eval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Leaderboard evaluation failed: {e}")

    results = []
    for crop, raw_pred in zip(crops, preds):
        tons_ha = round(float(raw_pred) / 10000.0, 3)
        fin = compute_financials(crop, tons_ha)
        results.append({
            "crop": crop,
            "predicted_yield_hg_ha": round(float(raw_pred), 2),
            "predicted_yield_tons_ha": tons_ha,
            "financials": fin,
        })

    # Sort descending by yield
    results.sort(key=lambda x: x["predicted_yield_tons_ha"], reverse=True)
    for idx, item in enumerate(results):
        item["rank"] = idx + 1

    return results


@app.post("/report/pdf")
def create_pdf_report(req: PredictRequest):
    pred_res = predict(req)
    data = req.model_dump()
    input_data = data.get("features") if data.get("features") else {k: v for k, v in data.items() if k not in ("features", "model_name")}

    pdf_bytes = generate_pdf_report(
        input_data,
        pred_res,
        pred_res.get("advisory", {}),
        pred_res.get("financials", {}),
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Crop_Yield_Report_{input_data.get('Area', 'Farm')}.pdf"}
    )


@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_name: str = Query("random_forest")
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    if model_name not in MODELS:
        model_name = "random_forest"
    selected_model = MODELS[model_name]

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing required columns: {missing}")

    X = df[FEATURES]
    predictions = selected_model.predict(X)
    df["predicted_yield_hg_ha"] = [round(float(p), 2) for p in predictions]
    df["predicted_yield_tons_ha"] = [round(float(p) / 10000.0, 3) for p in predictions]
    df["model_used"] = model_name

    output_csv = df.to_csv(index=False)
    return Response(
        content=output_csv,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predicted_{file.filename}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
