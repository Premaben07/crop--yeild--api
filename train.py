"""Train multi-model bundle (Random Forest, XGBoost, LightGBM) on crop yield data.
Usage: python train.py [path/to/data.csv] [target_column]"""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data" / "crop_data.csv"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "yield"

if not DATA.exists():
    generator = BASE / "make_sample_data.py"
    if not generator.exists():
        raise FileNotFoundError(f"No dataset found at {DATA} and no sample data generator exists.")
    subprocess.run([sys.executable, str(generator)], cwd=BASE, check=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset still missing after sample generation: {DATA}")

df = pd.read_csv(DATA).dropna()
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in {DATA}. Available columns: {list(df.columns)}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
numerical_cols = list(X.select_dtypes(include=["number"]).columns)

transformers = []
if categorical_cols:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
if numerical_cols:
    transformers.append(("num", "passthrough", numerical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

candidate_models = {
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    "lightgbm": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trained_pipelines = {}
metrics = {}

print("Training candidate models...")
for name, reg in candidate_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    pred = pipe.predict(X_test)
    r2 = round(r2_score(y_test, pred), 4)
    mae = round(mean_absolute_error(y_test, pred), 4)

    trained_pipelines[name] = pipe
    metrics[name] = {
        "r2": r2,
        "mae": mae,
        "train_time_sec": elapsed,
    }
    print(f" -> {name}: R2={r2}, MAE={mae}, Time={elapsed}s")

# Extract top feature importances from Random Forest
rf_pipe = trained_pipelines["random_forest"]
fitted_prep = rf_pipe.named_steps["preprocessor"]
fitted_rf = rf_pipe.named_steps["regressor"]

feature_names = []
if categorical_cols:
    cat_features = fitted_prep.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    feature_names.extend(list(cat_features))
feature_names.extend(numerical_cols)

importances = fitted_rf.feature_importances_
importance_list = sorted(
    [{"feature": f.replace("Item_", "Crop: ").replace("Area_", "Country: "), "importance": round(float(imp), 4)}
     for f, imp in zip(feature_names, importances)],
    key=lambda x: x["importance"],
    reverse=True
)[:12]

analytics = {
    "feature_importance": importance_list,
}

metadata = {
    "categorical": {col: sorted(list(df[col].dropna().unique())) for col in categorical_cols},
    "numerical": numerical_cols,
    "target": TARGET,
    "features": list(X.columns),
    "models": list(candidate_models.keys()),
}

joblib.dump(trained_pipelines, BASE / "model.joblib", compress=3)
(BASE / "features.json").write_text(json.dumps(metadata, indent=2))
(BASE / "metrics.json").write_text(json.dumps(metrics, indent=2))
(BASE / "analytics.json").write_text(json.dumps(analytics, indent=2))
print("Saved model.joblib (multi-model), features.json, metrics.json, analytics.json successfully.")
