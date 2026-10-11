"""Repository-relative access to archived model inputs and fitted estimators."""
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEST = {"C": "CatBoost", "K": "XGBoost", "R": "CatBoost"}


def predict_saved(saved, x):
    scaler = saved.get("scaler")
    return saved["model"].predict(scaler.transform(x) if scaler is not None else x)


def load_task(task, algorithm=None):
    algorithm = algorithm or BEST[task]
    folder = ROOT / "model" / task / algorithm
    record = json.loads((folder / "archive_record.json").read_text(encoding="utf-8"))
    saved = joblib.load(folder / "final_pipeline.joblib")
    frame = pd.read_csv(ROOT / record["training_matrix"])
    x = frame[saved["feature_columns"]].copy()
    if record["categorical_main"]:
        x["main"] = x["main"].fillna("none").astype(str)
    return saved, x, frame["target"].to_numpy(float), frame, record


def load_current_task(task, algorithm=None):
    saved, _, _, _, record = load_task(task, algorithm)
    names = {"C": "C_model.xlsx", "K": "K模型_qt统一.xlsx", "R": "R_model.xlsx"}
    source = ROOT / "data" / names[task]
    frame = pd.read_excel(source)
    target = record["target"]
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.loc[frame[target].notna()].copy().reset_index(drop=True)
    for prefix, column in [("metal", "metal_node_elements"), ("linker", "Linker family"), ("modulator", "modulator")]:
        tokens = frame[column].fillna("none").map(lambda value: {s.strip() for s in str(value).split(";") if s.strip()})
        for feature in saved["feature_columns"]:
            if feature.startswith(prefix + "__"):
                token = feature.split("__", 1)[1]
                frame[feature] = tokens.map(lambda value, token=token: int(token in value))
    x = frame[saved["feature_columns"]].copy()
    if not record["categorical_main"]:
        x["main"] = x.main.astype(str).str.extract(r"(\d+)")[0].astype(float)
    for column in x:
        if column == "main" and record["categorical_main"]:
            x[column] = x[column].fillna("none").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
            x[column] = x[column].fillna(0 if record["categorical_main"] else x[column].median())
    record = {**record, "source_workbook": str(source.relative_to(ROOT)), "rows": len(frame), "input_mode": "current"}
    return saved, x, frame[target].to_numpy(float), frame, record
