import json
import shutil
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
DATA_DIR = PACKAGE_ROOT / 'data'
OUTPUT_ROOT = PACKAGE_ROOT / 'outputs/historical_training'
N_TRIALS = 30
CV = KFold(n_splits=5, shuffle=True, random_state=2026)

SPECS = {
    "C": {"source": 'C_model.xlsx', "target": "csv_qe", "conditions": ["csv_pH", "csv_temperature_K", "csv_ce", "MOF_dose_g_L"], "label": "qe (mg/g)"},
    "K": {"source": 'K模型_qt统一.xlsx', "target": "csv_qt", "conditions": ["csv_t", "csv_pH", "csv_temperature_K", "MOF_dose_g_L"], "label": "qt (mg/g)"},
    "R": {"source": 'R_model.xlsx', "target": "csv_qr", "conditions": ["csv_cycle", "csv_pH", "csv_temperature_K", "MOF_dose_g_L"], "label": "qr (mg/g)"},
}


def add_multihot(df, column, prefix):
    tokens = sorted({token.strip() for value in df[column].fillna("none") for token in str(value).split(";") if token.strip()})
    names = []
    for token in tokens:
        name = f"{prefix}__{token}"
        df[name] = df[column].fillna("none").map(
            lambda value: int(token in {x.strip() for x in str(value).split(";") if x.strip()})
        )
        names.append(name)
    return names


def make_pipeline(main_features, numeric_features, estimator):
    preprocess = ColumnTransformer([
        ("main", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="none")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), main_features),
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric_features),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def suggest_params(trial, name):
    if name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 0.01, 1000, log=True)}
    if name == "SVR":
        return {
            "C": trial.suggest_float("C", 0.1, 2000, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 10, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 10, log=True),
        }
    if name in {"RF", "ExtraTrees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.7, 1.0),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20, log=True),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
    }


def estimator(name, parameters):
    if name == "Ridge":
        return Ridge(**parameters)
    if name == "SVR":
        return SVR(kernel="rbf", **parameters)
    if name == "RF":
        return RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, **parameters)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=True, **parameters)
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **parameters)
    return CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False, thread_count=-1, **parameters)


def train_one_dataset(code, spec):
    df = pd.read_excel(DATA_DIR / spec["source"])
    df = df[df[spec["target"]].notna()].copy().reset_index(drop=True)

    main = ["main"]
    binary = ["O_donor", "N_donor", "S_donor", "P_donor", "carboxyl", "amino", "thiol", "sulfonic", "phosphonic"]
    multi = (
        add_multihot(df, "tip", "tip")
        + add_multihot(df, "metal_node_elements", "metal")
        + add_multihot(df, "Linker family", "linker")
        + add_multihot(df, "modulator", "modulator")
    )
    pore = ["BET_m2_g", "total_pore_volume_cm3_g", "average_pore_diameter_nm"]
    numeric = binary + multi + pore + spec["conditions"]
    features = main + numeric
    x = df[features].copy()
    y = pd.to_numeric(df[spec["target"]], errors="coerce").to_numpy()
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)

    output = OUTPUT_ROOT / code
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ["Ridge", "SVR", "RF", "ExtraTrees", "XGBoost", "CatBoost"]:
        folder = output / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

        def objective(trial):
            trial_model = make_pipeline(main, numeric, estimator(name, suggest_params(trial, name)))
            return float(cross_val_score(trial_model, x, y, cv=CV, scoring="r2", n_jobs=1).mean())

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        params = study.best_params
        base = make_pipeline(main, numeric, estimator(name, params))
        oof = cross_val_predict(base, x, y, cv=CV, method="predict", n_jobs=1)
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(CV.split(x), 1):
            model = clone(base).fit(x.iloc[train_idx], y[train_idx])
            pred = model.predict(x.iloc[test_idx])
            fold_metrics.append({
                "fold": fold,
                "r2": float(r2_score(y[test_idx], pred)),
                "mae_mg_g": float(mean_absolute_error(y[test_idx], pred)),
                "rmse_mg_g": float(mean_squared_error(y[test_idx], pred) ** 0.5),
                "n_validation": int(len(test_idx)),
            })
        metrics = {
            "model": name,
            "dataset": code,
            "target": spec["target"],
            "target_label": spec["label"],
            "rows": int(len(df)),
            "materials": int(df["material_id"].nunique()),
            "papers": int(df["paper_id"].nunique()),
            "r2_oof": float(r2_score(y, oof)),
            "mae_oof_mg_g": float(mean_absolute_error(y, oof)),
            "rmse_oof_mg_g": float(mean_squared_error(y, oof) ** 0.5),
            "r2_fold_mean": float(np.mean([m["r2"] for m in fold_metrics])),
            "r2_fold_sd": float(np.std([m["r2"] for m in fold_metrics], ddof=1)),
            "best_cv_r2": float(study.best_value),
            "validation": "ordinary random 5-fold KFold, shuffle=True, random_state=2026",
            "optimization": "Optuna TPE Bayesian optimization using the same 5-fold CV",
            "n_trials": N_TRIALS,
            "numeric_transform": "none; all target and condition values retain original units",
        }
        pd.DataFrame({
            "paper_id": df["paper_id"], "material_id": df["material_id"],
            f"{code}_observed": y, f"{code}_oof_predicted": oof, "residual": y - oof,
        }).to_csv(folder / "oof_predictions_5fold.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_metrics).to_csv(folder / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(study.trials_dataframe()).to_csv(folder / "bayesopt_history.csv", index=False, encoding="utf-8-sig")
        (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "parameters.json").write_text(json.dumps({
            "categorical_feature": main,
            "numeric_features": numeric,
            "all_input_features": features,
            "target": spec["target"],
            "best_model_parameters": params,
            "optimization": metrics["optimization"],
            "validation": metrics["validation"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        base.fit(x, y)
        joblib.dump(base, folder / "final_pipeline.joblib")
        if hasattr(base.named_steps["model"], "feature_importances_"):
            encoded = base.named_steps["preprocess"].get_feature_names_out()
            pd.DataFrame({"encoded_feature": encoded, "importance": base.named_steps["model"].feature_importances_}).sort_values(
                "importance", ascending=False
            ).to_csv(folder / "feature_importance.csv", index=False, encoding="utf-8-sig")
        results.append(metrics)
        print(f"{code} {name}: OOF R2={metrics['r2_oof']:.4f}", flush=True)

    pd.DataFrame(sorted(results, key=lambda row: row["r2_oof"], reverse=True)).to_csv(
        output / "model_comparison_bayesopt_cv5.csv", index=False, encoding="utf-8-sig"
    )
    (output / "model_comparison_bayesopt_cv5.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


for code, spec in SPECS.items():
    train_one_dataset(code, spec)
