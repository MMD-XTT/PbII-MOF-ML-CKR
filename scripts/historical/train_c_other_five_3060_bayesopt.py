import json
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

optuna.logging.set_verbosity(optuna.logging.WARNING)
SOURCE = PACKAGE_ROOT / 'data/C_model.xlsx'
ROOT = PACKAGE_ROOT / 'outputs/historical_training/C'
CV = KFold(n_splits=5, shuffle=True, random_state=2026)
N_TRIALS = 30
MODELS = ("Ridge", "SVR", "RF", "ExtraTrees", "CatBoost")

def multihot(df, column, prefix):
    values = df[column].fillna("none").astype(str)
    tokens = sorted({t.strip() for value in values for t in value.split(";") if t.strip()})
    columns = []
    for token in tokens:
        name = f"{prefix}__{token}"
        df[name] = values.map(lambda value: int(token in {t.strip() for t in value.split(";") if t.strip()}))
        columns.append(name)
    return columns

def make_pipeline(categorical, numeric, estimator):
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="none")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def propose(trial, name):
    if name == "Ridge": return {"alpha": trial.suggest_float("alpha", 0.01, 1000, log=True)}
    if name == "SVR": return {"C": trial.suggest_float("C", 0.1, 2000, log=True), "epsilon": trial.suggest_float("epsilon", 0.001, 10, log=True), "gamma": trial.suggest_float("gamma", 1e-4, 10, log=True)}
    if name in {"RF", "ExtraTrees"}:
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100), "max_depth": trial.suggest_int("max_depth", 4, 30), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8), "max_features": trial.suggest_float("max_features", 0.3, 1.0), "max_samples": trial.suggest_float("max_samples", 0.7, 1.0)}
    return {"iterations": trial.suggest_int("iterations", 300, 1000, step=100), "depth": trial.suggest_int("depth", 4, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20, log=True), "random_strength": trial.suggest_float("random_strength", 0, 2), "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5)}

def estimator(name, params):
    if name == "Ridge": return Ridge(**params)
    if name == "SVR": return SVR(kernel="rbf", **params)
    if name == "RF": return RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, **params)
    if name == "ExtraTrees": return ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=True, **params)
    return CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False, thread_count=-1, **params)

df = pd.read_excel(SOURCE)
df["csv_qe"] = pd.to_numeric(df["csv_qe"], errors="coerce")
df = df[df["csv_qe"].notna()].copy().reset_index(drop=True)
categorical = ["main"]
binary = ["O_donor", "N_donor", "S_donor", "P_donor", "carboxyl", "amino", "thiol", "sulfonic", "phosphonic"]
multi = multihot(df, "tip", "tip") + multihot(df, "metal_node_elements", "metal") + multihot(df, "Linker family", "linker") + multihot(df, "modulator", "modulator")
numeric = binary + multi + ["BET_m2_g", "total_pore_volume_cm3_g", "average_pore_diameter_nm", "csv_pH", "csv_temperature_K", "csv_ce", "MOF_dose_g_L"]
features = categorical + numeric
x = df[features].copy()
for column in numeric: x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
y = df["csv_qe"].to_numpy(float)
summary = []

for name in MODELS:
    folder = ROOT / f"{name}_全部3060行_CV5"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite existing result folder: {folder}")
    folder.mkdir(parents=True)
    def objective(trial):
        model = make_pipeline(categorical, numeric, estimator(name, propose(trial, name)))
        return float(cross_val_score(model, x, y, scoring="r2", cv=CV, n_jobs=1).mean())
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    params = study.best_params
    base = make_pipeline(categorical, numeric, estimator(name, params))
    oof = cross_val_predict(base, x, y, cv=CV, method="predict", n_jobs=1)
    folds = []
    for fold, (train, test) in enumerate(CV.split(x), 1):
        model = clone(base).fit(x.iloc[train], y[train]); pred = model.predict(x.iloc[test])
        folds.append({"fold": fold, "r2": r2_score(y[test], pred), "mae_mg_g": mean_absolute_error(y[test], pred), "rmse_mg_g": mean_squared_error(y[test], pred) ** .5, "n_validation": len(test)})
    metrics = {"model": name, "dataset": "updated C模型.xlsx", "target": "csv_qe", "target_label": "qe (mg/g)", "rows": len(df), "papers": int(df.paper_id.nunique()), "materials": int(df.material_id.nunique()), "r2_oof": r2_score(y, oof), "mae_oof_mg_g": mean_absolute_error(y, oof), "rmse_oof_mg_g": mean_squared_error(y, oof) ** .5, "r2_fold_mean": float(np.mean([item["r2"] for item in folds])), "r2_fold_sd": float(np.std([item["r2"] for item in folds], ddof=1)), "best_cv_r2": study.best_value, "n_trials": N_TRIALS, "validation": "ordinary random 5-fold KFold, shuffle=True, random_state=2026", "optimization": "Optuna TPE Bayesian optimization using the same 5-fold CV", "value_transform": "none; qe retains mg/g"}
    pd.DataFrame(study.trials_dataframe()).to_csv(folder / "bayesopt_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(folder / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"paper_id": df.paper_id, "material_id": df.material_id, "qe_observed_mg_g": y, "qe_oof_predicted_mg_g": oof, "residual": y - oof}).to_csv(folder / "oof_predictions_5fold.csv", index=False, encoding="utf-8-sig")
    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "parameters.json").write_text(json.dumps({"input_features": features, "best_model_parameters": params, "target": "csv_qe", "optimization": metrics["optimization"], "n_trials": N_TRIALS, "validation": metrics["validation"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    base.fit(x, y); joblib.dump(base, folder / "final_pipeline.joblib")
    if hasattr(base.named_steps["model"], "feature_importances_"):
        names = base.named_steps["preprocess"].get_feature_names_out()
        pd.DataFrame({"encoded_feature": names, "importance": base.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False).to_csv(folder / "feature_importance.csv", index=False, encoding="utf-8-sig")
    summary.append(metrics); print(f"{name}: OOF R2={metrics['r2_oof']:.4f}", flush=True)

summary = sorted(summary, key=lambda item: item["r2_oof"], reverse=True)
pd.DataFrame(summary).to_csv(ROOT / "其他五模型_全部3060行_BayesOpt_CV5_对比.csv", index=False, encoding="utf-8-sig")
(ROOT / "其他五模型_全部3060行_BayesOpt_CV5_对比.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
