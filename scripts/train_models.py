"""Refit archived recipes; optionally run a new Bayesian search without overwriting archives."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from common import BEST, ROOT, load_task, load_current_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(BEST), required=True)
    parser.add_argument("--algorithm", default="best", choices=["best", "all", "Ridge", "SVR", "RF", "ExtraTrees", "XGBoost", "CatBoost"])
    parser.add_argument("--mode", choices=["refit", "bayesian"], default="refit")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-mode", choices=["current", "archived"], default="current")
    parser.add_argument("--cv", action="store_true", help="Recompute five-fold OOF metrics before full-data fit")
    args = parser.parse_args()
    algorithms = ["Ridge", "SVR", "RF", "ExtraTrees", "XGBoost", "CatBoost"] if args.algorithm == "all" else [BEST[args.task] if args.algorithm == "best" else args.algorithm]
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    for algorithm in algorithms:
        loader = load_current_task if args.input_mode == "current" else load_task
        saved, x, y, frame, record = loader(args.task, algorithm)
        estimator = clone(saved["model"])
        if saved.get("scaler") is not None:
            estimator = Pipeline([("scaler", clone(saved["scaler"])), ("model", estimator)])
        pipeline = hasattr(estimator, "named_steps")
        prefix = "model__" if pipeline else ""
        params = estimator.get_params()
        for key in [prefix + "n_jobs", prefix + "thread_count"]:
            if key in params:
                estimator.set_params(**{key: args.threads})
        output = ROOT / "outputs" / "training" / args.task / algorithm
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == "bayesian":
            import optuna
            from train_six_regressors import suggest_params
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def objective(trial):
                candidate = clone(estimator)
                proposed = suggest_params(trial, algorithm)
                candidate.set_params(**{prefix + key: value for key, value in proposed.items()})
                return float(cross_val_score(candidate, x, y, cv=cv, scoring="r2", n_jobs=1).mean())
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=args.trials)
            estimator.set_params(**{prefix + key: value for key, value in study.best_params.items()})
            study.trials_dataframe().to_csv(output / "search_history.csv", index=False)
        if args.cv:
            predictions = cross_val_predict(estimator, x, y, cv=cv, n_jobs=1)
            metrics = {"r2_oof": r2_score(y, predictions), "mae_oof": mean_absolute_error(y, predictions), "rmse_oof": float(np.sqrt(mean_squared_error(y, predictions))), "rows": len(y), "mode": args.mode}
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            pd.DataFrame({"observed": y, "oof_predicted": predictions}).to_csv(output / "oof_predictions.csv", index=False)
        estimator.fit(x, y)
        joblib.dump({**saved, "model": estimator, "scaler": None}, output / "final_pipeline.joblib")
        (output / "run.json").write_text(json.dumps({"task": args.task, "algorithm": algorithm, "mode": args.mode, "rows": len(y), "input_mode": args.input_mode, "source": record["source_workbook"] if args.input_mode == "current" else record["training_matrix"]}, indent=2), encoding="utf-8")
        print(f"{args.task}/{algorithm}: fitted {len(y)} rows -> {output}")


if __name__ == "__main__":
    main()
