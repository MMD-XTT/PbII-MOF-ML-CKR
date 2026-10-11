"""Compute native tree SHAP contributions, grouped importance, and beeswarm plots."""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from common import BEST, ROOT, load_task, load_current_task


def group_for(name):
    if name.startswith("main__") or name == "main":
        return "M"
    name = name.removeprefix("numeric__").removeprefix("missingindicator_")
    for prefix, group in [("metal__", "MN"), ("linker__", "LF"), ("modulator__", "Mod")]:
        if name.startswith(prefix):
            return group
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["C", "K", "R", "all"], default="all")
    parser.add_argument("--sample", type=int, default=0, help="0 uses the complete selected input population")
    parser.add_argument("--input-mode", choices=["current", "archived"], default="current")
    args = parser.parse_args()
    for task in BEST if args.task == "all" else [args.task]:
        loader = load_current_task if args.input_mode == "current" else load_task
        saved, x, _, _, _ = loader(task)
        if args.sample > 0 and len(x) > args.sample:
            x = x.sample(args.sample, random_state=2026)
        model = saved["model"]
        if hasattr(model, "named_steps"):
            pre = model.named_steps["preprocess"]
            matrix = pre.transform(x)
            matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            names = list(pre.get_feature_names_out())
            estimator = model.named_steps["model"]
        else:
            matrix, names, estimator = x.to_numpy(), list(x), model
        if task == "K":
            import xgboost as xgb
            values = estimator.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)[:, :-1]
        else:
            values = estimator.get_feature_importance(Pool(matrix, feature_names=names), type="ShapValues")[:, :-1]
        output = ROOT / "outputs" / "shap" / task
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps({"task": task, "rows_analyzed": len(x), "requested_sample": args.sample, "population": args.input_mode}, indent=2), encoding="utf-8")
        table = pd.DataFrame({"feature": names, "group": [group_for(name) for name in names], "mean_abs_shap": np.abs(values).mean(axis=0)})
        grouped = table.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
        grouped["percentage"] = 100 * grouped.mean_abs_shap / grouped.mean_abs_shap.sum()
        table.to_csv(output / "feature_importance.csv", index=False)
        grouped.to_csv(output / "grouped_importance.csv", index=False)
        np.savez_compressed(output / "shap_values.npz", values=values, features=np.asarray(names), matrix=matrix)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.barh(grouped.group.iloc[::-1], grouped.mean_abs_shap.iloc[::-1], color="#9BD8C4", edgecolor="black")
        ax.set_xlabel("mean |SHAP value|")
        fig.tight_layout()
        fig.savefig(output / "grouped_shap.png", dpi=300)
        plt.close(fig)
        import shap
        shap.summary_plot(values, matrix, feature_names=names, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(output / "beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"{task}: SHAP completed for {len(x)} {args.input_mode} rows")


if __name__ == "__main__":
    main()
