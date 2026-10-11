"""Verify archived hashes and exercise saved estimators without retraining."""
import argparse
import hashlib
import json
import numpy as np
from common import BEST, ROOT, load_task, predict_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        path = ROOT / entry["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    report = []
    algorithms = ["Ridge", "SVR", "RF", "ExtraTrees", "XGBoost", "CatBoost"]
    for task in BEST:
        for algorithm in algorithms if args.all_models else [BEST[task]]:
            saved, x, y, _, record = load_task(task, algorithm)
            prediction = predict_saved(saved, x.iloc[:8])
            assert np.isfinite(prediction).all()
            assert len(x) == record["rows"]
            report.append({"task": task, "algorithm": algorithm, "rows": len(x), "features": len(x.columns), "sample_prediction_min": float(np.min(prediction)), "sample_prediction_max": float(np.max(prediction))})
    output = ROOT / "outputs" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {len(manifest)} hashes and {len(report)} fitted models")


if __name__ == "__main__":
    main()
