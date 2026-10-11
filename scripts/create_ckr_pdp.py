"""Generate standardized PDP panels for the best C, K, and R models."""

import argparse
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = PACKAGE_ROOT / 'outputs'
MINT, PEACH = "#C5E5DC", "#F29F58"
AXES_W_IN, AXES_H_IN = 4.0, 3.0
MARGINS = {"left": 0.95, "right": 0.85, "top": 0.30, "bottom": 0.78}
LW, TICK_FS, LABEL_FS = 2.4, 13, 16

COMMON = {
    "csv_pH": ("pH", "pH"),
    "csv_temperature_K": ("Tem (K)", "Tem"),
    "MOF_dose_g_L": (r"C$_{\mathrm{MOF}}$ (g/L)", "CMOF"),
    "BET_m2_g": (r"BET (m$^2$/g)", "BET"),
    "total_pore_volume_cm3_g": (r"PV (cm$^3$/g)", "PV"),
    "average_pore_diameter_nm": ("PD (nm)", "PD"),
}
CONFIGS = {
    "C": {
        "model": PACKAGE_ROOT / 'model/C/CatBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/C_model.xlsx', "target": "csv_qe",
        "direct": True, "folder": ROOT / "C_CatBoost_PDP", "prefix": "C_CatBoost", "y": "Partial dependence (mg/g)",
        "features": {"csv_ce": (r"C$_{\mathrm{e}}$ (mg/L)", "Ce"), **COMMON},
    },
    "K": {
        "model": PACKAGE_ROOT / 'model/K/XGBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/K模型_qt统一.xlsx', "target": "csv_qt",
        "direct": False, "folder": ROOT / "K_XGBoost_PDP", "prefix": "K_BestModel", "y": "Partial dependence (mg/g)",
        # This is the unified-file counterpart of the original 4020.737 mg/g outlier.
        "exclude_material": "P0186_M1", "exclude_qt": 402.073732718887,
        "features": {"csv_t": ("t (min)", "t"), **COMMON},
    },
    "R": {
        "model": PACKAGE_ROOT / 'model/R/CatBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/R_model.xlsx', "target": "csv_retention_percent",
        "direct": False, "folder": ROOT / "R_CatBoost_PDP", "prefix": "R_BestModel", "y": "Partial dependence (%)",
        "features": {"csv_cycle": ("CN", "CN"), **COMMON},
    },
}


def tokens(value):
    return set() if pd.isna(value) else {item.strip() for item in str(value).split(";") if item.strip()}


def prepare_matrix(config, feature_columns):
    data = pd.read_excel(config["data"])
    data = data[data[config["target"]].notna()].copy()
    if "exclude_qt" in config:
        outlier = data["material_id"].eq(config["exclude_material"]) & np.isclose(
            pd.to_numeric(data["csv_qt"], errors="coerce"), config["exclude_qt"], rtol=0, atol=1e-6)
        data = data.loc[~outlier].copy()
    for prefix, source_column in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                                  ("modulator", "modulator")):
        source_tokens = data[source_column].map(tokens)
        for column in [item for item in feature_columns if item.startswith(f"{prefix}__")]:
            token = column.split("__", 1)[1]
            data[column] = source_tokens.map(lambda items: int(token in items))
    matrix = data[feature_columns].copy()
    if config["direct"]:
        matrix["main"] = matrix["main"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for column in matrix.columns:
        if not (column == "main" and not config["direct"]):
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
            if config["direct"]:
                matrix[column] = matrix[column].fillna(matrix[column].median())
    return matrix


def predict(saved, direct, matrix):
    return saved["model"].predict(matrix)


def partial_dependence(saved, direct, matrix, feature, points=30):
    observed = matrix[feature].dropna().astype(float)
    low, high = np.quantile(observed, [0.01, 0.99])
    grid = np.linspace(low, high, points)
    values = []
    for value in grid:
        counterfactual = matrix.copy()
        counterfactual[feature] = value
        values.append(np.mean(predict(saved, direct, counterfactual)))
    return observed.to_numpy(), grid, np.asarray(values)


def style(axis, visible):
    for name, spine in axis.spines.items():
        spine.set_visible(name in visible)
        spine.set_linewidth(LW)
    axis.tick_params(labelsize=TICK_FS, width=1.8, length=5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def draw(observed, grid, values, x_label, y_label, output):
    width, height = MARGINS["left"] + AXES_W_IN + MARGINS["right"], MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(width, height), dpi=300)
    axis = figure.add_axes([MARGINS["left"] / width, MARGINS["bottom"] / height, AXES_W_IN / width, AXES_H_IN / height])
    distribution = axis.twinx()
    distribution.hist(observed, bins=50, range=(grid.min(), grid.max()), color=MINT,
                      edgecolor="white", linewidth=0.25, zorder=0)
    distribution.set_ylabel("Frequency", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    distribution.set_ylim(bottom=0)
    distribution.set_zorder(0)
    axis.set_zorder(1)
    axis.patch.set_visible(False)
    axis.plot(grid, values, color=PEACH, linewidth=2.8, zorder=3)
    axis.set_xlabel(x_label, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_ylabel(y_label, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_xlim(grid.min(), grid.max())
    style(axis, {"left", "right", "top", "bottom"})
    style(distribution, {"right"})
    figure.savefig(output, dpi=600, facecolor="white")
    plt.close(figure)


def create_panels(code):
    config = CONFIGS[code]
    saved = joblib.load(config["model"])
    matrix = prepare_matrix(config, saved["feature_columns"])
    config["folder"].mkdir(parents=True, exist_ok=True)
    for feature, (x_label, short_name) in config["features"].items():
        observed, grid, values = partial_dependence(saved, config["direct"], matrix, feature)
        draw(observed, grid, values, x_label, config["y"], config["folder"] / f"{config['prefix']}_PDP_{short_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("C", "K", "R"), default=("C", "K", "R"))
    args = parser.parse_args()
    for model_code in args.models:
        create_panels(model_code)
