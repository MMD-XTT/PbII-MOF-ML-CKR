"""Create two standardized 2D PDP panels for the C CatBoost model."""

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

MODEL_PATH = PACKAGE_ROOT / 'model/C/CatBoost/final_pipeline.joblib'
DATA_PATH = PACKAGE_ROOT / 'data/C_model.xlsx'
OUTPUT_DIR = PACKAGE_ROOT / 'outputs/C_CatBoost_PDP'
KR_CONFIGS = {
    "K": {
        "model": PACKAGE_ROOT / 'model/K/XGBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/K模型_qt统一.xlsx',
        "target": "csv_qt", "folder": PACKAGE_ROOT / 'outputs/K_XGBoost_PDP',
        "prefix": "K_XGBoost", "unit": "mg/g", "exclude_material": "P0186_M1", "exclude_qt": 402.073732718887,
    },
    "R": {
        "model": PACKAGE_ROOT / 'model/R/CatBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/R_model.xlsx',
        "target": "csv_retention_percent", "folder": PACKAGE_ROOT / 'outputs/R_CatBoost_PDP',
        "prefix": "R_CatBoost", "unit": "%",
    },
}
AXES_W_IN, AXES_H_IN = 4.0, 3.0
MARGINS = {"left": 0.95, "right": 1.05, "top": 0.30, "bottom": 0.78}
LW, TICK_FS, LABEL_FS = 2.4, 13, 16
PDP_PAIRS = (
    ("total_pore_volume_cm3_g", "BET_m2_g", r"BET (m$^2$/g)", "PV_BET"),
    ("total_pore_volume_cm3_g", "average_pore_diameter_nm", "PD (nm)", "PV_PD"),
)


def tokens(value):
    return set() if pd.isna(value) else {item.strip() for item in str(value).split(";") if item.strip()}


def build_matrix(feature_columns):
    data = pd.read_excel(DATA_PATH)
    data = data[data["csv_qe"].notna()].copy()
    data["main"] = data["main"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for prefix, source_column in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                                  ("modulator", "modulator")):
        source_tokens = data[source_column].map(tokens)
        for column in [item for item in feature_columns if item.startswith(f"{prefix}__")]:
            token = column.split("__", 1)[1]
            data[column] = source_tokens.map(lambda items: int(token in items))
    matrix = data[feature_columns].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        matrix[column] = matrix[column].fillna(matrix[column].median())
    return matrix


def build_pipeline_matrix(config, feature_columns):
    """Build the raw feature frame expected by the saved K/R sklearn pipelines."""
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
    for column in matrix.columns:
        if column != "main":
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def calculate_2d_pdp(model, matrix, x_feature, y_feature, points=25):
    x_values = matrix[x_feature].to_numpy(float)
    y_values = matrix[y_feature].to_numpy(float)
    x_grid = np.linspace(*np.quantile(x_values, [0.01, 0.99]), points)
    y_grid = np.linspace(*np.quantile(y_values, [0.01, 0.99]), points)
    surface = np.empty((points, points))
    for row, y_value in enumerate(y_grid):
        for column, x_value in enumerate(x_grid):
            counterfactual = matrix.copy()
            counterfactual[x_feature] = x_value
            counterfactual[y_feature] = y_value
            surface[row, column] = np.mean(model.predict(counterfactual))
    return x_grid, y_grid, surface


def style_axis(axis):
    for spine in axis.spines.values():
        spine.set_linewidth(LW)
    axis.tick_params(labelsize=TICK_FS, width=1.8, length=5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def draw_surface(x_grid, y_grid, surface, x_label, y_label, output_path, unit="mg/g"):
    figure_width = MARGINS["left"] + AXES_W_IN + MARGINS["right"]
    figure_height = MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    axis = figure.add_axes([MARGINS["left"] / figure_width, MARGINS["bottom"] / figure_height,
                            AXES_W_IN / figure_width, AXES_H_IN / figure_height])
    filled = axis.contourf(x_grid, y_grid, surface, levels=18, cmap="YlGnBu")
    axis.contour(x_grid, y_grid, surface, levels=9, colors="#4F4F4F", linewidths=0.75)
    axis.set_xlabel(x_label, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_ylabel(y_label, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    style_axis(axis)

    color_axis = figure.add_axes([(MARGINS["left"] + AXES_W_IN + 0.24) / figure_width,
                                  MARGINS["bottom"] / figure_height, 0.13 / figure_width,
                                  AXES_H_IN / figure_height])
    colorbar = figure.colorbar(filled, cax=color_axis)
    colorbar.set_label(f"Partial dependence ({unit})", fontsize=12, fontweight="bold", fontname="Times New Roman")
    colorbar.ax.tick_params(labelsize=10, width=1.4, length=4)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    figure.savefig(output_path, dpi=600, facecolor="white")
    plt.close(figure)


def main():
    saved = joblib.load(MODEL_PATH)
    matrix = build_matrix(saved["feature_columns"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for x_feature, y_feature, label, name in PDP_PAIRS:
        x_grid, y_grid, surface = calculate_2d_pdp(saved["model"], matrix, x_feature, y_feature)
        draw_surface(x_grid, y_grid, surface, r"PV (cm$^3$/g)", label,
                     OUTPUT_DIR / f"C_CatBoost_2D_PDP_{name}.png")

    for code, config in KR_CONFIGS.items():
        saved = joblib.load(config["model"])
        matrix = build_pipeline_matrix(config, saved["feature_columns"])
        config["folder"].mkdir(parents=True, exist_ok=True)
        for x_feature, y_feature, label, name in PDP_PAIRS:
            x_grid, y_grid, surface = calculate_2d_pdp(saved["model"], matrix, x_feature, y_feature)
            draw_surface(x_grid, y_grid, surface, r"PV (cm$^3$/g)", label,
                         config["folder"] / f"{config['prefix']}_2D_PDP_{name}.png", config["unit"])


if __name__ == "__main__":
    main()
