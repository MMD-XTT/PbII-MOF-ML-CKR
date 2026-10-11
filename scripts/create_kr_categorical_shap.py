"""Create standardized categorical SHAP bars, categorical PDPs, and beeswarms for K/R."""

import argparse
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
import shap

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

AXES_W_IN, AXES_H_IN = 4.0, 3.0
MARGINS = {"left": 0.86, "right": 0.24, "top": 0.32, "bottom": 0.72}
LW, TICK_FS, LABEL_FS = 2.4, 13, 16
MINT, PEACH = "#72BDA9", "#F2AE72"

MODULATOR_LABELS = {
    "none": "None", "acetic acid": "AcOH", "hydrofluoric acid": "HF",
    "hydrochloric acid": "HCl", "sodium hydroxide": "NaOH", "formic acid": "HCOOH",
    "triethylamine": "TEA", "urea": "Urea", "oxalic acid": "OxA",
    "isonicotinic acid": "INA", "nitric acid": "HNO3", "butyric acid": "BuA",
    "benzoic acid": "BzA",
}

CONFIGS = {
    "K": {
        "model": PACKAGE_ROOT / 'model/K/XGBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/K模型_qt统一.xlsx',
        "target": "csv_qt",
        "shap": PACKAGE_ROOT / 'model/K/XGBoost/SHAP_分组占比.xlsx',
        "output": PACKAGE_ROOT / 'outputs/K_XGBoost_PDP',
        "prefix": "K_XGBoost",
    },
    "R": {
        "model": PACKAGE_ROOT / 'model/R/CatBoost/final_pipeline.joblib',
        "data": PACKAGE_ROOT / 'data/R_model.xlsx',
        "target": "csv_retention_percent",
        "shap": PACKAGE_ROOT / 'model/R/CatBoost/SHAP_分组占比.xlsx',
        "output": PACKAGE_ROOT / 'outputs/R_CatBoost_PDP',
        "prefix": "R_CatBoost",
    },
}

GROUPS = ("M", "MN", "LF", "Mod")
SHEETS = {"M": "M", "MN": "metal_node_elements", "LF": "Linker family", "Mod": "modulator"}


def tokens(value):
    return set() if pd.isna(value) else {part.strip() for part in str(value).split(";") if part.strip()}


def label(group, value):
    value = str(value)
    return MODULATOR_LABELS.get(value, value) if group == "Mod" else value


def category_table(config, group):
    table = pd.read_excel(config["shap"], sheet_name=SHEETS[group]).copy()
    table["display"] = table["component"].map(lambda item: label(group, item))
    return table


def layout_for(group, config):
    labels = category_table(config, group)["display"].astype(str).tolist()
    left = max(MARGINS["left"], 0.11 * max(map(len, labels)) + 0.18)
    tick = 10 if len(labels) >= 12 or max(map(len, labels)) >= 15 else TICK_FS
    return left, tick


def raw_matrix(config, feature_columns):
    data = pd.read_excel(config["data"])
    data = data[data[config["target"]].notna()].copy()
    for prefix, source in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                           ("modulator", "modulator")):
        source_tokens = data[source].map(tokens)
        for column in (item for item in feature_columns if item.startswith(f"{prefix}__")):
            token = column.split("__", 1)[1]
            data[column] = source_tokens.map(lambda values: int(token in values))
    matrix = data[feature_columns].copy()
    for column in matrix.columns:
        if column != "main":
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def frame(axis, tick_size=TICK_FS):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LW)
    axis.tick_params(labelsize=tick_size, width=1.8, length=5)
    for item in axis.get_xticklabels() + axis.get_yticklabels():
        item.set_fontname("Times New Roman")
        item.set_fontweight("bold")


def draw_bar(config, group):
    table = category_table(config, group).sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    left, tick_size = layout_for(group, config)
    width, height = left + AXES_W_IN + MARGINS["right"], MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(width, height), dpi=300)
    axis = figure.add_axes([left / width, MARGINS["bottom"] / height, AXES_W_IN / width, AXES_H_IN / height])
    frame(axis, tick_size)
    count = len(table)
    bar_height = min(0.66, max(0.42, 7.4 / count))
    axis.barh(np.arange(count), table["mean_abs_shap"], height=bar_height,
              color=MINT, edgecolor="black", linewidth=0.8)
    axis.set_yticks(np.arange(count))
    axis.set_yticklabels(table["display"])
    axis.set_xlabel("mean |SHAP value|", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_title(group, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman", pad=8)
    axis.set_xlim(0, table["mean_abs_shap"].max() * 1.08)
    axis.set_ylim(-0.60, count - 0.40)
    axis.tick_params(axis="y", labelsize=tick_size, width=1.8, length=0)
    config["output"].mkdir(parents=True, exist_ok=True)
    figure.savefig(config["output"] / f"{config['prefix']}_{group}_SHAP_Bar.png", dpi=600, facecolor="white")
    plt.close(figure)


def raw_column(group, component):
    if group == "M":
        return "main"
    return {"MN": f"metal__{component}", "LF": f"linker__{component}", "Mod": f"modulator__{component}"}[group]


def transformed_column(group, component):
    if group == "M":
        return f"main__main_{component}"
    return f"numeric__{raw_column(group, component)}"


def draw_pdp(config, group, saved, matrix):
    table = category_table(config, group).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    components, labels = table["component"].tolist(), table["display"].tolist()
    values = []
    for component in components:
        counterfactual = matrix.copy()
        if group == "M":
            counterfactual["main"] = component
        else:
            counterfactual[raw_column(group, component)] = 1
        values.append(float(np.mean(saved["model"].predict(counterfactual))))

    width, height = MARGINS["left"] + AXES_W_IN + MARGINS["right"], MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(width, height), dpi=300)
    axis = figure.add_axes([MARGINS["left"] / width, MARGINS["bottom"] / height, AXES_W_IN / width, AXES_H_IN / height])
    x = np.arange(len(labels))
    axis.plot(x, values, color=PEACH, marker="o", markersize=4.6, linewidth=1.5)
    axis.set_xlim(-0.45, len(labels) - 0.55)
    padding = max((max(values) - min(values)) * 0.13, 0.03)
    axis.set_ylim(min(values) - padding, max(values) + padding)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=52, ha="right", rotation_mode="anchor")
    axis.set_ylabel("Partial Dependence", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    x_tick = 8 if len(labels) >= 12 or max(map(len, labels)) >= 15 else TICK_FS
    frame(axis)
    for item in axis.get_xticklabels():
        item.set_fontsize(x_tick)
        item.set_transform(item.get_transform() + ScaledTranslation(5 / 72, 0, figure.dpi_scale_trans))
    config["output"].mkdir(parents=True, exist_ok=True)
    figure.savefig(config["output"] / f"{config['prefix']}_{group}_PDP.png", dpi=600, facecolor="white")
    plt.close(figure)


def draw_beeswarm(config, group, transformed, transformed_names, shap_values):
    table = category_table(config, group).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    components = table["component"].tolist()
    wanted = [transformed_column(group, component) for component in components]
    index_by_name = {name: index for index, name in enumerate(transformed_names)}
    indices = [index_by_name[name] for name in wanted if name in index_by_name]
    present = [components[i] for i, name in enumerate(wanted) if name in index_by_name]
    labels = [label(group, value) for value in present]
    left, tick_size = layout_for(group, config)
    width, height = left + AXES_W_IN + 0.90, MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(width, height), dpi=300)
    shap.summary_plot(shap_values[:, indices], transformed[:, indices], feature_names=labels,
                      max_display=len(indices), show=False, plot_size=None,
                      color_bar=True, color_bar_label="Feature value")
    axis = plt.gca()
    axis.set_position([left / width, MARGINS["bottom"] / height, AXES_W_IN / width, AXES_H_IN / height])
    axis.set_xlabel("SHAP value", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    frame(axis, tick_size)
    color_axis = next(item for item in figure.axes if item is not axis)
    color_axis.set_position([(left + AXES_W_IN + 0.24) / width, MARGINS["bottom"] / height,
                             0.13 / width, AXES_H_IN / height])
    color_axis.set_ylabel("Feature value", fontsize=12, fontweight="bold", fontname="Times New Roman")
    color_axis.tick_params(labelsize=10, width=1.4, length=4)
    for item in color_axis.get_yticklabels():
        item.set_fontname("Times New Roman")
        item.set_fontweight("bold")
    config["output"].mkdir(parents=True, exist_ok=True)
    figure.savefig(config["output"] / f"{config['prefix']}_{group}_SHAP_Beeswarm.png", dpi=600, facecolor="white")
    plt.close(figure)


def run(code, chart):
    config = CONFIGS[code]
    saved = joblib.load(config["model"])
    matrix = raw_matrix(config, saved["feature_columns"])
    if chart in {"beeswarm", "all"}:
        preprocessor = saved["model"].named_steps["preprocess"]
        estimator = saved["model"].named_steps["model"]
        transformed = preprocessor.transform(matrix)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed_names = list(preprocessor.get_feature_names_out())
        shap_values = shap.TreeExplainer(estimator).shap_values(transformed)
    for group in GROUPS:
        if chart in {"bar", "all"}:
            draw_bar(config, group)
        if chart in {"pdp", "all"}:
            draw_pdp(config, group, saved, matrix)
        if chart in {"beeswarm", "all"}:
            draw_beeswarm(config, group, transformed, transformed_names, shap_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("K", "R"), default=("K", "R"))
    parser.add_argument("--chart", choices=("bar", "pdp", "beeswarm", "all"), default="all")
    arguments = parser.parse_args()
    for model_code in arguments.models:
        run(model_code, arguments.chart)
