"""Draw standardized subcategory SHAP bar charts for the C CatBoost model."""

import argparse
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from catboost import Pool
from matplotlib.transforms import ScaledTranslation

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

SHAP_WORKBOOK = PACKAGE_ROOT / 'model/C/CatBoost/SHAP_分组占比.xlsx'
MODEL_PATH = PACKAGE_ROOT / 'model/C/CatBoost/final_pipeline.joblib'
DATA_PATH = PACKAGE_ROOT / 'data/C_model.xlsx'
OUTPUT_DIR = PACKAGE_ROOT / 'outputs/C_CatBoost_PDP'
AXES_W_IN, AXES_H_IN = 4.0, 3.0
MARGINS = {"left": 0.86, "right": 0.24, "top": 0.32, "bottom": 0.72}
LW, TICK_FS, LABEL_FS = 2.4, 13, 16
MINT = "#72BDA9"
PEACH = "#F2AE72"
MODULATOR_LABELS = {
    "none": "None", "acetic acid": "AcOH", "hydrofluoric acid": "HF",
    "hydrochloric acid": "HCl", "sodium hydroxide": "NaOH", "formic acid": "HCOOH",
    "triethylamine": "TEA", "urea": "Urea", "oxalic acid": "OxA",
    "isonicotinic acid": "INA", "nitric acid": "HNO3", "butyric acid": "BuA",
    "benzoic acid": "BzA",
}

# Workbook sheets are addressed by index because some existing Chinese sheet titles are legacy-encoded.
GROUPS = {"MN": 3, "M": 1, "LF": 4, "Mod": 5}


def clean_label(feature, group):
    value = str(feature)
    if group == "MN":
        return value.replace("metal__", "")
    if group == "M":
        return value.replace("main__", "")
    if group == "LF":
        return value.replace("linker__", "")
    raw = value.replace("modulator__", "")
    return MODULATOR_LABELS.get(raw, raw)


def feature_prefix(group):
    return {"MN": "metal__", "M": "main__", "LF": "linker__", "Mod": "modulator__"}[group]


def category_layout(group):
    """Return category-sensitive label margin and font size for the 4:3 data area."""
    category_data = pd.read_excel(SHAP_WORKBOOK, sheet_name=GROUPS[group])
    labels = [clean_label(value, group) for value in category_data["feature"]]
    left_margin = max(MARGINS["left"], 0.11 * max(map(len, labels)) + 0.18)
    tick_size = 10 if len(labels) >= 12 or max(map(len, labels)) >= 15 else TICK_FS
    return left_margin, tick_size


def tokens(value):
    return set() if pd.isna(value) else {item.strip() for item in str(value).split(";") if item.strip()}


def build_model_matrix(feature_columns):
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


def draw_group(group):
    if group not in GROUPS:
        raise ValueError(f"Unsupported category group: {group}")
    data = pd.read_excel(SHAP_WORKBOOK, sheet_name=GROUPS[group])
    data["label"] = data["feature"].map(lambda item: clean_label(item, group))
    data = data.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

    left_margin, tick_size = category_layout(group)
    figure_width = left_margin + AXES_W_IN + MARGINS["right"]
    figure_height = MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    axis = figure.add_axes([left_margin / figure_width, MARGINS["bottom"] / figure_height,
                            AXES_W_IN / figure_width, AXES_H_IN / figure_height])
    for spine in axis.spines.values():
        spine.set_linewidth(LW)

    count = len(data)
    bar_height = min(0.66, max(0.42, 7.4 / count))
    axis.barh(np.arange(count), data["mean_abs_shap"], height=bar_height,
              color=MINT, edgecolor="black", linewidth=0.8)
    axis.set_yticks(np.arange(count))
    axis.set_yticklabels(data["label"])
    axis.set_xlabel("mean |SHAP value|", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_title(group, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman", pad=8)
    axis.set_xlim(0, data["mean_abs_shap"].max() * 1.08)
    axis.set_ylim(-0.60, count - 0.40)
    axis.tick_params(axis="x", labelsize=TICK_FS, width=1.8, length=5, top=False)
    axis.tick_params(axis="y", labelsize=tick_size, width=1.8, length=0, right=False, labelright=False)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"C_CatBoost_{group}_SHAP_Bar.png", dpi=600, facecolor="white")
    plt.close(figure)


def draw_beeswarm(group):
    if group not in GROUPS:
        raise ValueError(f"Unsupported category group: {group}")
    saved = joblib.load(MODEL_PATH)
    feature_columns = saved["feature_columns"]
    matrix = build_model_matrix(feature_columns)
    shap_values = saved["model"].get_feature_importance(
        Pool(matrix, feature_names=feature_columns), type="ShapValues")[:, :-1]
    left_margin, tick_size = category_layout(group)
    if group == "M":
        # The trained model stores M as one ordinal column.  Split that one SHAP
        # contribution across mutually exclusive M1-M10 indicators solely for a
        # standard beeswarm visualization consistent with MN/LF/Mod.
        group_data = pd.read_excel(SHAP_WORKBOOK, sheet_name=GROUPS[group])
        group_features = group_data["feature"].tolist()
        main_index = feature_columns.index("main")
        main_values = matrix["main"].round().astype(int).to_numpy()
        group_matrix = pd.DataFrame(0, index=matrix.index, columns=group_features)
        group_shap_values = np.zeros((len(matrix), len(group_features)))
        for index, label in enumerate(group_features):
            category = int(str(label).replace("M", ""))
            mask = main_values == category
            group_matrix.loc[mask, label] = 1
            group_shap_values[mask, index] = shap_values[mask, main_index]
    else:
        indices = [index for index, feature in enumerate(feature_columns) if feature.startswith(feature_prefix(group))]
        group_features = [feature_columns[index] for index in indices]
        group_matrix = matrix[group_features]
        group_shap_values = shap_values[:, indices]

    figure_width = left_margin + AXES_W_IN + 0.90
    figure_height = MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    shap.summary_plot(group_shap_values, group_matrix,
                      feature_names=[clean_label(feature, group) for feature in group_features],
                      max_display=len(group_features), show=False, plot_size=None,
                      color_bar=True, color_bar_label="Feature value")
    axis = plt.gca()
    axis.set_position([left_margin / figure_width, MARGINS["bottom"] / figure_height,
                       AXES_W_IN / figure_width, AXES_H_IN / figure_height])
    axis.set_xlabel("SHAP value", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LW)
    axis.tick_params(labelsize=tick_size, width=1.8, length=5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    color_axis = next(item for item in figure.axes if item is not axis)
    color_axis.set_position([(left_margin + AXES_W_IN + 0.24) / figure_width,
                             MARGINS["bottom"] / figure_height, 0.13 / figure_width,
                             AXES_H_IN / figure_height])
    color_axis.set_ylabel("Feature value", fontsize=12, fontweight="bold", fontname="Times New Roman")
    color_axis.tick_params(labelsize=10, width=1.4, length=4)
    for label in color_axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    figure.savefig(OUTPUT_DIR / f"C_CatBoost_{group}_SHAP_Beeswarm.png", dpi=600, facecolor="white")
    plt.close(figure)


def draw_group_pdp(group):
    """Plot group-level one-hot PDP values in descending SHAP-importance order."""
    if group not in GROUPS:
        raise ValueError(f"Unsupported category group: {group}")

    saved = joblib.load(MODEL_PATH)
    feature_columns = saved["feature_columns"]
    matrix = build_model_matrix(feature_columns)
    model = saved["model"]

    importance = pd.read_excel(SHAP_WORKBOOK, sheet_name=GROUPS[group])
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    category_features = importance["feature"].tolist()
    labels = [clean_label(feature, group) for feature in category_features]

    # Each point is the average predicted qe after setting the selected MN
    # binary feature to present while preserving every other observed feature.
    partial_dependence = []
    for feature in category_features:
        counterfactual = matrix.copy()
        if group == "M":
            counterfactual["main"] = int(str(feature).replace("M", ""))
        else:
            counterfactual[feature] = 1
        partial_dependence.append(float(np.mean(model.predict(counterfactual))))

    figure_width = MARGINS["left"] + AXES_W_IN + MARGINS["right"]
    figure_height = MARGINS["bottom"] + AXES_H_IN + MARGINS["top"]
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    axis = figure.add_axes([MARGINS["left"] / figure_width, MARGINS["bottom"] / figure_height,
                            AXES_W_IN / figure_width, AXES_H_IN / figure_height])
    for spine in axis.spines.values():
        spine.set_linewidth(LW)

    x_values = np.arange(len(labels))
    axis.plot(x_values, partial_dependence, color=PEACH, marker="o", markersize=4.6,
              markerfacecolor=PEACH, markeredgecolor=PEACH, linewidth=1.5)
    axis.set_xlim(-0.45, len(labels) - 0.55)
    padding = max((max(partial_dependence) - min(partial_dependence)) * 0.13, 3.0)
    axis.set_ylim(min(partial_dependence) - padding, max(partial_dependence) + padding)
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=52, ha="right", rotation_mode="anchor")
    axis.set_ylabel("Partial Dependence", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    x_tick_size = 8 if len(labels) >= 12 or max(map(len, labels)) >= 15 else TICK_FS
    axis.tick_params(axis="both", labelsize=TICK_FS, width=1.8, length=4.5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    for label in axis.get_xticklabels():
        label.set_fontsize(x_tick_size)
    # Retain the clean diagonal-label geometry while nudging each label rightward
    # a few typographic points to align it visually with its tick mark.
    offset = ScaledTranslation(5 / 72, 0, figure.dpi_scale_trans)
    for label in axis.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"C_CatBoost_{group}_PDP.png", dpi=600, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUPS) + ("all",), default="MN")
    parser.add_argument("--chart", choices=("bar", "beeswarm", "pdp", "both", "all"), default="both")
    args = parser.parse_args()
    groups = GROUPS if args.group == "all" else (args.group,)
    for group in groups:
        if args.chart in {"bar", "both", "all"}:
            draw_group(group)
        if args.chart in {"beeswarm", "both", "all"}:
            draw_beeswarm(group)
        if args.chart in {"pdp", "all"}:
            draw_group_pdp(group)
