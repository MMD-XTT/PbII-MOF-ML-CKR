"""Create grouped SHAP bar, categorical-pie, and beeswarm figures for the K/R best models."""

import argparse
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

BAR_LW = 2.0
TICK_FS = 13
LABEL_FS = 16
PALETTE = ["#9BD8C4", "#72BDA9", "#56A7B6", "#79B7D2", "#E1C78F", "#F3B47A"]
LABELS = {
    "csv_t": "t", "csv_cycle": "CN", "csv_pH": "pH", "M": "M",
    "MOF_dose_g_L": r"C$_{\mathrm{MOF}}$", "BET_m2_g": "BET",
    "Linker family": "LF", "modulator": "Mod", "carboxyl": "-COOH",
    "average_pore_diameter_nm": "PD", "total_pore_volume_cm3_g": "PV",
    "S_donor": "S", "csv_temperature_K": "Tem", "N_donor": "N",
    "amino": r"-NH$_2$", "thiol": "-SH", "sulfonic": r"-SO$_3$H",
    "P_donor": "P", "phosphonic": r"-PO$_3$H$_2$", "O_donor": "O",
    "metal_node_elements": "MN",
}
MODULATOR_LABELS = {
    "none": "None", "acetic acid": "AcOH", "hydrofluoric acid": "HF",
    "hydrochloric acid": "HCl", "sodium hydroxide": "NaOH", "formic acid": "HCOOH",
    "triethylamine": "TEA", "urea": "Urea", "oxalic acid": "OxA",
    "isonicotinic acid": "INA", "nitric acid": "HNO3", "butyric acid": "BuA",
    "benzoic acid": "BzA",
}


def split_tokens(value):
    return set() if pd.isna(value) else {item.strip() for item in str(value).split(";") if item.strip()}


def style_axis(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(BAR_LW)
    axis.tick_params(labelsize=TICK_FS, width=1.8, length=5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def component_label(group, feature):
    feature = str(feature).replace("numeric__", "").replace("main__main_", "")
    if group == "metal_node_elements":
        return feature.replace("metal__", "")
    if group == "Linker family":
        return feature.replace("linker__", "")
    if group == "modulator":
        key = feature.replace("modulator__", "")
        return MODULATOR_LABELS.get(key, key)
    return feature


def load_pie_tables(path):
    book = pd.ExcelFile(path)
    groups = {"M": "M", "metal_node_elements": "metal_node_elements",
              "Linker family": "Linker family", "modulator": "modulator"}
    tables = {}
    for group, sheet_name in groups.items():
        table = pd.read_excel(path, sheet_name=sheet_name)
        table["display"] = table["component"].map(lambda value: component_label(group, value))
        tables[group] = table
    return tables


def draw_pies(tables, shares, model_code, output):
    figure = plt.figure(figsize=(5.50, 7.15), dpi=300)
    layout = {"M": [0.10, 0.70, 0.80, 0.20], "metal_node_elements": [0.10, 0.46, 0.80, 0.20],
              "Linker family": [0.10, 0.22, 0.80, 0.20], "modulator": [0.10, 0.00, 0.80, 0.20]}
    titles = {"M": "M", "metal_node_elements": "MN", "Linker family": "LF", "modulator": "Mod"}
    for group, position in layout.items():
        axis = figure.add_axes(position)
        table = tables[group]
        wedges, _ = axis.pie(table["share_within_group_percent"], startangle=90, counterclock=False,
                              colors=[PALETTE[i % len(PALETTE)] for i in range(len(table))],
                              wedgeprops={"edgecolor": "white", "linewidth": 0.6})
        axis.set_title(f"{titles[group]} ({shares.get(group, 0):.2f}%)", fontsize=10,
                       fontweight="bold", fontname="Times New Roman", pad=1)
        axis.legend(wedges, table["display"], loc="center left", bbox_to_anchor=(0.94, 0.5),
                    frameon=False, ncol=2, handlelength=0.9, handletextpad=0.35,
                    labelspacing=0.30, columnspacing=0.55,
                    prop={"family": "Times New Roman", "weight": "bold", "size": 6.8})
    figure.savefig(output, dpi=600, facecolor="white")
    plt.close(figure)


def draw_bar(summary, model_code, output):
    chart = summary.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True).copy()
    chart["label"] = chart["feature_group"].map(LABELS).fillna(chart["feature_group"])
    figure = plt.figure(figsize=(5.10, 7.15), dpi=300)
    axis = figure.add_axes([0.30, 0.07, 0.64, 0.87])
    style_axis(axis)
    plotted = axis.barh(np.arange(len(chart)), chart["mean_abs_shap"], height=0.64,
                        color=[PALETTE[i % len(PALETTE)] for i in range(len(chart))],
                        edgecolor="black", linewidth=0.8)
    axis.set_yticks(np.arange(len(chart)))
    axis.set_yticklabels(chart["label"])
    axis.set_xlim(0, chart["mean_abs_shap"].max() * 1.28)
    axis.set_ylim(-0.77, len(chart) - 0.23)
    axis.set_xlabel("mean |SHAP value|", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_title(f"{model_code} SHAP", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman", pad=8)
    offset = chart["mean_abs_shap"].max() * 0.012
    for bar, percentage in zip(plotted, chart["shap_percentage"]):
        axis.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, f"{percentage:.2f}%",
                  va="center", ha="left", fontsize=9, fontweight="bold", fontname="Times New Roman")
    figure.savefig(output, dpi=600, facecolor="white")
    plt.close(figure)


def build_raw_matrix(source, target, feature_columns):
    data = pd.read_excel(source)
    data = data[data[target].notna()].copy()
    for prefix, raw_column in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                               ("modulator", "modulator")):
        tokens = data[raw_column].map(split_tokens)
        for column in [item for item in feature_columns if item.startswith(f"{prefix}__")]:
            token = column.split("__", 1)[1]
            data[column] = tokens.map(lambda items: int(token in items))
    matrix = data[feature_columns].copy()
    for column in matrix.columns:
        if column != "main":
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def normalized_group(name):
    if name.startswith("main__main_"):
        return "M"
    name = name.replace("numeric__", "")
    if name.startswith("missingindicator_"):
        name = name.replace("missingindicator_", "")
    if name.startswith("metal__"):
        return "metal_node_elements"
    if name.startswith("linker__"):
        return "Linker family"
    if name.startswith("modulator__"):
        return "modulator"
    return name


def grouped_values(raw, groups):
    values = {}
    for group in groups:
        if group == "M":
            values[group] = raw["main"].astype(str).str.extract(r"(\d+)")[0].astype(float).to_numpy()
        elif group == "metal_node_elements":
            values[group] = raw[[c for c in raw if c.startswith("metal__")]].sum(axis=1).to_numpy()
        elif group == "Linker family":
            values[group] = raw[[c for c in raw if c.startswith("linker__")]].sum(axis=1).to_numpy()
        elif group == "modulator":
            values[group] = raw[[c for c in raw if c.startswith("modulator__")]].sum(axis=1).to_numpy()
        else:
            values[group] = raw[group].to_numpy()
    return values


def draw_beeswarm(model_path, source, target, summary, model_code, output):
    saved = joblib.load(model_path)
    pipeline, feature_columns = saved["model"], saved["feature_columns"]
    raw = build_raw_matrix(source, target, feature_columns)
    preprocessor, estimator = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    transformed = preprocessor.transform(raw)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_names = preprocessor.get_feature_names_out()
    shap_values = shap.TreeExplainer(estimator).shap_values(transformed)
    group_order = summary["feature_group"].tolist()
    feature_groups = [normalized_group(name) for name in transformed_names]
    contributions = np.column_stack([
        shap_values[:, [i for i, group in enumerate(feature_groups) if group == wanted]].sum(axis=1)
        for wanted in group_order
    ])
    raw_values = grouped_values(raw, group_order)
    display_values = np.column_stack([raw_values[group] for group in group_order])

    figure = plt.figure(figsize=(5.10, 7.15), dpi=300)
    shap.summary_plot(contributions, display_values,
                      feature_names=[LABELS.get(group, group) for group in group_order],
                      max_display=len(group_order), sort=False, show=False,
                      plot_size=None, color_bar=True, color_bar_label="Feature value")
    axis = plt.gca()
    axis.set_position([0.20, 0.08, 0.60, 0.82])
    figure.suptitle(f"{model_code} SHAP", y=0.975, fontsize=LABEL_FS,
                    fontweight="bold", fontname="Times New Roman")
    axis.set_xlabel("SHAP value", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    style_axis(axis)
    color_axis = next((item for item in figure.axes if item is not axis), None)
    if color_axis is not None:
        color_axis.set_position([0.84, 0.08, 0.025, 0.82])
        color_axis.set_ylabel("Feature value", fontsize=11, fontweight="bold", fontname="Times New Roman")
        color_axis.tick_params(labelsize=10, width=1.4, length=4)
        for label in color_axis.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontweight("bold")
    figure.savefig(output, dpi=600, facecolor="white")
    with Image.open(output) as image:
        image.convert("RGB").save(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, choices=["K", "R"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--shap-csv", required=True)
    parser.add_argument("--shap-xlsx", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    summary = pd.read_csv(args.shap_csv)
    draw_bar(summary, args.code, output_dir / f"{args.code}_BestModel_SHAP_Bar.png")
    shares = dict(zip(summary["feature_group"], summary["shap_percentage"]))
    draw_pies(load_pie_tables(args.shap_xlsx), shares, args.code,
              output_dir / f"{args.code}_BestModel_SHAP_CategoricalPies.png")
    draw_beeswarm(args.model, args.data, args.target, summary, args.code,
                  output_dir / f"{args.code}_BestModel_SHAP_Beeswarm.png")


if __name__ == "__main__":
    main()
