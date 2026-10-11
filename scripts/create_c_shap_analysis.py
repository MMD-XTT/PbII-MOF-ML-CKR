"""Create a publication-style grouped SHAP analysis panel for the C CatBoost model."""

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
from PIL import Image

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

LW = 2.4
BAR_LW = 2.0
TICK_FS = 13
LABEL_FS = 16
PALETTE = ["#9BD8C4", "#72BDA9", "#56A7B6", "#79B7D2", "#E1C78F", "#F3B47A"]
LABELS = {
    "csv_ce": r"C$_{\mathrm{e}}$", "csv_pH": "pH", "M": "M", "MOF_dose_g_L": r"C$_{\mathrm{MOF}}$",
    "BET_m2_g": "BET", "Linker family": "LF", "modulator": "Mod",
    "carboxyl": "-COOH", "average_pore_diameter_nm": "PD",
    "total_pore_volume_cm3_g": "PV", "S_donor": "S", "csv_temperature_K": "Tem",
    "N_donor": "N", "amino": r"-NH$_2$", "thiol": "-SH", "sulfonic": r"-SO$_3$H",
    "P_donor": "P", "phosphonic": r"-PO$_3$H$_2$", "O_donor": "O",
    "metal_node_elements": "MN",
}
MODULATOR_LABELS = {
    "none": "None", "acetic acid": "AcOH", "hydrofluoric acid": "HF",
    "hydrochloric acid": "HCl", "sodium hydroxide": "NaOH", "formic acid": "HCOOH",
    "triethylamine": "TEA", "urea": "Urea", "oxalic acid": "OxA",
    "isonicotinic acid": "INA", "nitric acid": "HNO3", "butyric acid": "Butyric acid",
    "benzoic acid": "BzA", "butyric acid": "BuA",
}

GROUP_ORDER = [
    "csv_ce", "metal_node_elements", "csv_pH", "M", "MOF_dose_g_L", "BET_m2_g",
    "Linker family", "modulator", "carboxyl", "average_pore_diameter_nm",
    "total_pore_volume_cm3_g", "S_donor", "csv_temperature_K", "N_donor", "amino",
    "thiol", "sulfonic", "P_donor", "phosphonic", "O_donor",
]


def split_tokens(value):
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def build_c_model_matrix(source_path, feature_columns):
    """Recreate the numeric feature matrix used by the saved C CatBoost model."""
    data = pd.read_excel(source_path)
    data = data[data["csv_qe"].notna()].copy()
    data["main"] = data["main"].astype(str).str.extract(r"(\d+)")[0].astype(float)

    for prefix, raw_column in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                               ("modulator", "modulator")):
        matching = [column for column in feature_columns if column.startswith(f"{prefix}__")]
        tokens = data[raw_column].map(split_tokens)
        for column in matching:
            token = column.split("__", 1)[1]
            data[column] = tokens.map(lambda values: int(token in values))

    matrix = data[feature_columns].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        matrix[column] = matrix[column].fillna(matrix[column].median())
    return data, matrix


def grouped_beeswarm_inputs(shap_values, matrix):
    """Sum one-hot SHAP effects back to the visual feature domains."""
    group_columns = {
        "M": ["main"],
        "metal_node_elements": [column for column in matrix if column.startswith("metal__")],
        "Linker family": [column for column in matrix if column.startswith("linker__")],
        "modulator": [column for column in matrix if column.startswith("modulator__")],
    }
    for group in GROUP_ORDER:
        if group not in group_columns:
            group_columns[group] = [group]

    positions = {column: index for index, column in enumerate(matrix.columns)}
    contributions, display_values = [], []
    for group in GROUP_ORDER:
        columns = group_columns[group]
        indices = [positions[column] for column in columns]
        contributions.append(shap_values[:, indices].sum(axis=1))
        display_values.append(matrix[columns].sum(axis=1).to_numpy())
    return np.column_stack(contributions), np.column_stack(display_values)


def draw_beeswarm(model_path, source_path, output_path):
    saved = joblib.load(model_path)
    model, feature_columns = saved["model"], saved["feature_columns"]
    _, matrix = build_c_model_matrix(source_path, feature_columns)
    shap_values = model.get_feature_importance(Pool(matrix, feature_names=feature_columns),
                                              type="ShapValues")[:, :-1]
    grouped_shap, grouped_values = grouped_beeswarm_inputs(shap_values, matrix)

    figure = plt.figure(figsize=(5.10, 7.15), dpi=300)
    shap.summary_plot(grouped_shap, grouped_values,
                      feature_names=[LABELS.get(group, group) for group in GROUP_ORDER],
                      max_display=len(GROUP_ORDER), sort=False, show=False,
                      plot_size=None, color_bar=True, color_bar_label="Feature value")
    axis = plt.gca()
    axis.set_position([0.20, 0.08, 0.60, 0.82])
    figure.suptitle("C SHAP", y=0.975, fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    axis.set_xlabel("SHAP value", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    style_axis(axis)
    for spine in axis.spines.values():
        spine.set_linewidth(BAR_LW)
    for label in axis.get_yticklabels():
        label.set_fontsize(TICK_FS)

    color_axis = next((item for item in figure.axes if item is not axis), None)
    if color_axis is not None:
        color_axis.set_position([0.84, 0.08, 0.025, 0.82])
        color_axis.set_ylabel("Feature value", fontsize=11, fontweight="bold", fontname="Times New Roman")
        color_axis.tick_params(labelsize=10, width=1.4, length=4)
        for label in color_axis.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontweight("bold")
    figure.savefig(output_path, dpi=600, facecolor="white")
    # SHAP can write a transparent PNG; normalize it for reliable Word/Codex previews.
    with Image.open(output_path) as image:
        image.convert("RGB").save(output_path)
    plt.close(figure)


def style_axis(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LW)
    axis.tick_params(labelsize=TICK_FS, width=1.8, length=5)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def short_component(group, feature):
    if group == "MN":
        return feature.replace("metal__", "")
    if group == "LF":
        return feature.replace("linker__", "")
    if group == "Mod":
        return MODULATOR_LABELS.get(feature.replace("modulator__", ""), feature.replace("modulator__", ""))
    return feature


def load_group_tables(path):
    workbook = pd.ExcelFile(path)
    # The workbook's Chinese sheet names are intentionally accessed by index.
    mapping = {"M": 1, "MN": 3, "LF": 4, "Mod": 5}
    tables = {}
    for group, sheet_index in mapping.items():
        table = pd.read_excel(path, sheet_name=workbook.sheet_names[sheet_index])
        table["display"] = table["feature"].map(lambda value: short_component(group, str(value)))
        tables[group] = table
    return tables


def draw_pie(axis, table, title):
    values = table["shap_percentage"].to_numpy(float)
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(values))]
    wedges, _ = axis.pie(values, colors=colors, startangle=90, counterclock=False,
                          wedgeprops={"edgecolor": "white", "linewidth": 0.6})
    axis.set_title(title, fontsize=10, fontweight="bold", fontname="Times New Roman", pad=1)
    axis.legend(wedges, table["display"], loc="center left", bbox_to_anchor=(0.94, 0.5),
                frameon=False, ncol=2, fontsize=6.8, handlelength=0.9, handletextpad=0.35,
                labelspacing=0.30, columnspacing=0.55,
                prop={"family": "Times New Roman", "weight": "bold", "size": 6.8})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-csv", required=True)
    parser.add_argument("--shap-xlsx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(PACKAGE_ROOT / 'model/C/CatBoost/final_pipeline.joblib'))
    parser.add_argument("--data", default=str(PACKAGE_ROOT / 'data/C_model.xlsx'))
    args = parser.parse_args()

    shap = pd.read_csv(args.shap_csv).sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    shap["label"] = shap["feature_group"].map(LABELS).fillna(shap["feature_group"])
    group_tables = load_group_tables(args.shap_xlsx)
    group_share = {row.feature_group: row.shap_percentage for row in shap.itertuples()}

    # Keep a compact visual buffer between the outer bars and the frame.
    figure = plt.figure(figsize=(5.10, 7.15), dpi=300)
    bars = figure.add_axes([0.30, 0.07, 0.64, 0.87])
    style_axis(bars)
    for spine in bars.spines.values():
        spine.set_linewidth(BAR_LW)
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(shap))]
    plotted = bars.barh(np.arange(len(shap)), shap["mean_abs_shap"], height=0.64,
                        color=colors, edgecolor="black", linewidth=0.8)
    bars.set_yticks(np.arange(len(shap)))
    bars.set_yticklabels(shap["label"])
    bars.set_xlabel("mean |SHAP value|", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman")
    bars.set_title("C SHAP", loc="center", fontsize=LABEL_FS, fontweight="bold", fontname="Times New Roman", pad=8)
    x_offset = shap["mean_abs_shap"].max() * 0.012
    for bar, percentage in zip(plotted, shap["shap_percentage"]):
        bars.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
                  f"{percentage:.2f}%", va="center", ha="left", fontsize=9.0,
                  fontweight="bold", fontname="Times New Roman")
    bars.set_xlim(0, shap["mean_abs_shap"].max() * 1.28)
    bars.set_ylim(-0.77, len(shap) - 0.23)
    bar_output = Path(args.output).with_name("C_CatBoost_SHAP_Bar.png")
    figure.savefig(bar_output, dpi=600, facecolor="white")
    plt.close(figure)

    # Pie-chart panel is separate so categorical contributions remain legible.
    figure = plt.figure(figsize=(5.50, 7.15), dpi=300)
    layout = {"M": [0.10, 0.70, 0.80, 0.20], "MN": [0.10, 0.46, 0.80, 0.20],
              "LF": [0.10, 0.22, 0.80, 0.20], "Mod": [0.10, 0.00, 0.80, 0.20]}
    for group, position in layout.items():
        axis = figure.add_axes(position)
        draw_pie(axis, group_tables[group], f"{group} ({group_share.get({'M': 'M', 'MN': 'metal_node_elements', 'LF': 'Linker family', 'Mod': 'modulator'}[group], 0):.2f}%)")

    pie_output = Path(args.output).with_name("C_CatBoost_SHAP_CategoricalPies.png")
    figure.savefig(pie_output, dpi=600, facecolor="white")
    plt.close(figure)

    beeswarm_output = Path(args.output).with_name("C_CatBoost_SHAP_Beeswarm.png")
    draw_beeswarm(args.model, args.data, beeswarm_output)


if __name__ == "__main__":
    main()
