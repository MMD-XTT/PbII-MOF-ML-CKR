"""Create an evidence-integrated, material-level preliminary Figure 5 PAI screen."""

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import textwrap
import re

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = PACKAGE_ROOT / 'outputs'
OUT_DIR = ROOT / "Figure5_PAI_Ce300_pH6_CMOF0p1_test"
DATA = PACKAGE_ROOT / 'data/master.xlsx'
W_BOOK = ROOT / str(PACKAGE_ROOT / "data/screening/water_stability_register_1.xlsx")
OUT_DATA = OUT_DIR / "Figure5_PAI_reference_Ce300_pH5_CMOF0p2_candidate_data.csv"

MODELS = {
    "C": (PACKAGE_ROOT / 'model/C/CatBoost/final_pipeline.joblib',
          PACKAGE_ROOT / 'data/C_model.xlsx', "csv_qe"),
    "R": (PACKAGE_ROOT / 'model/R/CatBoost/final_pipeline.joblib',
          PACKAGE_ROOT / 'data/R_model.xlsx', "csv_retention_percent"),
}
REFERENCE = {"csv_ce": 300.0, "csv_pH": 5.0, "csv_temperature_K": 298.0,
             "MOF_dose_g_L": 0.2, "csv_cycle": 3.0}


def tokens(value):
    return set() if pd.isna(value) else {item.strip() for item in str(value).split(";") if item.strip()}


def training_matrix(training, features):
    frame = training.copy()
    frame["main"] = frame["main"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for prefix, source in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                           ("modulator", "modulator")):
        source_values = frame[source].map(tokens)
        for feature in (item for item in features if item.startswith(prefix + "__")):
            token = feature.split("__", 1)[1]
            frame[feature] = source_values.map(lambda values: int(token in values))
    result = frame[features].copy()
    for feature in result:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result


def direct_matrix(materials, features, training):
    frame = materials.copy()
    frame["main"] = frame["main"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for prefix, source in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                           ("modulator", "modulator")):
        source_values = frame[source].map(tokens)
        for feature in (item for item in features if item.startswith(prefix + "__")):
            token = feature.split("__", 1)[1]
            frame[feature] = source_values.map(lambda values: int(token in values))
    result = frame[features].copy()
    for feature, value in REFERENCE.items():
        if feature in result:
            result[feature] = value
    reference_train = training_matrix(training, features)
    for feature in result:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
        result[feature] = result[feature].fillna(reference_train[feature].median())
    return result


def pipeline_matrix(materials, features):
    frame = materials.copy()
    for prefix, source in (("metal", "metal_node_elements"), ("linker", "Linker family"),
                           ("modulator", "modulator")):
        source_values = frame[source].map(tokens)
        for feature in (item for item in features if item.startswith(prefix + "__")):
            token = feature.split("__", 1)[1]
            frame[feature] = source_values.map(lambda values: int(token in values))
    result = frame[features].copy()
    for feature, value in REFERENCE.items():
        if feature in result:
            result[feature] = value
    return result


def observed_k60(raw):
    kinetics = raw[raw["csv_data_type"].astype(str).str.lower().eq("kinetics")].copy()
    kinetics = kinetics[kinetics["csv_t"].notna() & kinetics["csv_qt"].notna()].copy()
    environment = ["csv_pH", "csv_temperature_K", "MOF_dose_g_L", "csv_ce"]
    for feature in environment:
        kinetics[feature] = pd.to_numeric(kinetics[feature], errors="coerce").round(4).fillna(-99999)
    records = []
    for key, curve in kinetics.groupby(["material_id"] + environment, dropna=False):
        curve = curve.sort_values("csv_t").drop_duplicates("csv_t")
        time = curve["csv_t"].to_numpy(float)
        uptake = curve["csv_qt"].to_numpy(float)
        if len(curve) < 4 or time.min() > 60 or time.max() < 60:
            continue
        q60 = float(np.interp(60, time, uptake))
        qmax = float(np.nanmax(uptake))
        if qmax <= 0:
            continue
        records.append({"material_id": key[0], "K_curve_q60_mg_g": q60, "K_curve_qmax_mg_g": qmax,
                        "K_score_observed_q60_qmax": float(np.clip(q60 / qmax, 0, 1)),
                        "K_curve_count": 1})
    curves = pd.DataFrame(records)
    aggregation = curves.groupby("material_id", as_index=False).agg(
        K_curve_q60_mg_g=("K_curve_q60_mg_g", "median"),
        K_curve_qmax_mg_g=("K_curve_qmax_mg_g", "median"),
        K_score_observed_q60_qmax=("K_score_observed_q60_qmax", "median"),
        K_curve_count=("K_curve_count", "sum"),
    )
    return aggregation


def qscale(series):
    low, high = np.nanquantile(series, [.05, .95])
    return np.clip((series - low) / (high - low), 0, 1), low, high


def pareto(frame):
    value = frame[["C_score", "K_score_observed_q60_qmax", "R_score", "W_score"]].to_numpy(float)
    frontier = np.ones(len(frame), dtype=bool)
    for index, current in enumerate(value):
        dominating = np.all(value >= current, axis=1) & np.any(value > current, axis=1)
        if dominating.any():
            frontier[index] = False
    return frontier


def short_name(value, limit=25):
    value = str(value)
    return value if len(value) <= limit else value[:limit - 1] + "…"


raw = pd.read_excel(DATA)
materials = raw.drop_duplicates("material_id").copy()
result = materials[["paper_id", "material_id", "material_original_name", "main", "metal_node_elements", "Linker family", "modulator"]].copy()

c_saved = joblib.load(MODELS["C"][0])
c_train = pd.read_excel(MODELS["C"][1])
c_train = c_train[c_train[MODELS["C"][2]].notna()]
c_matrix = direct_matrix(materials, c_saved["feature_columns"], c_train)
result["qe_ref_mg_g"] = c_saved["model"].predict(c_matrix)

r_saved = joblib.load(MODELS["R"][0])
r_matrix = pipeline_matrix(materials, r_saved["feature_columns"])
result["retention_cycle3_percent"] = r_saved["model"].predict(r_matrix)

w_raw = pd.read_excel(W_BOOK, sheet_name=0, header=2)
w = pd.DataFrame({"material_id": w_raw.iloc[:, 1].astype(str),
                  "W_score": pd.to_numeric(w_raw.iloc[:, 15], errors="coerce"),
                  "W_evidence_level": w_raw.iloc[:, 12].astype(str),
                  "W_confidence": pd.to_numeric(w_raw.iloc[:, 13], errors="coerce")})
result = result.merge(w, on="material_id", how="left")

# Fix the C scale before filtering for available kinetic curves. This prevents
# kinetic-data availability from changing the capacity-score benchmark.
result["C_score"], c05, c95 = qscale(result["qe_ref_mg_g"])
result["R_score"] = np.clip(result["retention_cycle3_percent"] / 100, 0, 1)
result["W_score"] = result["W_score"].fillna(0.5)
result = result.merge(observed_k60(raw), on="material_id", how="inner")
result["PAI"] = (result["C_score"] * result["K_score_observed_q60_qmax"] * result["R_score"] * result["W_score"]) ** 0.25
result["Pareto_front"] = pareto(result)
result = result.sort_values("PAI", ascending=False).reset_index(drop=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")
top10 = result.head(10).copy()
top10.to_csv(OUT_DIR / "Figure5_Top10_PAI_ranking.csv", index=False, encoding="utf-8-sig")


def axes_43(left=0.85, right=.25, top=.32, bottom=.70, data_w=4.0, data_h=3.0):
    """Create a fixed-height data area following the project's plotting template."""
    fig = plt.figure(figsize=(left + data_w + right, bottom + data_h + top), dpi=240)
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(),
                       data_w / fig.get_figwidth(), data_h / fig.get_figheight()])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.tick_params(width=2, labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    return fig, ax


def clean_ticks(ax, axis="both"):
    ax.tick_params(axis=axis, which="both", length=0)


def two_line_name(value, width=18):
    """Return a readable two-line material name for compact manuscript panels."""
    value = str(value)
    if len(value) <= width:
        return value
    midpoint = len(value) // 2
    separators = [index for index, char in enumerate(value) if char in "@-/ "]
    cut = min(separators, key=lambda index: abs(index - midpoint)) if separators else midpoint
    first, second = value[:cut + 1].strip(), value[cut + 1:].strip()
    return f"{first}\n{short_name(second, width)}"


# Panel a: no hidden colour/size encoding. The orange ring has one explicit
# meaning only: non-dominated across C, K, R and W.
fig, ax = axes_43(left=.92, right=.68, bottom=.72)
scatter = ax.scatter(result["C_score"], result["K_score_observed_q60_qmax"], s=45,
           c=result["W_score"], cmap=LinearSegmentedColormap.from_list("w_mint", ["#E4F2EC", "#7FC7AB", "#187A57"]),
           vmin=0, vmax=1, edgecolors="#34534B", linewidths=.7, alpha=.95,
           label=f"Candidate materials (N = {len(result)})")
front = result[result["Pareto_front"]]
ax.scatter(front["C_score"], front["K_score_observed_q60_qmax"], s=105,
           facecolors="none", edgecolors=(0.906, 0.490, 0.157, 0.80), linewidths=2.2,
           label=f"Pareto-front materials (N = {len(front)})")
ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
ax.set_xlabel("Capacity score (C)", fontsize=17, fontweight="bold")
ax.set_ylabel(r"Kinetic progress, $q_{60}/q_{max}$ (K)", fontsize=17, fontweight="bold")
ax.grid(alpha=.18, linewidth=.8)
ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(.95, .02))
cax = fig.add_axes([(0.92 + 4.0 + .16) / fig.get_figwidth(), .72 / fig.get_figheight(),
                    .14 / fig.get_figwidth(), 3.0 / fig.get_figheight()])
colorbar = fig.colorbar(scatter, cax=cax)
colorbar.set_label("Water-stability evidence score (W)", fontsize=10, fontweight="bold")
colorbar.ax.tick_params(labelsize=9, width=1.5)
fig.savefig(OUT_DIR / "Figure5a_Pareto_screening.png", dpi=600, facecolor="white")
plt.close(fig)

# Panel b: ranked bars, four-sided border, one colour to avoid implying an
# unlabelled secondary variable.
ranked = top10.iloc[::-1]
fig, ax = axes_43(left=3.05, bottom=.70)
y = np.arange(len(ranked))
ax.barh(y, ranked["PAI"], color="#9DD6C3", edgecolor="black", linewidth=.8)
ax.set_yticks(y, [two_line_name(name, 16) for name in ranked["material_original_name"]])
ax.set_xlim(0, 1.02)
ax.set_xlabel("PAI", fontsize=17, fontweight="bold")
clean_ticks(ax, "y")
for label in ax.get_yticklabels():
    label.set_fontsize(8)
for yy, value in zip(y, ranked["PAI"]):
    ax.text(value + .015, yy, f"{value:.3f}", va="center", fontsize=9, fontweight="bold")
fig.savefig(OUT_DIR / "Figure5b_Top10_PAI_ranking.png", dpi=600, facecolor="white")
plt.close(fig)

# Panel c: order remains high-to-low from top to bottom; ticks retain labels
# but have no tick marks, as requested.
matrix = top10[["C_score", "K_score_observed_q60_qmax", "R_score", "W_score", "PAI"]].to_numpy()
cmap = LinearSegmentedColormap.from_list("mint", ["#F4F7F5", "#9DD6C3", "#218765"])
fig, ax = axes_43(left=2.55, bottom=.70)
image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap, origin="upper")
ax.set_xticks(range(5), ["C", "K", "R", "W", "PAI"])
ax.set_yticks(range(len(top10)), [two_line_name(name, 16) for name in top10["material_original_name"]])
clean_ticks(ax)
for label in ax.get_yticklabels():
    label.set_fontsize(8)
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold")
fig.savefig(OUT_DIR / "Figure5c_Top10_score_matrix.png", dpi=600, facecolor="white")
plt.close(fig)

# Panel d: component profiles for all top ten candidates; labels use material
# names rather than material IDs.
fig, ax = axes_43(left=.85, bottom=1.50, top=.28, data_w=4.0)
x = np.arange(len(top10)); width = .19
components = (("C_score", "C", "#9DD6C3"), ("K_score_observed_q60_qmax", "K", "#68B79A"),
              ("R_score", "R", "#F6C88E"), ("W_score", "W", "#EE9E58"))
for offset, (column, label, colour) in enumerate(components):
    ax.bar(x + (offset - 1.5) * width, top10[column], width=width, color=colour,
           edgecolor="black", linewidth=.35, label=label)
ax.set_xticks(x, [two_line_name(name, 8) for name in top10["material_original_name"]], rotation=45, ha="right")
ax.set_ylim(0, 1.17)
ax.set_ylabel("Component score", fontsize=17, fontweight="bold")
clean_ticks(ax, "x")
for label in ax.get_xticklabels():
    label.set_fontsize(6.5)
ax.legend(ncol=4, frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(.985, .985))
ax.grid(axis="y", alpha=.18, linewidth=.8)
fig.savefig(OUT_DIR / "Figure5d_Top10_component_profiles.png", dpi=600, facecolor="white")
plt.close(fig)

note = f"""# Figure 5 reference condition and calculation record

Reference condition: Ce = {REFERENCE['csv_ce']:.0f} mg/L; pH = {REFERENCE['csv_pH']:.1f}; T = {REFERENCE['csv_temperature_K']:.0f} K; CMOF = {REFERENCE['MOF_dose_g_L']:.1f} g/L; cycle = {REFERENCE['csv_cycle']:.0f}.

C is the C-model prediction at the reference condition. C is normalised with the P5--P95 range across all materials before filtering for kinetic-curve availability: P5 = {c05:.3f} mg/g and P95 = {c95:.3f} mg/g.

K is the observed kinetic progress q60/qmax obtained by interpolation of each material's usable literature kinetic curves. R is the R-model prediction at cycle 3 and the reference pH, temperature and dose. W is the water-stability evidence score. PAI = (C*K*R*W)^(1/4).
"""
(OUT_DIR / "Figure5_reference_condition_and_scoring.md").write_text(note, encoding="utf-8")

print(f"Scored {len(result)} materials; full-library C P05/P95: {c05:.2f}/{c95:.2f} mg/g")
print(result.head(10)[["material_id", "material_original_name", "C_score", "K_score_observed_q60_qmax", "R_score", "W_score", "PAI", "Pareto_front"]].to_string(index=False))
