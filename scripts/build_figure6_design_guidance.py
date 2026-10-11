"""Generate evidence-driven next-step figures after the Figure 5 PAI screen."""

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = PACKAGE_ROOT / 'outputs'
FIG5_DIR = ROOT / "Figure5_PAI_Ce300_pH6_CMOF0p1_test"
OUT_DIR = ROOT / "Figure6_Design_Guidance"
SCORES = FIG5_DIR / "Figure5_PAI_reference_Ce300_pH5_CMOF0p2_candidate_data.csv"
DATA = PACKAGE_ROOT / 'data/master.xlsx'

MINT = "#9DD6C3"
MINT_DARK = "#218765"
ORANGE = "#EE9E58"
ORANGE_LIGHT = "#F6C88E"
INK = "#17212B"


def token_set(value):
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip() and item.strip().lower() != "nan"}


def short_name(value, limit=24):
    value = str(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def two_line_name(value, limit=15):
    """Wrap long material names only at a visually sensible separator."""
    value = str(value)
    if len(value) <= limit:
        return value
    for separator in ("@", "-", "/"):
        position = value.find(separator, max(1, limit - 6))
        if position != -1:
            return value[:position + 1] + "\n" + value[position + 1:]
    return value[:limit] + "\n" + value[limit:]


def style_axes(ax, label_size=17, tick_size=12):
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.tick_params(width=2, labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def axes_43(left=.85, right=.30, top=.28, bottom=.72):
    data_w, data_h = 4.0, 3.0
    fig = plt.figure(figsize=(left + data_w + right, bottom + data_h + top), dpi=240)
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(),
                       data_w / fig.get_figwidth(), data_h / fig.get_figheight()])
    style_axes(ax)
    return fig, ax


def enriched_categories(frame, top_n=20):
    """Estimate transparent top-PAI enrichment for interpretable material features."""
    top_ids = set(frame.head(top_n)["material_id"])
    N = len(frame)
    rows = []
    feature_defs = [
        ("Main category", "main", lambda row: token_set(row["main"])),
        ("Metal node", "metal_node_elements", lambda row: token_set(row["metal_node_elements"])),
        ("Linker family", "Linker family", lambda row: token_set(row["Linker family"])),
        ("Donor / functionality", None, lambda row: {
            label for column, label in [("O_donor", "O donor"), ("N_donor", "N donor"), ("S_donor", "S donor"),
                                         ("carboxyl", "Carboxyl"), ("amino", "Amino"), ("thiol", "Thiol")]
            if str(row[column]).strip().lower() in {"1", "yes", "true"}
        }),
        ("Modification state", None, lambda row: {
            label for column, label in [("post_modification", "Post-modified"), ("composite", "Composite"),
                                         ("magnetic", "Magnetic")]
            if str(row[column]).strip().lower() in {"1", "yes", "true"}
        }),
    ]
    for domain, _, extractor in feature_defs:
        counts_all, counts_top = {}, {}
        for _, row in frame.iterrows():
            for item in extractor(row):
                counts_all[item] = counts_all.get(item, 0) + 1
                if row["material_id"] in top_ids:
                    counts_top[item] = counts_top.get(item, 0) + 1
        for item, all_count in counts_all.items():
            top_count = counts_top.get(item, 0)
            # Haldane correction makes the odds ratio finite for sparse classes.
            odds, p_value = fisher_exact([[top_count, top_n - top_count], [all_count - top_count, N - all_count - top_n + top_count]])
            ratio = (top_count / top_n) / (all_count / N)
            rows.append({"domain": domain, "feature": item, "all_count": all_count,
                         "top20_count": top_count, "enrichment_ratio": ratio,
                         "odds_ratio": odds, "fisher_p": p_value})
    return pd.DataFrame(rows).sort_values(["enrichment_ratio", "top20_count"], ascending=False)


OUT_DIR.mkdir(parents=True, exist_ok=True)
scores = pd.read_csv(SCORES)
scores = scores.sort_values("PAI", ascending=False).reset_index(drop=True)
raw = pd.read_excel(DATA)
materials = raw.drop_duplicates("material_id")
feature_columns = ["material_id", "main", "metal_node_elements", "Linker family", "O_donor", "N_donor", "S_donor",
                   "carboxyl", "amino", "thiol", "post_modification", "composite", "magnetic"]
scores = scores.merge(materials[feature_columns], on="material_id", how="left", suffixes=("", "_db"))

enrichment = enriched_categories(scores, top_n=20)
enrichment.to_csv(OUT_DIR / "Top20_PAI_feature_enrichment.csv", index=False, encoding="utf-8-sig")

# Candidate triad: a balanced lead, a high-W sentinel, and a W-limited control.
balanced = scores.iloc[0]
high_w = scores[scores["W_score"] >= .80].sort_values("PAI", ascending=False).iloc[0]
limited_pool = scores[(scores["W_score"] <= .60) & (scores["C_score"] >= .80)]
w_limited = limited_pool.sort_values("PAI", ascending=False).iloc[0]
selected = pd.DataFrame([
    {"role": "Balanced high-PAI candidate", **balanced.to_dict()},
    {"role": "High-W validation sentinel", **high_w.to_dict()},
    {"role": "W-limited control", **w_limited.to_dict()},
])
selected.to_csv(OUT_DIR / "Experimental_validation_candidate_triad.csv", index=False, encoding="utf-8-sig")

# Figure 6a: C-R design space. Point colour is W and size is observed K.
fig, ax = axes_43(left=.92, right=.68, bottom=.72)
cmap = LinearSegmentedColormap.from_list("water", ["#E4F2EC", "#7FC7AB", "#187A57"])
scatter = ax.scatter(scores["C_score"], scores["R_score"], c=scores["W_score"],
                     s=28 + 85 * scores["K_score_observed_q60_qmax"], cmap=cmap,
                     vmin=0, vmax=1, edgecolors="#315348", linewidths=.6)
for candidate_number, (_, row) in enumerate(selected.iterrows(), start=1):
    ax.scatter(row["C_score"], row["R_score"], s=125, facecolors="none", edgecolors=ORANGE,
               linewidths=2.0, zorder=4)
    ax.text(row["C_score"], row["R_score"], str(candidate_number), ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK, zorder=5)
ax.axvline(.70, color="#6B7280", linestyle="--", linewidth=1)
ax.axhline(.80, color="#6B7280", linestyle="--", linewidth=1)
ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
ax.set_xlabel("Capacity score (C)", fontsize=17, fontweight="bold")
ax.set_ylabel("Regeneration score (R)", fontsize=17, fontweight="bold")
ax.grid(alpha=.16)
candidate_key = [
    "1  Balanced high-PAI: ZIF-8/MS",
    "2  High-W sentinel: Fe3O4-COOH@MIL-125",
    "3  W-limited control: Cu-BTC@STD",
]
for index, entry in enumerate(candidate_key):
    ax.text(.035, .155 - index * .038, entry, transform=ax.transAxes,
            fontsize=6.8, fontweight="bold", color=INK)
cax = fig.add_axes([(0.92 + 4.0 + .16) / fig.get_figwidth(), .72 / fig.get_figheight(),
                    .14 / fig.get_figwidth(), 3.0 / fig.get_figheight()])
cb = fig.colorbar(scatter, cax=cax)
cb.set_label("Water-stability evidence score (W)", fontsize=10, fontweight="bold")
cb.ax.tick_params(labelsize=9, width=1.5)
fig.savefig(OUT_DIR / "Figure6a_C_R_W_design_space.png", dpi=600, facecolor="white")
plt.close(fig)

# Figure 6b: enrichment, filtered to avoid single-observation design rules.
robust = enrichment[(enrichment["all_count"] >= 10) & (enrichment["top20_count"] >= 2)].copy()
robust = robust.sort_values("enrichment_ratio", ascending=False).head(10).iloc[::-1]
fig, ax = axes_43(left=1.88, bottom=.72)
y = np.arange(len(robust))
colors = [MINT_DARK if value >= 1 else ORANGE for value in robust["enrichment_ratio"]]
ax.hlines(y, 0, robust["enrichment_ratio"], color="#BFC8C5", linewidth=2)
ax.scatter(robust["enrichment_ratio"], y, s=55, color=colors, edgecolor="black", linewidth=.5, zorder=3)
ax.axvline(1, color="#6B7280", linestyle="--", linewidth=1.2)
ax.set_yticks(y, [f"{domain}: {feature}" for domain, feature in zip(robust["domain"], robust["feature"])])
ax.set_xlabel("Top-20 PAI enrichment ratio", fontsize=16, fontweight="bold")
ax.tick_params(axis="y", length=0)
for label in ax.get_yticklabels():
    label.set_fontsize(8)
fig.savefig(OUT_DIR / "Figure6b_Top20_feature_enrichment.png", dpi=600, facecolor="white")
plt.close(fig)

# Figure 6c: planned comparative validation. Same C/K/R/W architecture as Figure 5d.
fig, ax = axes_43(left=.85, bottom=1.10)
components = [("C_score", "C", MINT), ("K_score_observed_q60_qmax", "K", "#68B79A"),
              ("R_score", "R", ORANGE_LIGHT), ("W_score", "W", ORANGE)]
x = np.arange(len(selected)); width = .18
for offset, (column, label, color) in enumerate(components):
    ax.bar(x + (offset - 1.5) * width, selected[column], width=width, color=color,
           edgecolor="black", linewidth=.6, label=label)
ax.set_xticks(x, [two_line_name(name, 15) for name in selected["material_original_name"]])
ax.set_ylim(0, 1.17)
ax.set_ylabel("Component score", fontsize=17, fontweight="bold")
ax.tick_params(axis="x", length=0)
for label in ax.get_xticklabels():
    label.set_fontsize(8)
ax.legend(ncol=4, frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(.99, .99))
ax.grid(axis="y", alpha=.16)
fig.savefig(OUT_DIR / "Figure6c_Validation_candidate_triad.png", dpi=600, facecolor="white")
plt.close(fig)

# Figure 6d: an explicit experimental route, not a claim of a new material.
# A horizontal canvas keeps the four pre-specified stages readable in a manuscript panel.
fig = plt.figure(figsize=(9.0, 3.5), dpi=240)
ax = fig.add_axes([.035, .12, .93, .76])
ax.set_axis_off()
steps = [
    ("1", "Reference\nwindow", "pH 5.0 | 298 K\nCMOF 0.2 g/L\nCe = 300 mg/L"),
    ("2", "Three-material\ncomparison", "Balanced high PAI\nHigh-W sentinel\nW-limited control"),
    ("3", "Prospective\nvalidation", "Isotherm | q60/qmax\n3 cycles | PXRD\nMetal leaching"),
    ("4", "Design\ndecision", "Retain a feature domain\nonly after W is\nexperimentally confirmed"),
]
xs = [.025, .275, .525, .775]
for index, (number, heading, body) in enumerate(steps):
    color = MINT if index < 2 else ORANGE_LIGHT
    box = FancyBboxPatch((xs[index], .20), .19, .58, boxstyle="round,pad=.012,rounding_size=.014",
                         linewidth=1.8, edgecolor=INK, facecolor=color, transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(xs[index] + .022, .685, number, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
    ax.text(xs[index] + .022, .565, heading, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=INK,
            va="center")
    ax.text(xs[index] + .022, .345, body, transform=ax.transAxes, fontsize=6.7, color=INK, va="center")
    if index < len(steps) - 1:
        arrow = FancyArrowPatch((xs[index] + .196, .49), (xs[index + 1] - .012, .49),
                                transform=ax.transAxes, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.4, color=INK)
        ax.add_patch(arrow)
fig.savefig(OUT_DIR / "Figure6d_Experimental_design_route.png", dpi=600, facecolor="white")
plt.close(fig)

# A concise research direction that can be used directly in drafting.
top_rules = robust.iloc[::-1].head(5)
rule_text = "\n".join(f"- {row.domain}: {row.feature}; enrichment = {row.enrichment_ratio:.2f}× "
                      f"(top-20 n={int(row.top20_count)}, overall n={int(row.all_count)})." for _, row in top_rules.iterrows())
report = f"""# Post-PAI research direction and validation plan

## What the current dataset supports

The Figure 5 screen contains {len(scores)} materials with usable kinetic curves. Scores are calculated at the fixed reference window: Ce = 300 mg/L, pH = 5.0, 298 K, CMOF = 0.2 g/L, and cycle 3 for R. C is model-predicted and normalised using the full-library P5--P95 range; K is observed q60/qmax; R is model-predicted retention; W is evidence-based water stability.

## Immediate direction

Do not jump directly from the highest PAI material to a new synthesis. First validate whether the PAI ranking remains meaningful when stability is measured prospectively. The three-candidate triad deliberately separates balanced performance, strong stability evidence, and a W-limited comparison:

1. Balanced high-PAI candidate: {balanced.material_original_name} ({balanced.material_id}; PAI={balanced.PAI:.3f}).
2. High-W validation sentinel: {high_w.material_original_name} ({high_w.material_id}; PAI={high_w.PAI:.3f}; W={high_w.W_score:.3f}).
3. W-limited control: {w_limited.material_original_name} ({w_limited.material_id}; PAI={w_limited.PAI:.3f}; W={w_limited.W_score:.3f}).

Run all three under the same pH, temperature and dose. Obtain an isotherm and interpolate qe at Ce = 300 mg/L; measure a full kinetic curve and calculate q60/qmax; run at least three adsorption-desorption cycles; then perform post-use PXRD and quantify metal leaching. This validates both the performance scores and W rather than simply reproducing a single high-qe experiment.

## Data-supported feature signals

The following are enrichment signals, not causal prescriptions. They should be used to form a small design space and only retained after the prospective W test:

{rule_text}

## What should become the next manuscript figure

Use the generated Figure 6 panels as the bridge from Figure 5 to experimental validation: C-R-W design space, robust feature enrichment, the three-candidate comparison, and the pre-specified validation route. A final Figure 7 should only be produced after experiments, showing the new material or selected material against the high-W and W-limited controls using isotherms, kinetics, cycling, PXRD and metal-leaching evidence.
"""
(OUT_DIR / "Post_PAI_research_direction.md").write_text(report, encoding="utf-8")
print("Selected validation candidates:")
print(selected[["role", "material_id", "material_original_name", "PAI", "W_score"]].to_string(index=False))
print("\nRobust enrichment:")
print(robust.iloc[::-1][["domain", "feature", "all_count", "top20_count", "enrichment_ratio", "fisher_p"]].to_string(index=False))
