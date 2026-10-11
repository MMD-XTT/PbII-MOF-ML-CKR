"""Create a separate, publication-oriented Figure 5 screening set.

This script only reads the archived candidate-score table copied into its own
folder; it does not alter the earlier Figure 5 files.
"""

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = PACKAGE_ROOT / 'outputs/Figure5_Formal_Screening_v2'
DATA = OUT_DIR / "Figure5_PAI_reference_Ce300_pH5_CMOF0p2_candidate_data.csv"

MINT = "#9DD6C3"
MINT_DARK = "#68B79A"
ORANGE = "#EE9E58"
ORANGE_LIGHT = "#F6C88E"
INK = "#17212B"
R_MIN = .70
K_MIN = .60
W_MIN = .725


def style_axes(ax, tick_size=13):
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.tick_params(width=2, labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")


def make_axes(data_w=4.0, data_h=3.0, left=.85, right=.25, bottom=.70, top=.25):
    fig = plt.figure(figsize=(left + data_w + right, bottom + data_h + top), dpi=240)
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(),
                       data_w / fig.get_figwidth(), data_h / fig.get_figheight()])
    style_axes(ax)
    return fig, ax


def two_line_name(name, limit=16):
    name = str(name)
    if len(name) <= limit:
        return name
    for char in ("@", "/", "-"):
        positions = [i for i, item in enumerate(name) if item == char]
        if positions:
            cut = min(positions, key=lambda position: abs(position - limit))
            if 4 <= cut <= len(name) - 4:
                return name[:cut + 1] + "\n" + name[cut + 1:]
    return name[:limit] + "\n" + name[limit:]


def component_name(name, line_limit=13):
    """Use two or three compact lines for the dense Top-10 component panel."""
    name = str(name)
    if len(name) <= line_limit:
        return name
    chunks = []
    remaining = name
    while len(remaining) > line_limit and len(chunks) < 2:
        separators = [position for position, char in enumerate(remaining[: line_limit + 5]) if char in "@/-"]
        cut = separators[-1] + 1 if separators else line_limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    chunks.append(remaining)
    return "\n".join(chunks)


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=600, facecolor="white")
    plt.close(fig)


scores = pd.read_csv(DATA).sort_values("PAI", ascending=False).reset_index(drop=True)
eligible_mask = (
    (scores["W_score"] >= W_MIN)
    & (scores["K_score_observed_q60_qmax"] >= K_MIN)
    & (scores["R_score"] >= R_MIN)
    & (scores["PAI"] > 0)
)
eligible = scores.loc[eligible_mask].sort_values("PAI", ascending=False).reset_index(drop=True)
top10 = eligible.head(10).copy()

assert len(scores) == 176, f"Expected 176 scored materials, found {len(scores)}"
assert len(eligible) == 44, f"Expected 44 PAI-rankable materials, found {len(eligible)}"

# Figure 5a: all scored materials; eligibility and final Top-10 are layered over it.
fig, ax = make_axes(left=.92, right=.68, bottom=.72)
point_size = 24 + 92 * scores["K_score_observed_q60_qmax"]
water_cmap = LinearSegmentedColormap.from_list("water", ["#E4F2EC", "#7FC7AB", "#187A57"])
scatter = ax.scatter(scores["C_score"], scores["R_score"], s=point_size, c=scores["W_score"],
                     cmap=water_cmap, vmin=0, vmax=1, edgecolors="#34534B", linewidths=.55,
                     alpha=.30, zorder=1)
ax.scatter(eligible["C_score"], eligible["R_score"],
           s=24 + 92 * eligible["K_score_observed_q60_qmax"], c=eligible["W_score"],
           cmap=water_cmap, vmin=0, vmax=1, edgecolors=INK, linewidths=1.15, alpha=1.0, zorder=2)
ax.scatter(top10["C_score"], top10["R_score"],
           s=24 + 92 * top10["K_score_observed_q60_qmax"], facecolors="none",
           edgecolors=ORANGE, linewidths=2.25, zorder=3)
ax.axhline(R_MIN, color="#6B7280", linestyle="--", linewidth=1.4, zorder=0)
ax.text(.02, R_MIN + .014, "R = 0.70", fontsize=8, fontweight="bold", color="#4B5563")
ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
ax.set_xlabel("Capacity score (C)", fontsize=17, fontweight="bold")
ax.set_ylabel("Regeneration score (R)", fontsize=17, fontweight="bold")
ax.grid(alpha=.16, linewidth=.8)

size_values = [.60, .80, 1.00]
size_handles = [ax.scatter([], [], s=24 + 92 * value, color="#A7B9B3", edgecolor="#34534B", linewidth=.6)
                for value in size_values]
size_legend = ax.legend(size_handles, [f"{value:.2f}" for value in size_values],
                        title="Kinetic score (K)", frameon=False, fontsize=7.5, title_fontsize=8,
                        loc="lower left", borderpad=.2, handletextpad=.6, labelspacing=.65)
ax.add_artist(size_legend)
all_handle = ax.scatter([], [], s=42, color="#8AB7A7", edgecolor="#34534B", alpha=.45)
eligible_handle = ax.scatter([], [], s=42, color="#8AB7A7", edgecolor=INK, linewidth=1.15)
top_handle = ax.scatter([], [], s=42, facecolors="none", edgecolor=ORANGE, linewidth=2.2)
ax.legend([all_handle, eligible_handle, top_handle],
          ["Scored materials (N = 176)", "Eligible pool (N = 44)", "Final Top 10"],
          frameon=False, fontsize=7.3, loc="lower right", bbox_to_anchor=(.99, .02), handletextpad=.45)

cax = fig.add_axes([(0.92 + 4.0 + .16) / fig.get_figwidth(), .72 / fig.get_figheight(),
                    .14 / fig.get_figwidth(), 3.0 / fig.get_figheight()])
colorbar = fig.colorbar(scatter, cax=cax)
colorbar.set_label("Water-stability evidence score (W)", fontsize=10, fontweight="bold")
colorbar.ax.axhline(W_MIN, color=ORANGE, linewidth=2.0)
colorbar.ax.text(1.60, W_MIN, "0.725", transform=colorbar.ax.get_yaxis_transform(),
                 color=ORANGE, fontsize=8, fontweight="bold", va="center")
colorbar.ax.tick_params(labelsize=9, width=1.5)
save(fig, "Figure5a_Formal_screening_space.png")

# Figure 5b: four components explaining the final Top-10 PAI set.
fig, ax = make_axes(data_w=4.0, data_h=3.0, left=.85, bottom=2.05, top=.28)
x = np.arange(len(top10)); width = .19
components = (("C_score", "C", MINT), ("K_score_observed_q60_qmax", "K", MINT_DARK),
              ("R_score", "R", ORANGE_LIGHT), ("W_score", "W", ORANGE))
for offset, (column, label, colour) in enumerate(components):
    ax.bar(x + (offset - 1.5) * width, top10[column], width=width, color=colour,
           edgecolor="black", linewidth=.45, label=label)
ax.set_xticks(x, [component_name(name) for name in top10["material_original_name"]])
ax.set_ylim(0, 1.16)
ax.set_ylabel("Component score", fontsize=17, fontweight="bold")
ax.tick_params(axis="x", length=0)
for label in ax.get_xticklabels():
    label.set_fontsize(4.5)
    label.set_rotation(45)
    label.set_ha("right")
ax.legend(ncol=4, frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(.985, .985))
ax.grid(axis="y", alpha=.16, linewidth=.8)
save(fig, "Figure5b_Top10_component_profiles.png")

# Figure 5c: all 45 eligible candidates, ranked by PAI. Top-10 is a design-set cut, not a PAI threshold.
fig, ax = make_axes(data_w=10.0, data_h=3.0, left=.85, right=.20, bottom=3.10, top=.28)
x = np.arange(len(eligible))
colours = [ORANGE if index < 10 else MINT for index in range(len(eligible))]
ax.bar(x, eligible["PAI"], color=colours, edgecolor="black", linewidth=.38, width=.72)
ax.axvline(9.5, color="#6B7280", linestyle="--", linewidth=1.35)
ax.text(9, eligible.loc[9, "PAI"] + .025, "Top 10", color="#4B5563", fontsize=8,
        fontweight="bold", ha="center", va="bottom")
ax.set_xlim(-.65, len(eligible) - .35)
ax.set_ylim(0, 1.0)
ax.set_ylabel("PAI", fontsize=17, fontweight="bold")
ax.set_xlabel("Eligible materials ranked by PAI", fontsize=17, fontweight="bold")
ax.set_xticks(x, eligible["material_original_name"])
ax.tick_params(axis="x", length=0, pad=2)
for label in ax.get_xticklabels():
    label.set_rotation(90)
    label.set_ha("center")
    label.set_fontsize(7.1)
ax.grid(axis="y", alpha=.16, linewidth=.8)
save(fig, "Figure5c_Eligible45_PAI_ranking.png")

rule = f"""# Figure 5 formal screening rule

Scored pool: N = {len(scores)} materials with C, K, R and W available.

Eligibility gate:
- W >= {W_MIN:.3f} (E3/E4 water-stability evidence threshold)
- K >= {K_MIN:.2f}
- R >= {R_MIN:.2f}

Eligible pool: N = {len(eligible)}.

Final design set: the ten highest PAI materials in the eligible pool. The dashed divider in Figure 5c is the Top-10 selection boundary, not a universal PAI threshold.
"""
(OUT_DIR / "Figure5_formal_screening_rule.md").write_text(rule, encoding="utf-8")
eligible.to_csv(OUT_DIR / "Figure5_Eligible45_PAI_ranking.csv", index=False, encoding="utf-8-sig")
top10.to_csv(OUT_DIR / "Figure5_FinalTop10_PAI_ranking.csv", index=False, encoding="utf-8-sig")
print(f"Created Figure 5 formal screening set: N={len(scores)}, eligible={len(eligible)}, top10={len(top10)}")
