"""Draw the Figure 6e parent-MOF comparison bar chart.

The plot retains every *Figure 5 eligible* material carrying the Zn + F1 +
M1 labels. CALF-20 is intentionally shown first as the preselected parent
for prospective validation; the remaining materials follow in descending PAI
order.
"""

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


# The eligible pool already enforces W >= 0.725, K >= 0.60 and R >= 0.70.
# It is the single source of truth for Figure 6e.
SOURCE = Path(
    str(PACKAGE_ROOT / 'outputs/Figure5_Formal_Screening_v2/Figure5_Eligible45_PAI_ranking.csv')
)
OUT_DIR = PACKAGE_ROOT / 'outputs/Figure6_Design_Guidance'
OUT_FILE = OUT_DIR / "Figure6e_Zn_F1_M1_PAI_comparison.png"

MINT = "#9ED8C6"
ORANGE = "#F3B66F"
INK = "#111820"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def has_token(value, token):
    """Match a semicolon-separated categorical label exactly."""
    return token in {part.strip() for part in str(value).split(";")}


def display_name(value):
    """Keep long material names readable without shrinking the font."""
    text = str(value).replace(" = ", "\n")
    return "\n".join(textwrap.wrap(text, width=32, break_long_words=False))


def main():
    data = pd.read_csv(SOURCE)
    subset = data.loc[
        (data["main"] == "M1")
        & data["metal_node_elements"].map(lambda x: has_token(x, "Zn"))
        & data["Linker family"].map(lambda x: has_token(x, "F1")),
        ["material_original_name", "PAI"],
    ].copy()

    # IUST-2 was assigned to this subset in error in the source workbook.
    # Exclude it from the Figure 6e parent-candidate comparison only.
    subset = subset.loc[~subset["material_original_name"].eq("IUST-2")].copy()

    if subset.empty:
        raise ValueError("No Zn + F1 + M1 materials were found in the source table.")

    subset = subset.sort_values("PAI", ascending=False, kind="stable")
    calf = subset.loc[subset["material_original_name"].eq("CALF-20")]
    if calf.empty:
        raise ValueError("CALF-20 was not found in the Zn + F1 + M1 subset.")
    subset = pd.concat([calf, subset.loc[~subset.index.isin(calf.index)]], ignore_index=True)
    subset["label"] = subset["material_original_name"].map(display_name)

    # Fixed 4:3 data region; the wider canvas only accommodates material names.
    axes_w, axes_h = 4.0, 3.0
    # Long chemical names need a generous left margin; the plotted data area
    # remains a fixed 4:3 rectangle.
    left, right, top, bottom = 4.80, 0.75, 0.35, 0.70
    fig_w, fig_h = left + axes_w + right, bottom + axes_h + top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=360)
    ax = fig.add_axes([left / fig_w, bottom / fig_h, axes_w / fig_w, axes_h / fig_h])

    colors = [ORANGE] + [MINT] * (len(subset) - 1)
    y = list(range(len(subset)))
    bars = ax.barh(y, subset["PAI"], color=colors, edgecolor=INK, linewidth=2.0, height=0.62)
    ax.invert_yaxis()

    upper = max(0.88, float(subset["PAI"].max()) + 0.10)
    ax.set_xlim(0, upper)
    ax.set_xlabel("PAI score", fontsize=22, fontweight="bold", labelpad=3)
    ax.set_yticks(y, subset["label"])
    ax.tick_params(axis="x", labelsize=17, width=2.0, length=7)
    ax.tick_params(axis="y", labelsize=14, width=0, length=0, pad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color(INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    for bar, score in zip(bars, subset["PAI"]):
        ax.text(
            bar.get_width() + upper * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            ha="left", va="center", fontsize=14, fontweight="bold", color=INK,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=360, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")
    print(subset[["material_original_name", "PAI"]].to_string(index=False))


if __name__ == "__main__":
    main()
