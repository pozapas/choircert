"""
Figure 4 -- reporting-channel scenario sensitivity for prediction-set width.

The figure reports a conditional decomposition for a finite grid of declared channel
scenarios. It does not display an identified interval or an optimized bound over a
continuous admissible set.

One horizontal bar per declared stratum, width in KABCO categories (1..5):
  - solid CORAL  [1, Nstar_R5]        : excess implied by the evaluated R=5 scenario
  - hatched CORAL[Nstar_R5, Nstar_R1] : range to the selection-invariance scenario
  - teal wash    [Nstar_R1, achieved] : remaining gap to achieved base width
  - achieved marker + full-scale (5) reference

Numbers are derived summaries rather than raw rows. The scenario floors come from
experiments/e9_channel_floor.py at alpha 0.10. That script solves the latent-composition
problem exactly over the declared finite channel set and reported-share rounding
intervals. The achieved value is the smallest base Mondrian width among the seven models.
Regenerate the audit and update the four rows below when the scenario set changes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# --- house palette (inlined from experiments/figstyle.py) ---
INK = "#1a1a1a"
ACCENT_TEAL = "#11746c"
CORAL = "#c1443c"
GOLD = "#a87c2a"
SLATE = "#697784"
MIST = "#e6eeec"
SAND = "#f6e0d9"
# recoverable-segment green: a step darker than seqteal3 (#5fb0a5), toward the
# signature teal seqteal4 (#11746c), for a firmer read
SEQTEAL3 = "#3d938a"

for _fam in ("Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"):
    if any(_fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _fam
        break
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.labelweight": "bold", "axes.labelpad": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.0,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK, "axes.linewidth": 0.9,
})

# stratum : (Nstar_lo=R=5 floor, Nstar_hi=R=1 floor, achieved, display label, share label)
# Solid shows the minimum evaluated R=5 scenario value. Hatched extends to the minimum
# evaluated selection-invariance value. Numbers come from e9_channel_floor.py.
ROWS = [
    ("motorcycle",      1.03, 1.64, 3.55, "Motorcyclist",     None),
    ("unrestrained",    0.96, 1.36, 3.58, "Unrestrained",     None),
    ("rural_highspeed", 0.92, 1.00, 1.95, "Rural high-speed", None),
    ("baseline",        0.91, 0.97, 1.40, "Baseline",         None),
]


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    bar_h = 0.52
    ys = np.arange(len(ROWS))

    for i, (_key, lo, hi, ach, _lab, _share) in enumerate(ROWS):
        y = ys[i]
        yb = y - bar_h / 2
        # (1) mandatory category [0, 1] that any nonempty set carries
        ax.add_patch(Rectangle((0, yb), 1.0, bar_h,
                               facecolor="#d9d9d9", edgecolor=INK, linewidth=0.7, zorder=3))
        # (2) channel-imposed excess above 1: solid to the heavy-selection floor (lo, R=5),
        #     hatched out to the invariance floor (hi, R=1). Nonzero only where the floor > 1.
        ex_lo, ex_hi = max(1.0, lo), max(1.0, hi)
        if ex_lo > 1.0:
            ax.add_patch(Rectangle((1.0, yb), ex_lo - 1.0, bar_h,
                                   facecolor=CORAL, edgecolor=INK, linewidth=0.7, zorder=3))
        if ex_hi > ex_lo:
            ax.add_patch(Rectangle((ex_lo, yb), ex_hi - ex_lo, bar_h,
                                   facecolor=SAND, edgecolor=CORAL, linewidth=0.7,
                                   hatch="////", zorder=3))
        # (3) recoverable remainder [max(1, hi), achieved]
        if ach > ex_hi:
            ax.add_patch(Rectangle((ex_hi, yb), ach - ex_hi, bar_h,
                                   facecolor=SEQTEAL3, edgecolor=ACCENT_TEAL, linewidth=0.7, zorder=2))
        # achieved width marker
        ax.plot([ach, ach], [yb - 0.06, y + bar_h / 2 + 0.06],
                color=INK, linewidth=1.6, zorder=5)

    # full-scale reference (K=5) and minimum-width reference (1)
    ax.axvline(5.0, color=SLATE, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(5.0, len(ROWS) - 0.35, "full scale", ha="right", va="bottom",
            color=SLATE, fontsize=6.8, rotation=90)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[4] for r in ROWS])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlim(0, 5.25)
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("mean prediction-set width (KABCO categories), $\\alpha=0.10$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        Patch(facecolor="#d9d9d9", edgecolor=INK, label="one category (mandatory)"),
        Patch(facecolor=CORAL, edgecolor=INK, label="scenario excess ($R{=}5$)"),
        Patch(facecolor=SAND, edgecolor=CORAL, hatch="////",
              label="scenario range ($R{=}5$ to $1$)"),
        Patch(facecolor=SEQTEAL3, edgecolor=ACCENT_TEAL, label="remaining width gap"),
        Line2D([0], [0], color=INK, linewidth=1.6, label="achieved base width"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.995, 0.985),
              frameon=True, framealpha=0.95, edgecolor=SLATE, ncol=2, borderpad=0.6,
              columnspacing=1.4, handlelength=1.6)

    fig.tight_layout()
    out = Path(__file__).resolve().parent
    fig.savefig(out / "fig4_channel_floor.pdf", bbox_inches="tight")
    fig.savefig(out / "fig4_channel_floor.png", dpi=400, bbox_inches="tight")
    print("wrote fig4_channel_floor.pdf / .png")


if __name__ == "__main__":
    main()
