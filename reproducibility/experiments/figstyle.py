"""Shared Nature-style figure identity for the CHOIR paper.

One palette, one typographic system, used by every figure so the paper reads as a
designed whole. Palette is aligned to the stored illustration colors (HdrBlue,
AccentTeal) so figures and tables share a visual language.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Core palette (RGB matched to main.tex \definecolor). Ink is near-black, not pure.
INK = "#1a1a1a"
HDR_BLUE = "#1f4e79"      # deep blue, structural
ACCENT_TEAL = "#11746c"   # teal, the CHOIR signature
CORAL = "#c1443c"         # failure / undercoverage
GOLD = "#a87c2a"          # secondary highlight
GREEN = "#00695c"         # conservation / success
SLATE = "#697784"         # muted rule / secondary text
MIST = "#e6eeec"          # light teal fill
SAND = "#f6e0d9"          # light coral fill

# Sequential (coverage) and diverging (deviation from nominal) ramps.
SEQ_TEAL = ["#e6eeec", "#a9d3cc", "#5fb0a5", "#22867a", "#0c5a51"]
DIVERGING = [CORAL, "#e8a08a", "#f2e4d6", "#a9d3cc", ACCENT_TEAL]

# Categorical cycle for models/methods (colorblind-aware, harmonized).
CATEGORICAL = [HDR_BLUE, ACCENT_TEAL, CORAL, GOLD, GREEN, SLATE, "#7b6d8d"]


def apply():
    """Install the house style. Call once at the top of every figure script."""
    for family in ("Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros"):
        if any(family in f.name for f in font_manager.fontManager.ttflist):
            mpl.rcParams["font.family"] = family
            break
    mpl.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e9edf0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def nominal_line(ax, y, label=r"nominal $1-\alpha$"):
    """A consistent reference line for the target coverage level."""
    ax.axhline(y, color=INK, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.8, color=SLATE, style="italic")


def finalize(fig, path):
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
