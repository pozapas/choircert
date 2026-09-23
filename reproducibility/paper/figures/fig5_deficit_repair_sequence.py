"""
Figure 5 v3 — "The deficit, and what closes it".

REDESIGN of the dropped 3D Wigner-surface Fig 5. The field is noisy per-cell
binomial coverage, not a smooth surface, so v3 is a 2D diptych partner of Fig 2
in the labeled shared-colorbar heatmap grammar of template_contour_grid.png:

  TOP ROW    (a) marginal / (b) KMeans-8 generic partition / (c) declared
             speed x time partition — three EQUAL raw (hour x speed_bin)
             coverage-deviation heatmaps sharing the speed-limit y-axis and ONE
             shared diverging colorbar (deliberate content-driven adaptation of
             the reference's per-panel-colorbar convention: our three maps encode
             the identical quantity/scale, the reference's rows do not).
  BOTTOM ROW (d) coverage vs speed limit / (e) coverage vs hour — two curve
             panels mirroring the reference's own line-panel style (rows a-c),
             three methods + dashed nominal-0.90 line, one shared legend.

Sparse cells (n < 400) get the exact hatched-wash idiom from fig2_severity_
landscape.py (white facecolor, alpha 0.55, edgecolor SLATE, hatch "////").

Self-contained / standalone: loads the two frozen CSVs via pandas by path
(never inlines row values), resolving the path like sibling fig2/fig4 scripts.
Final home: paper/figures/fig5_deficit_repair_sequence.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42   # camera-ready: no Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# --- house palette (inlined from experiments/figstyle.py) ---
INK = "#1a1a1a"           # near-black structure/ink
HDR_BLUE = "#1f4e79"
ACCENT_TEAL = "#11746c"   # over-coverage (deviation > 0); declared-method curve
CORAL = "#c1443c"         # under-coverage (deviation < 0); marginal curve
GOLD = "#a87c2a"          # KMeans-8 (mondrian) curve
SLATE = "#697784"         # muted secondary / hatch edge
MIST = "#e6eeec"          # diverging-map zero point
SAND = "#f6e0d9"
# Diverging deviation ramp: CORAL (< 0, under) <-> MIST (0) <-> ACCENT_TEAL (> 0).
DIVERGING = [CORAL, "#e8a08a", MIST, "#a9d3cc", ACCENT_TEAL]

# Font family: mirror fig2's family-probe (Helvetica/Arial house sans, DejaVu fallback).
for _fam in ("Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"):
    if any(_fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _fam
        break
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",   # match fig2_severity_landscape's bold axis labels
    "axes.labelpad": 9,           # match fig2's axis-label padding
    "xtick.labelsize": 7.5,       # match fig2's tick label size
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.9,
})

# === DATA SECTOR (file load, not inline literals) ===
def resolve_data_path(name: str) -> Path:
    here = Path(__file__).resolve()
    canonical = here.parent.parent / "figure_src" / "data" / name
    if canonical.exists():
        return canonical
    for p in here.parents:
        cand = p / "paper" / "figure_src" / "data" / name
        if cand.exists():
            return cand
    return canonical

SURFACE_CSV = resolve_data_path("fig05_surface.csv")
WALLS_CSV = resolve_data_path("fig05_walls.csv")

surf = pd.read_csv(SURFACE_CSV)
walls = pd.read_csv(WALLS_CSV)

NOMINAL = 0.90          # target coverage (alpha = 0.10)
MIN_N = 400             # cells with fewer records -> masked "sparse"
VLIM = 0.25             # symmetric diverging colorbar limit; deepest value saturates
# === END DATA SECTOR ===

# ---- build the three (speed_bin x hour) deviation grids ----
hours = np.arange(0, 24)
speeds = np.arange(15, 85, 5)                  # bin floors 15..80
h_edges = np.arange(-0.5, 24.5, 1.0)
s_edges = np.arange(15, 90, 5)                 # 15..85, each bin spans [floor, floor+5)

# Titles carry the narrative from the plan; single-row where the panel width
# allows it (checked by the floor self-check below) -- (a)'s title alone is too
# long for its narrow panel and collides with (b) at one row, so it keeps a
# two-line wrap; (b) and (c) fit in one row.
METHOD_COLS = {
    "(a)": ("cov_marginal", "Marginal deficit on\nthe severity ridge"),
    "(b)": ("cov_mondrian", "A generic partition narrows it"),
    "(c)": ("cov_declared", "Declared bands close the ridge"),
}

N_grid = surf.pivot(index="speed_bin", columns="hour", values="n").reindex(
    index=speeds, columns=hours).values
sparse = N_grid < MIN_N

def dev_grid(col):
    g = surf.pivot(index="speed_bin", columns="hour", values=col).reindex(
        index=speeds, columns=hours).values
    return g - NOMINAL

div_cmap = LinearSegmentedColormap.from_list("coral_mist_teal", DIVERGING)
norm = Normalize(vmin=-VLIM, vmax=VLIM)

# ---------------------------------------------------------------- figure/layout
fig = plt.figure(figsize=(7.0, 5.6), dpi=300)

# Manual axes placement (like fig2) for precise control of the shared colorbar
# and the 3-over-2 composition. Top row: 3 equal heatmaps + one shared colorbar.
LEFT = 0.068
CB_X = 0.878          # colorbar left edge
MAP_RIGHT = 0.850     # right edge of the 3-map band (leaves a gutter before cbar)
WGAP = 0.016          # inter-map gap (tight, camera-ready)
map_w = (MAP_RIGHT - LEFT - 2 * WGAP) / 3.0
TOP_Y0, TOP_Y1 = 0.585, 0.905     # top-row axes bottom/top
map_h = TOP_Y1 - TOP_Y0

map_axes = []
for i in range(3):
    x0 = LEFT + i * (map_w + WGAP)
    ax = fig.add_axes([x0, TOP_Y0, map_w, map_h])
    map_axes.append(ax)

pcm = None
for i, (letter, (col, title)) in enumerate(METHOD_COLS.items()):
    ax = map_axes[i]
    Z = dev_grid(col)
    pcm = ax.pcolormesh(h_edges, s_edges, Z, cmap=div_cmap, norm=norm,
                        shading="flat", rasterized=True, zorder=1)
    # sparse cells: translucent white wash + diagonal hatch (fig2 idiom exactly)
    if sparse.any():
        ys_m, xs_m = np.where(sparse)
        for yi, xi in zip(ys_m, xs_m):
            ax.add_patch(Rectangle(
                (h_edges[xi], s_edges[yi]),
                h_edges[xi + 1] - h_edges[xi], s_edges[yi + 1] - s_edges[yi],
                facecolor="white", alpha=0.55, edgecolor=SLATE, linewidth=0.3,
                hatch="////", zorder=2))
    # axes cosmetics
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(15, 85)
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_xticklabels([f"{h:02d}" for h in np.arange(0, 24, 6)])
    ax.set_yticks(np.arange(20, 85, 15))
    ax.tick_params(length=3, width=0.8, color=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.9)
    ax.set_xlabel("hour of day")
    if i == 0:
        ax.set_ylabel("posted speed limit (mph)")
    else:
        ax.set_yticklabels([])   # shared y-axis; only (a) carries labels
    # panel letter + title, top-left above the axes (two lines, tight to panel width)
    ax.annotate(f"{letter} {title}", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.0, fontweight="bold",
                color=INK, linespacing=1.15)

# ---- deepest-pocket callout on (a): hour=3, speed_bin=75, cov_marginal=0.338 ----
map_axes[0].annotate(
    "deepest pocket\n0.34 @ 75 mph, 03h",
    xy=(3, 77.5), xycoords="data",
    xytext=(6.5, 34), textcoords="data",
    fontsize=6.4, color=INK, ha="left", va="center", zorder=7,
    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7,
                    shrinkA=1.5, shrinkB=3.0))

# ---- shared diverging colorbar (right of the top row), "min" extend arrow ----
cax = fig.add_axes([CB_X, TOP_Y0, 0.018, map_h])
cb = fig.colorbar(pcm, cax=cax, extend="min")
cb.set_label("coverage $-$ 0.90", fontsize=7.8, color=INK)
cb.set_ticks([-0.25, -0.12, 0.0, 0.12, 0.25])
cb.set_ticklabels(["$-$0.25", "$-$0.12", "0.00", "+0.12", "+0.25"])
cb.ax.tick_params(labelsize=6.6, length=3, width=0.8, color=INK)
cb.outline.set_edgecolor(INK)
cb.outline.set_linewidth(0.9)

# ================================================================ BOTTOM ROW
# Two curve panels, slimmer than the maps, mirroring the reference's line-panel
# style. (d) coverage vs speed limit, (e) coverage vs hour. Shared legend.
BOT_Y0, BOT_Y1 = 0.095, 0.420
bot_h = BOT_Y1 - BOT_Y0
BGAP = 0.095          # extra gap for the second panel's y-tick labels
bot_w = (MAP_RIGHT - LEFT - BGAP) / 2.0

ax_d = fig.add_axes([LEFT, BOT_Y0, bot_w, bot_h])
ax_e = fig.add_axes([LEFT + bot_w + BGAP, BOT_Y0, bot_w, bot_h])

# line style spec: marginal CORAL solid, mondrian thin GOLD, declared TEAL dashed
LINES = [
    ("cov_marginal", CORAL, "-", 1.6, "Marginal"),
    ("cov_mondrian", GOLD, "-", 1.1, "KMeans-8 partition"),
    ("cov_declared", ACCENT_TEAL, "--", 1.6, "Declared bands"),
]

def draw_curves(ax, sub, xcol):
    x = sub[xcol].values
    for col, c, ls, lw, _lab in LINES:
        ax.plot(x, sub[col].values, color=c, linestyle=ls, linewidth=lw,
                solid_capstyle="round", zorder=3)
    ax.axhline(NOMINAL, color=INK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=2)
    # All-4 box spines mirror the reference's own line-panel frame (rows a-c).
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8, color=INK)
    ax.set_ylabel("empirical coverage")

# (d) coverage vs speed limit
sp = walls[walls["axis"] == "speed_limit"].sort_values("bin")
draw_curves(ax_d, sp, "bin")
ax_d.set_xlim(sp["bin"].min() - 2, sp["bin"].max() + 2)
ax_d.set_xticks(np.arange(20, 85, 15))
ax_d.set_xlabel("posted speed limit (mph)")
ax_d.set_ylim(0.30, 1.02)
ax_d.set_yticks([0.4, 0.6, 0.8, 1.0])
ax_d.annotate("(d) Only declared bands hold the high-speed tail",
              xy=(0.0, 1.0), xycoords="axes fraction",
              xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
              fontsize=7.0, fontweight="bold", color=INK)
# 75 mph pooled callout: marginal 0.611 -> KMeans-8 0.692
ax_d.annotate("75 mph\n0.61 $\\to$ 0.69", xy=(75, 0.611), xycoords="data",
              xytext=(58, 0.44), textcoords="data", fontsize=6.4, color=INK,
              ha="left", va="center", zorder=6,
              arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=0.7,
                              shrinkA=1.5, shrinkB=3.0))

# (e) coverage vs hour
hr = walls[walls["axis"] == "hour"].sort_values("bin")
draw_curves(ax_e, hr, "bin")
ax_e.set_xlim(-0.5, 23.5)
ax_e.set_xticks(np.arange(0, 24, 6))
ax_e.set_xticklabels([f"{h:02d}" for h in np.arange(0, 24, 6)])
ax_e.set_xlabel("hour of day")
ax_e.set_ylim(0.55, 0.97)
ax_e.set_yticks([0.6, 0.7, 0.8, 0.9])
ax_e.annotate("(e) Declared bands hold the night; the residue\nmoves to the band edges",
              xy=(0.0, 1.0), xycoords="axes fraction",
              xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
              fontsize=7.0, fontweight="bold", color=INK, linespacing=1.15)

# ---- one shared legend (mirrors reference's frameless line-end legend) ----
legend_handles = [
    Line2D([0], [0], color=CORAL, linestyle="-", linewidth=1.6, label="Marginal"),
    Line2D([0], [0], color=GOLD, linestyle="-", linewidth=1.1, label="KMeans-8 partition"),
    Line2D([0], [0], color=ACCENT_TEAL, linestyle="--", linewidth=1.6, label="Declared bands"),
    Line2D([0], [0], color=INK, linestyle=(0, (4, 3)), linewidth=0.9, label="nominal 0.90"),
]
leg = ax_d.legend(handles=legend_handles, loc="lower left",
                  bbox_to_anchor=(0.0, -0.02), frameon=False, ncol=2,
                  handlelength=1.7, handletextpad=0.4, columnspacing=1.1,
                  borderpad=0.3, labelspacing=0.35, fontsize=6.6)
leg.set_zorder(6)

# ===================== FLOOR SELF-CHECK =====================
import matplotlib as _mpl


def _texts(fig):
    out = []
    for a in fig.axes:
        for t in a.get_xticklabels() + a.get_yticklabels():
            if t.get_text() and t.get_visible():
                out.append(("tick", t))
        for lbl in (a.title, a.xaxis.label, a.yaxis.label):
            if lbl.get_text():
                out.append(("label", lbl))
        for ch in a.get_children():
            if isinstance(ch, _mpl.text.Annotation) and ch.get_text():
                out.append(("annot", ch))
    return out


def floor_selfcheck(fig, path):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = _texts(fig)
    bb = [(k, t, t.get_window_extent(renderer=r)) for k, t in items]
    viol = []
    for i, (ka, ta, ba) in enumerate(bb):
        for kb, tb, bbx in bb[i + 1:]:
            if ba.overlaps(bbx):
                viol.append(f"OVERLAP {ka}('{ta.get_text()}') <-> {kb}('{tb.get_text()}')")
    fbox = fig.bbox
    clip = []
    for k, t, b in bb:
        if not (fbox.contains(b.x0, b.y0) and fbox.contains(b.x1, b.y1)):
            clip.append(f"CLIP {k}('{t.get_text()}') bbox=({b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f})")
    status = "PASS" if not viol and not clip else "FAIL"
    lines = [f"FLOOR SELF-CHECK: {status}",
             f"text items checked: {len(bb)}",
             f"overlap violations: {len(viol)}"] + viol + \
            [f"clip violations: {len(clip)}"] + clip
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return status


OUT = Path(__file__).resolve().parent
fig.savefig(OUT / "fig5_deficit_repair_sequence.png", dpi=300)
fig.savefig(OUT / "fig5_deficit_repair_sequence.pdf")
floor_selfcheck(fig, OUT / "floor_selfcheck_iter2.txt")
plt.close(fig)
