"""Figure 4 — "The heterogeneity atlas".

Style is transferred from a single-cell UMAP velocity 4-panel reference: a 2x2
grid of the SAME embedding recolored per panel, floating (no spines / ticks /
gridlines), bold noun-phrase panel titles, a single shared UMAP axis-arrow
glyph, in-plane bold white-haloed cluster labels, and slim right-edge colorbars.
Content + palette are OURS (the house Nature-style identity).

Self-contained / standalone: loads the frozen parquet via pandas (never inlines
row values), resolving the path relative to this file exactly like the sibling
fig2/fig3 scripts. Final home: paper/figures/fig4_heterogeneity_atlas.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42          # camera-ready: no Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# --- house typography (mirrors experiments/figstyle.py; inlined to stay standalone) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial",
                                   "TeX Gyre Heros", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
})

# --- house palette (from experiments/figstyle.py) ---
INK = "#1a1a1a"
HDR_BLUE = "#1f4e79"
ACCENT_TEAL = "#11746c"
CORAL = "#c1443c"
GOLD = "#a87c2a"
GREEN = "#00695c"
SLATE = "#697784"
MIST = "#e6eeec"
SAND = "#f6e0d9"
SEQ_TEAL = ["#e6eeec", "#a9d3cc", "#5fb0a5", "#22867a", "#0c5a51"]
# Teal-coral DIVERGING ramp (figstyle.py's certification-field ramp): coral =
# class-conditional sets WIDER than marginal (cost), teal = narrower (gain).
DIVERGING = [CORAL, "#e8a08a", "#f2e4d6", "#a9d3cc", ACCENT_TEAL]
# CATEGORICAL from figstyle.py lists only 7 hues; two collisions verified by
# redmean perceptual color distance (the pairwise-closest metric, not eyeballed)
# and fixed:
#  - Class 4 was GREEN (#00695c), redmean 25.8 from ACCENT_TEAL (class 1,
#    #11746c) -- both read as "the dark teal one" side by side. Replaced with a
#    yellow-leaning olive (#5b8c3a); redmean to every other class now >85.
#  - Class 6's original muted purple (#7b6d8d, also figstyle's plate-diagram
#    accent, unrelated to this figure) was redmean 37.4 from SLATE (class 5,
#    #697784) -- the single closest pair in the whole 8-color set, closer even
#    than the pre-fix class 1/4 collision. Replaced with a more saturated
#    orchid (#8b5fa3); redmean to every other class now >85.
CLASS4_OLIVE = "#5b8c3a"
CLASS6_ORCHID = "#8b5fa3"
CAT8 = [HDR_BLUE, ACCENT_TEAL, CORAL, GOLD, CLASS4_OLIVE, SLATE, CLASS6_ORCHID, "#a86a72"]

# === DATA SECTOR (file load, not inline literals) ===
# Canonical location paper/figures/<name>.py -> parent.parent = paper/, so
# parent.parent/figure_src/data resolves the frozen parquet (mirrors fig2/fig3).
# In the figmirror workdir that path won't exist, so walk up to find it.
def resolve_data_path() -> Path:
    here = Path(__file__).resolve()
    canonical = here.parent.parent / "figure_src" / "data" / "fig04_embedding.parquet"
    if canonical.exists():
        return canonical
    for p in here.parents:
        cand = p / "paper" / "figure_src" / "data" / "fig04_embedding.parquet"
        if cand.exists():
            return cand
    return canonical

DATA_PATH = resolve_data_path()
df = pd.read_parquet(DATA_PATH)
# === END DATA SECTOR ===

# Shared embedding extent (all four panels share u1/u2 axes, per the reference's
# recurring point-cloud geometry). Small pad around the data range.
U1 = df["u1"].to_numpy()
U2 = df["u2"].to_numpy()
# Clip the shared extent to the 1st/99th percentile (+ small pad) so the main
# atlas fills the panels like the reference, instead of a few far-flung outlier
# islands stretching the frame and leaving every panel's corners empty. Same
# XLIM/YLIM on all four panels -> shared silhouette preserved.
x1, x99 = np.percentile(U1, [1, 99])
y1, y99 = np.percentile(U2, [1, 99])
XLIM = (x1 - 1.2, x99 + 1.2)
YLIM = (y1 - 1.2, y99 + 1.2)

PT_S = 1.6        # marker size for the 120k-point cloud
PT_A = 0.45       # marker alpha
RASTER = True
# Shared hexbin grid for all three hex-aggregated panels (a, b, d): at gs=150
# each hex is ~4px at this dpi, close to the reference's point texture.
HEX_GS = 150


def strip_axes(ax):
    """Floating-embedding look: no spines, ticks, grid, or numeric labels."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_heading(host_ax, letter, title, letter_pos, letter_transform):
    """Panel letter + LEFT-aligned title as one heading line. The title is
    xycoords-CHAINED to the letter artist's own rendered bbox (xy=(1,0) on the
    letter, offset 4pt) so it always starts right after the letter's actual
    width at draw time -- no manual text-width measurement/estimate needed,
    and it stays left-aligned regardless of letter_transform/letter_pos, which
    control where the whole (letter+title) unit begins.
    """
    letter_txt = host_ax.text(*letter_pos, letter, transform=letter_transform,
                              fontsize=11, fontweight="bold", ha="left",
                              va="bottom", color=INK)
    host_ax.annotate(title, xy=(1, 0), xycoords=letter_txt,
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=9.0, fontweight="bold", ha="left", va="bottom",
                     color=INK)
    return letter_txt


# ---------------------------------------------------------------- figure/layout
fig = plt.figure(figsize=(7.0, 5.6), dpi=300)
axes = np.empty((2, 2), dtype=object)
for i in range(2):
    for j in range(2):
        axes[i, j] = fig.add_subplot(2, 2, i * 2 + j + 1)
fig.subplots_adjust(left=0.085, right=0.90, top=0.925, bottom=0.11,
                    wspace=0.07, hspace=0.19)
for ax in axes.ravel():
    strip_axes(ax)

# Column-1 (a)/(c) headings are anchored in FIGURE coordinates at the class
# legend's own left edge (bbox_to_anchor x=0.008, set below), so the two
# headings and the legend all share one left margin. y is each row's axes
# top edge (from the actual layout, not hand-tuned) plus the same ~1.5%
# axes-height gap the original ax.transAxes-based letter used (y=1.015).
LEFT_X = 0.008
_row1_y0, _row1_y1 = axes[0, 0].get_position().y0, axes[0, 0].get_position().y1
_row2_y0, _row2_y1 = axes[1, 0].get_position().y0, axes[1, 0].get_position().y1
ROW1_HEAD_Y = _row1_y0 + 1.015 * (_row1_y1 - _row1_y0)
ROW2_HEAD_Y = _row2_y0 + 1.015 * (_row2_y1 - _row2_y0)

# ================================================================ Panel (a)
# Declared KMeans-8 partition. FIX (iter4): the raw 8-color alpha-scatter
# rendered as salt-and-pepper wherever classes' point clouds interleave in the
# main landmass, visually contradicting a "distinct territory" claim. Switched
# to the SAME hexbin-aggregation family already used for (b)/(d): color each
# hex by its MAJORITY class (mode, not mean -- these are categorical labels).
# This trades local-mixture visibility for a clean territorial read, so the
# title is downgraded to the claim the hexes actually support: the partition
# ORGANIZES space (broad regions dominated by one class), not that it holds
# "distinct" (i.e. non-overlapping) territory, which the pre-fix scatter
# itself refuted.
ax_a = axes[0, 0]
kmeans_arr = df["kmeans8"].to_numpy().astype(int)


def majority_mode(arr):
    """Fast integer mode via bincount (arr is always in 0..7 here). hexbin
    passes each cell's accumulator as a plain Python list, not an ndarray."""
    return np.argmax(np.bincount(np.asarray(arr, dtype=int), minlength=8))


cmap_cat = ListedColormap(CAT8)
norm_cat = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), cmap_cat.N)
hb_a = ax_a.hexbin(U1, U2, C=kmeans_arr, reduce_C_function=majority_mode,
                   gridsize=HEX_GS, cmap=cmap_cat, norm=norm_cat, mincnt=2,
                   linewidths=0.0, extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]),
                   rasterized=RASTER, zorder=2)

# Numbered-dot legend for all 8 classes. Out-of-axes placement: a genuinely
# empty vertical strip in the figure's LEFT margin (subplots_adjust
# left=0.085), single-column, upper-left, alongside panel (a) -- guaranteed
# zero overlap with ANY panel's data.
handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CAT8[c],
                  markeredgecolor="none", markersize=4.5, label=str(c))
           for c in range(8)]
LEGEND_TOP_Y = 0.895   # nudged down from 0.918 (a little lower, per feedback)
leg = ax_a.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(0.008, LEGEND_TOP_Y), bbox_transform=fig.transFigure,
                  borderaxespad=0.0, ncol=1, frameon=True,
                  framealpha=1.0, edgecolor="#cccccc", facecolor="white",
                  borderpad=0.4, handletextpad=0.3, labelspacing=0.35,
                  title="class")
leg.get_title().set_fontsize(6.3)
for t in leg.get_texts():
    t.set_fontsize(6.3)
leg.get_frame().set_linewidth(0.6)
leg.set_zorder(6)

# Footnote: class 6 is 630/120,000 rows (0.53%) -- verified directly against
# hb_a's returned per-hex winners: it takes the MAJORITY vote in only 4 of the
# map's 3,906 populated hexes (~0.1%), so it is listed in the legend but
# effectively invisible in the map itself. Say so explicitly, with the actual
# count (not a rounded-to-zero guess), rather than let a reader hunt for a
# color that's honestly almost never there. Placed directly below the legend,
# same left margin, in figure coordinates measured off the legend's own
# rendered bbox (not a hand-guessed offset).
fig.canvas.draw()
_leg_bb = leg.get_window_extent(fig.canvas.get_renderer())
_leg_bottom_fig_y = fig.transFigure.inverted().transform((_leg_bb.x0, _leg_bb.y0))[1]
fig.text(LEFT_X, _leg_bottom_fig_y - 0.012,
          "class 6: 630/120,000 rows (0.5%) —\nwins only 4 of 3,906 hexes",
          transform=fig.transFigure, fontsize=5.6, color=SLATE, style="italic",
          ha="left", va="top", linespacing=1.35)

panel_heading(ax_a, "(a)", "The declared partition organizes covariate space",
             letter_pos=(LEFT_X, ROW1_HEAD_Y), letter_transform=fig.transFigure)

# ================================================================ Panel (b)
# Conformal ordinal score. FIX (iter1): the iter0 alpha-blended scatter drew the
# highest score last everywhere, so every dense region blended toward its local
# high tail -> a near-uniform dark plateau with no visible spatial gradient.
# Replaced with hexbin aggregating the TRUE per-cell mean score, so color tracks
# local mean, not z-order. Same SEQ_TEAL palette + right colorbar.
ax_b = axes[0, 1]
teal_cont = LinearSegmentedColormap.from_list("seq_teal", SEQ_TEAL)
# gs=150 (defined once, shared by panels a/b/d, near the top of the file) gives
# a stable per-cell mean (~15 pts/cell in the body; mincnt=2 drops singletons).
hb_b = ax_b.hexbin(U1, U2, C=df["score"].to_numpy(), reduce_C_function=np.mean,
                   gridsize=HEX_GS, cmap=teal_cont, mincnt=2, linewidths=0.0,
                   extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]),
                   rasterized=RASTER, zorder=2)
# Stretch clim to the 2/98 pct of realized cell means so the real spatial
# variation reads (means concentrate ~0.4-0.8; a fixed 0..1 clim would wash out).
_vb = hb_b.get_array()
_lo_b, _hi_b = np.nanpercentile(_vb, [2, 98])
hb_b.set_clim(_lo_b, _hi_b)
panel_heading(ax_b, "(b)", "Conformal score",
             letter_pos=(0.0, 1.015), letter_transform=ax_b.transAxes)

# ================================================================ Panel (c)
# Named strata over a grey baseline cloud. FIX (iter4): the severe-share
# streamplot never earned its ink through three prior iters (invisible ->
# sparse discrete arrowheads -> dense "woven" texture that competed with the
# strata points for attention) -- cut it rather than force a fourth pass; the
# strata overlays carry the panel's message on their own.
ax_c = axes[1, 0]
strat = df["stratum"].to_numpy()
base = strat == "baseline"
ax_c.scatter(U1[base], U2[base], s=PT_S, c="#c9cdd0", alpha=0.30,
             linewidths=0, rasterized=RASTER, zorder=1)
# Draw RAREST-first / MOST-COMMON-last (n: motorcycle 1231 < unrestrained 1847
# < rural_highspeed 13627) so the most common stratum (rural_highspeed, blue)
# ends up on top and stays visible everywhere it co-occurs with a sparser
# stratum, instead of being buried under whichever was drawn last.
STRATA = [("motorcycle", GOLD, "Motorcycle"),
          ("unrestrained", CORAL, "Unrestrained"),
          ("rural_highspeed", HDR_BLUE, "Rural high-speed")]
for i, (key, col, _label) in enumerate(STRATA):
    m = strat == key
    ax_c.scatter(U1[m], U2[m], s=PT_S + 0.6, c=col, alpha=0.6,
                 linewidths=0, rasterized=RASTER, zorder=3 + i)

# In-plane bold white-haloed labels (reference's cell-type label idiom). FIX
# (iter1): iter0 anchored labels at each stratum's MEDIAN, but these strata are
# multi-island crash subsets, so medians landed between islands ("Motorcycle" in
# empty canvas) and the two central strata stacked on the same point. Now anchor
# each label at its DENSEST sub-cluster (2D-histogram peak) and pull the text
# into clear space with a short leader line so no two labels stack.
halo = [pe.withStroke(linewidth=2.2, foreground="white")]

# Anchors below are DATA-DERIVED on the frozen parquet, then frozen as constants
# (a raw density peak is bins-sensitive and, for the fully-overlapping crash
# strata, can land a coral label on a blue-dominated cell). Each anchor is the
# cell where THAT stratum's own color is locally present/dominant among the
# colored strata, so every leader terminates on real same-color points:
#   rural_highspeed (12.29,-0.09): 1343 rural vs 106 unrestrained nearby (blue).
#   unrestrained    (1.75,10.97):  ~124 unrestrained, coral clearly visible.
#   motorcycle      (20.01,-7.84): 1069 motorcycle, its own bottom-right island.
# (dx, dy) = leader offset in DISPLAY points, + text ha; chosen so no two labels
# stack and each leader points back onto its cluster.
LABEL_ANCHOR = {
    "rural_highspeed": (12.29, -0.09, -18, 48, "center"),
    "unrestrained":    (1.75, 10.97, 32, 20, "left"),
    "motorcycle":      (20.01, -7.84, -46, 26, "right"),
}
for key, col, label in STRATA:
    ax_, ay_, dx, dy, ha = LABEL_ANCHOR[key]
    ax_c.annotate(label, xy=(ax_, ay_), xycoords="data",
                  xytext=(dx, dy), textcoords="offset points",
                  fontsize=7.8, fontweight="bold", color=col,
                  ha=ha, va="center", zorder=6, path_effects=halo,
                  arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                  alpha=0.85, shrinkA=1.0, shrinkB=2.0))
panel_heading(ax_c, "(c)", "Named strata",
             letter_pos=(LEFT_X, ROW2_HEAD_Y), letter_transform=fig.transFigure)

# ================================================================ Panel (d)
# Delta between class-conditional (Mondrian) and marginal split-conformal set
# width. FIX (iter4): raw mean class-conditional width just re-showed (b)'s
# spatial pattern (both track local score) -- no second panel earns its ink by
# repeating the first. The export already carries width_marginal AND
# width_mondrian per record (zero extra compute); their difference answers a
# question (b) cannot: does class-conditional calibration widen sets overall,
# or move width around? 18.7% of records change width (mostly +-1 category),
# mean delta -0.006 -- essentially a redistribution, not an inflation. Diverging
# ramp, centered at 0: coral = class-conditional wider than marginal (cost)
# HERE, teal = narrower (gain) HERE -- the two must net out close to zero
# figure-wide by construction (equal-alpha coverage), but not cell-by-cell.
ax_d = axes[1, 1]
delta_w = (df["width_mondrian"].to_numpy() - df["width_marginal"].to_numpy()).astype(float)
# figstyle.py's canonical DIVERGING = [CORAL, ..., ACCENT_TEAL] fixes CORAL to
# the LOW/negative end (its only precedent, Fig 5's coverage deficit, has
# negative = under-coverage = bad = coral by coincidence of THAT field's sign,
# not a hard rule). For width-delta the "bad" direction is the opposite sign
# (wider = POSITIVE = cost), so reverse the ramp here to keep the coral=cost /
# teal=gain semantic intact rather than importing Fig 5's sign polarity verbatim.
div_cmap = LinearSegmentedColormap.from_list("teal_coral_div", list(reversed(DIVERGING)))
hb_d = ax_d.hexbin(U1, U2, C=delta_w, reduce_C_function=np.mean,
                   gridsize=HEX_GS, cmap=div_cmap, mincnt=2, linewidths=0.0,
                   extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]),
                   rasterized=RASTER, zorder=2)
# Symmetric clim from the realized cell means (2nd/98th pct, then mirrored) so
# the diverging ramp's white/cream midpoint sits exactly at delta=0.
_vd = hb_d.get_array()
_lo_d, _hi_d = np.nanpercentile(_vd, [2, 98])
_clim_d = max(abs(_lo_d), abs(_hi_d))
hb_d.set_norm(TwoSlopeNorm(vmin=-_clim_d, vcenter=0.0, vmax=_clim_d))
panel_heading(ax_d, "(d)", "Class-conditional calibration redistributes width",
             letter_pos=(0.0, 1.015), letter_transform=ax_d.transAxes)

# ---------------------------------------------------------------- colorbars
fig.canvas.draw()   # positions settle before we read them
pos_b = ax_b.get_position()
# Shrink the bar height a hair so the top tick label cannot clip the canvas top.
cax_b = fig.add_axes([pos_b.x1 + 0.012, pos_b.y0, 0.014, pos_b.height * 0.94])
cb_b = fig.colorbar(hb_b, cax=cax_b)
cb_b.set_label("mean ordinal score $s$", fontsize=8.0, color=INK)
cb_b.set_ticks([0.4, 0.6, 0.8])            # interior of the stretched clim
cb_b.ax.tick_params(labelsize=7.0, length=2, width=0.6, color=INK)
cb_b.outline.set_linewidth(0.6)
cb_b.outline.set_edgecolor(SLATE)

pos_d = ax_d.get_position()
cax_d = fig.add_axes([pos_d.x1 + 0.012, pos_d.y0, 0.014, pos_d.height * 0.94])
cb_d = fig.colorbar(hb_d, cax=cax_d)
cb_d.set_label("mean $\\Delta$ width (class-cond. $-$ marginal)", fontsize=7.6, color=INK)
_tick_d = round(_clim_d, 1) if _clim_d >= 0.15 else round(_clim_d, 2)
cb_d.set_ticks([-_tick_d, 0.0, _tick_d])
cb_d.ax.tick_params(labelsize=7.0, length=2, width=0.6, color=INK)
cb_d.outline.set_linewidth(0.6)
cb_d.outline.set_edgecolor(SLATE)

# ---------------------------------------------------------------- UMAP glyph
# Single shared axis-arrow glyph. The reference places it in the a/b gutter, but
# our tight wspace (0.07) leaves no clear room there; the plan's sanctioned
# fallback is the bottom-left of the whole 2x2 grid, in the outer margin below
# panel (c) where no data or label can collide.
# Enlarged (iter2): iter1's 0.085 device was cramped and the labels sat directly
# ON the arrow shafts ("UMAP 2" struck through the vertical shaft). Enlarge to
# 0.12 and adopt the standard embedding corner-glyph idiom: a common bottom-left
# origin, SHORT arrows, and each label offset clearly BESIDE its own arrow in the
# direction away from the plot — "UMAP 1" centered BELOW the horizontal shaft,
# "UMAP 2" rotated and set to the LEFT of the vertical shaft. No overlap.
gax = fig.add_axes([0.010, 0.026, 0.12, 0.12])
gax.set_xlim(0, 1)
gax.set_ylim(0, 1)
gax.set_xticks([])
gax.set_yticks([])
gax.axis("off")
OX, OY = 0.22, 0.30                       # common origin (leaves room for labels)
arr = dict(arrowstyle="-|>", color=INK, linewidth=1.1, mutation_scale=8)
gax.annotate("", xy=(OX, 0.96), xytext=(OX, OY), arrowprops=arr)   # up  -> UMAP 2
gax.annotate("", xy=(0.96, OY), xytext=(OX, OY), arrowprops=arr)   # right-> UMAP 1
# "UMAP 1" under the horizontal shaft (shaft at y=OY=0.30; text top well below it).
gax.text((OX + 0.96) / 2, 0.08, "UMAP 1", fontsize=6.6, fontweight="bold",
         color=INK, ha="center", va="bottom")
# "UMAP 2" left of the vertical shaft (shaft at x=OX=0.22; text right edge left of it).
gax.text(0.05, (OY + 0.96) / 2, "UMAP 2", fontsize=6.6, fontweight="bold",
         color=INK, ha="center", va="center", rotation=90)

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
        for t in a.texts:                       # in-plane labels, panel letters, glyph
            # a.texts also contains Annotation objects already collected above;
            # skip them here so they are not double-counted (self-overlap).
            if isinstance(t, _mpl.text.Annotation):
                continue
            if t.get_text() and t.get_visible():
                out.append(("text", t))
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
fig.savefig(OUT / "fig4_heterogeneity_atlas.png", dpi=300)
fig.savefig(OUT / "fig4_heterogeneity_atlas.pdf")
floor_selfcheck(fig, OUT / "floor_selfcheck_iter2.txt")
plt.close(fig)
