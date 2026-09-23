"""
Figure 2 — "The severity landscape of 5.2M driver records".

Style is transferred from a sulfur P-T phase diagram (gradient-filled field +
vertical colorbar + labeled boundary curves + in-plane region names + sparse
boundary markers). Colors/content are OURS: a warm severity field (speed/risk
register) over the (hour-of-day x posted-speed) plane, with 2% / 5% KA-share
boundary contours, hatched sparse cells, and a cool-accent top-cell marker
overlay (kept legible against the warm field).

Self-contained: loads the locked CSV via pandas; palette inlined (mirrors
experiments/figstyle.py) so the script renders standalone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

plt.rcParams["pdf.fonttype"] = 42   # camera-ready: no Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# --- house palette (inlined from experiments/figstyle.py) ---
INK = "#1a1a1a"          # near-black structure/ink
CORAL = "#c1443c"        # severe accent (kept for text/inline use, not the field)
HDR_BLUE = "#1f4e79"     # cool marker accent -- pops against the warm field
SLATE = "#697784"        # muted secondary text
# Warm sequential ramp (speed/severity register): cream -> gold -> orange -> coral -> maroon
SEQ_WARM = ["#fff5eb", "#fdd8ae", "#f7a75f", "#e2703a", "#c1443c", "#7a2420"]
GRAY_MASK = "#c9cdd0"    # base tone for the sparse hatch overlay

for _fam in ("Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros",
             "DejaVu Sans"):
    if any(_fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _fam
        break
plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",
    "axes.labelpad": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.9,
})

# === DATA SECTOR (edit here) ===
REPO_ROOT = Path(__file__).resolve().parents[2]
# Resolve the stored figure data from the reproducibility root.
_candidates = [Path(__file__).resolve().parents[2]]
CSV = None
for root in _candidates:
    p = root / "paper" / "figure_src" / "data" / "fig02_landscape.csv"
    if p.exists():
        CSV = p
        break
if CSV is None:
    raise FileNotFoundError("fig02_landscape.csv not found under candidate roots")

df = pd.read_csv(CSV)
MIN_N = 500          # cells with fewer records are masked "sparse"
MARKER_MIN_N = 2000  # top-cell marker overlay eligibility
REF_SHARE = 0.0168   # overall KA share across full population (1.68%)
CONTOUR_LEVELS = [0.02, 0.05]  # fraction; 2% solid, 5% dashed
SMOOTH_SIGMA = 0.8   # display smoothing (cells)
# === END DATA SECTOR ===

# ---- build the grid (14 speed bins x 24 hours) ----
hours = np.arange(0, 24)
speeds = np.arange(15, 85, 5)  # bin floors 15..80
Z = df.pivot(index="speed_bin", columns="hour", values="share_ka").values
N = df.pivot(index="speed_bin", columns="hour", values="n").values
mask = N < MIN_N  # sparse cells (n<500): unreliable, excluded from the contour surface

# The HEATMAP FILL stays RAW (per-cell share_ka, no smoothing) -- sparse cells
# are still shown here but get hatched over below, so their raw noise doesn't
# matter for the fill.
Z_raw = np.nan_to_num(Z, nan=0.0)

# The CONTOUR surface is a SEPARATE, smoothed field, built with two additional
# guarantees the raw fill doesn't need:
#  1. Sparse cells must not influence the contour geometry at all -- not even
#     through the blur bleeding a neighbor's smoothed value into them. Plain
#     gaussian_filter(nan_to_num(...)) treats a sparse cell's own (noisy, low-n)
#     share_ka as real signal and smears it into the contour; a referee reading
#     a 5% line through the masked "sparse" band would rightly call that out.
#     Fix: normalized ("NaN-aware") convolution -- blur the sparse-zeroed field
#     and a valid-cell indicator separately, divide, so only UNMASKED cells
#     contribute weight anywhere near the mask boundary.
#  2. Sparse cells are then hard-set to NaN in the final contour surface, so
#     matplotlib's contour() leaves a genuine GAP there instead of drawing an
#     interpolated line across ungrounded territory.
valid = (~mask).astype(float)
Z0 = np.where(mask, 0.0, Z_raw)
num = gaussian_filter(Z0, SMOOTH_SIGMA)
den = gaussian_filter(valid, SMOOTH_SIGMA)
with np.errstate(invalid="ignore", divide="ignore"):
    Zc = np.where(den > 1e-6, num / den, np.nan)
Zc = np.where(mask, np.nan, Zc)  # hard-exclude sparse cells from the contour itself

# cell edges (pcolormesh) and cell centers (contour)
h_edges = np.arange(-0.5, 24.5, 1.0)
s_edges = np.arange(15, 90, 5)          # 15..85, each bin spans [floor, floor+5)
h_cent = hours.astype(float)
s_cent = speeds + 2.5                    # bin center

warm = LinearSegmentedColormap.from_list("seq_warm", SEQ_WARM)
VMAX = 0.10  # cap slightly above smoothed max (~0.085) for headroom

# ---- figure ----
# Ref aspect ~1504/1147 = 1.31. Width column-locked at 7.0; height 4.9 -> 1.43
# figure aspect, but axes box is inset so the DATA box lands near ref. (documented)
fig = plt.figure(figsize=(7.0, 4.9), dpi=300)
ax = fig.add_axes([0.085, 0.135, 0.775, 0.815])

# gradient-filled severity field -- RAW per-cell values, no smoothing
pcm = ax.pcolormesh(h_edges, s_edges, Z_raw, cmap=warm, vmin=0.0, vmax=VMAX,
                    shading="flat", rasterized=True, zorder=1)

# masked sparse cells: a translucent wash + diagonal hatch, NOT a flat opaque
# block -- the underlying field color still shows faintly through, so "sparse"
# reads as low-confidence rather than a missing/colorless hole.
from matplotlib.patches import Rectangle

if mask.any():
    ys_m, xs_m = np.where(mask)
    for yi, xi in zip(ys_m, xs_m):
        ax.add_patch(Rectangle(
            (h_edges[xi], s_edges[yi]), h_edges[xi + 1] - h_edges[xi], s_edges[yi + 1] - s_edges[yi],
            facecolor="white", alpha=0.55, edgecolor=SLATE, linewidth=0.3,
            hatch="////", zorder=2,
        ))

# boundary contours of the smoothed, sparse-excluded field (2% solid, 5% dashed)
cs2 = ax.contour(h_cent, s_cent, Zc, levels=[0.02], colors=INK,
                 linewidths=1.4, linestyles="solid", zorder=4)
cs5 = ax.contour(h_cent, s_cent, Zc, levels=[0.05], colors=INK,
                 linewidths=1.4, linestyles="dashed", zorder=4)

# in-plane contour labels (mirror ref's labeled phase-boundary curves)
ax.clabel(cs2, fmt={0.02: "2%"}, inline=True, fontsize=7.0, colors=INK)
ax.clabel(cs5, fmt={0.05: "5%"}, inline=True, fontsize=7.0, colors=INK)

# ---- marker overlay: top-10 highest-share cells with n>=2000 ----
# HDR_BLUE (cool) is used here, not CORAL, so the markers stay legible against
# the warm field instead of blending into its high end.
elig = df[df["n"] >= MARKER_MIN_N].nlargest(10, "share_ka").copy()
smin, smax = elig["n"].min(), elig["n"].max()
sizes = 26 + 74 * (elig["n"] - smin) / max(smax - smin, 1)  # scale by n
ax.scatter(elig["hour"], elig["speed_bin"] + 2.5, s=sizes, facecolor=HDR_BLUE,
           edgecolor="white", linewidth=0.6, zorder=6)

# annotate exactly ONE marker (the single highest share) with its value.
# Text register consolidated to near-black INK (L1: ref uses quiet monochrome
# labels). The coral dots themselves carry the failure-marker accent, not text.
# A thin leader line ties the callout to its exact cell -- without it, a
# floating "7.4%" near the 5%/7% inline contour labels reads as another
# contour level rather than a single-cell value.
top = elig.iloc[0]
ax.annotate(f"{top['share_ka']*100:.1f}%",
            xy=(top["hour"], top["speed_bin"] + 2.5), xycoords="data",
            xytext=(78, -34), textcoords="offset points",
            fontsize=7.0, color=INK, ha="left", va="center", zorder=7,
            arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=0.7,
                             shrinkA=1.5, shrinkB=4.5))

# ---- in-plane regime labels (ref's caps / lowercase hierarchy idiom) ----
# L1: reference region names are PLAIN near-black text set directly on the field
# with NO white halo/stroke — legible over light OR dark fill (cf. "POLYMERIC
# LIQUID" over orange, "SOLID" over grey). Drop iter0's path_effects halos and
# set all in-field text to INK for a quiet monochrome register.
ax.text(4.4, 62, "HIGH-SPEED\nNIGHT RIDGE", fontsize=8.0, color=INK,
        ha="left", va="center", linespacing=1.15, zorder=5)
ax.text(12.5, 30, "urban commuting plateau", fontsize=8.0, color=INK,
        ha="center", va="center", style="italic", zorder=5)
ax.text(12.5, 82, "sparse", fontsize=7.5, color=INK,
        ha="center", va="center", zorder=5)

# ---- legend: what the markers and the hatched cells mean ----
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=HDR_BLUE,
           markeredgecolor="white", markeredgewidth=0.6, markersize=7.5,
           label="top-10 highest-severity cells (size $\\propto$ n)"),
    Patch(facecolor="white", alpha=0.55, edgecolor=SLATE, linewidth=0.5,
          hatch="////", label="sparse (n < 500, masked)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=6.6,
          frameon=True, framealpha=0.92, edgecolor=INK, facecolor="white",
          borderpad=0.5, handletextpad=0.5, labelspacing=0.5).set_zorder(8)

# ---- axes cosmetics ----
ax.set_xlim(-0.5, 23.5)
ax.set_ylim(15, 85)
ax.set_xticks(np.arange(0, 24, 3))
ax.set_xticklabels([f"{h:02d}:00" for h in np.arange(0, 24, 3)])
ax.set_yticks(np.arange(20, 85, 10))
ax.set_xlabel("hour of day")
ax.set_ylabel("posted speed limit (mph)")
ax.tick_params(length=3, width=0.8, color=INK)
for s in ax.spines.values():
    s.set_visible(True)
    s.set_color(INK)
    s.set_linewidth(0.9)

# ---- colorbar (vertical, right) with 1.68% reference tick ----
cax = fig.add_axes([0.885, 0.135, 0.022, 0.815])
cb = fig.colorbar(pcm, cax=cax)
cb.set_label("share of severe or fatal outcomes (K+A) per cell (%)",
             fontsize=8.0, color=INK)
ticks = np.arange(0.0, VMAX + 1e-9, 0.02)
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t*100:.0f}" for t in ticks])
cb.ax.tick_params(labelsize=7.0, length=3, width=0.8, color=INK)
cb.outline.set_edgecolor(INK)
cb.outline.set_linewidth(0.9)
# overall-population reference: a line on the colorbar at the true value, with
# its caption as a small TITLE above the bar (a side annotation here would
# compete with the rotated colorbar label for the same margin and clip off
# the canvas edge -- a title above has its own room).
cax.axhline(REF_SHARE, color=INK, lw=1.1, ls="-")
cax.set_title("overall 1.68%", fontsize=6.8, color=INK, pad=10)

# ===================== FLOOR SELF-CHECK =====================
import matplotlib as _mpl


def _texts(fig):
    out = []
    for a in fig.axes:
        for t in a.get_xticklabels() + a.get_yticklabels():
            if t.get_text():
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
    lines = []
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
    lines.append(f"text items checked: {len(bb)}")
    lines.append(f"overlap violations: {len(viol)}")
    lines += viol
    lines.append(f"clip violations: {len(clip)}")
    lines += clip
    status = "PASS" if not viol and not clip else "FAIL"
    lines.insert(0, f"FLOOR SELF-CHECK: {status}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return status


OUT = Path(__file__).resolve().parent
fig.savefig(OUT / "fig2_severity_landscape.png", dpi=300)
fig.savefig(OUT / "fig2_severity_landscape.pdf")
floor_selfcheck(fig, OUT / "floor_selfcheck_iter1.txt")
plt.close(fig)
