"""
Figure 8 — "Risk control and the price surface".

Style transferred from a 9-panel physics figure (template): the reference's
line-panel row (a-c) and per-panel-colorbar labeled-contour heatmap rows
(d-i) supply the visual grammar. Content is OURS. Composition here is a
5-panel adaptation driven by our data:

  TOP ROW    (a) joint fatal-omission probability vs budget beta / (b) severity-weighted
             cost risk vs beta / (c) escalation workload vs beta -- three
             LOG-x line panels (a,b are log-log with a dashed INK budget
             diagonal; c is a semilog workload curve, no bound). E7's four
             original tested budgets (is_e7_grid) are overlaid as OPEN
             checkpoint circles on every curve.
  BOTTOM ROW (d) marginal prediction-set width / (e) declared-band width
             over the (hour x speed_bin) plane -- two SEQ_TEAL sequential
             heatmaps, INK iso-width contours labeled inline, sparse cells
             (n<400) hatched (fig2/fig5 idiom), EACH with its OWN colorbar
             (per-panel convention, mirroring the reference's d-i rows).

HONESTY: panels (a)/(b) plot the TRUE data including the low-beta edge points
where the empirical value slightly EXCEEDS its nominal bound (1 crossing in a,
6 in b). These are NOT clipped, smoothed, or hidden -- they are expected
finite-sample noise around a marginal CRC guarantee and are annotated as such.

Self-contained / standalone: loads the two frozen CSVs via pandas by path
(never inlines row values), resolving the path like sibling fig2/fig5 scripts.
Final home: paper/figures/fig8_risk_price_surface.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd


def gaussian_filter(a, sigma):
    """Separable 2D Gaussian blur with reflect padding (numpy-only; the repo's
    python has no scipy, so we inline the small filter fig2/fig5 got from
    scipy.ndimage). Same normalized-convolution use downstream."""
    radius = max(1, int(3 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    out = a.astype(float)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"),
                                  axis, padded)
    return out


plt.rcParams["pdf.fonttype"] = 42   # camera-ready: no Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# --- house palette (inlined from experiments/figstyle.py) ---
INK = "#1a1a1a"           # near-black structure/ink; bound lines, contours
HDR_BLUE = "#1f4e79"      # workload / operating-curve register (panel c)
ACCENT_TEAL = "#11746c"   # risk curves (a,b); anchor of the width ramp
SLATE = "#697784"         # muted secondary / hatch edge / leader lines
GOLD = "#a87c2a"
MIST = "#e6eeec"          # pale teal-tinted ramp low end
SAND = "#f6e0d9"
# Sequential width ramp (magnitude, NOT a deviation) anchored on ACCENT_TEAL,
# analogous to fig2's SEQ_WARM anchored on CORAL: pale teal tint -> ACCENT_TEAL
# -> a deeper teal so the high-width ridge separates cleanly from the bulk.
SEQ_TEAL = ["#f1f7f5", "#cfe6e2", "#96ccc4", "#4a9f96", ACCENT_TEAL, "#0a4c47"]

# Font family: mirror fig2/fig5's family-probe (Helvetica/Arial house sans).
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
    "axes.labelweight": "bold",   # match fig2/fig5 bold axis labels
    "axes.labelpad": 5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.4,
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

CURVES_CSV = resolve_data_path("fig08_curves.csv")
PLANES_CSV = resolve_data_path("fig08_planes.csv")

curves = pd.read_csv(CURVES_CSV).sort_values("beta").reset_index(drop=True)
planes = pd.read_csv(PLANES_CSV)

MIN_N = 400                     # cells with fewer records -> masked "sparse"
KAPPA_MAX = 1500                # cost bound slope (cost_bound = beta * kappa_max)
N_FATAL = 2475                  # verified fatal count (NOT stale 2498)
# Two genuinely different quantities, but the "price surface" redistribution
# (declared widens the deficit ridge, narrows the easy bulk) is only readable if
# both maps share ONE numeric scale. So: per-panel colorbars (reference d-i
# convention) BUT a shared vmin/vmax so teal-depth is directly comparable.
WVMIN, WVMAX = 2.2, 4.8
CONTOUR_LEVELS = [2, 3, 4]      # integer KABCO-category iso-width lines
SMOOTH_SIGMA = 0.8              # display smoothing (cells), fig2 idiom
# === END DATA SECTOR ===

BETA = curves["beta"].values
E7 = curves[curves["is_e7_grid"]]

# ---------------------------------------------------------------- figure/layout
fig = plt.figure(figsize=(7.0, 6.2), dpi=300)

# ---- top row: 3 log-x line panels (each keeps its own y scale) ----
LEFT = 0.075
RIGHT = 0.985
TGAP = 0.078                    # inter-panel gap (room for each panel's y ticks)
top_w = (RIGHT - LEFT - 2 * TGAP) / 3.0
TOP_Y0, TOP_Y1 = 0.600, 0.945
top_h = TOP_Y1 - TOP_Y0

axA = fig.add_axes([LEFT, TOP_Y0, top_w, top_h])
axB = fig.add_axes([LEFT + top_w + TGAP, TOP_Y0, top_w, top_h])
axC = fig.add_axes([LEFT + 2 * (top_w + TGAP), TOP_Y0, top_w, top_h])

XT = [1e-3, 1e-2, 1e-1]
XTL = ["0.001", "0.01", "0.1"]
XLIM = (4e-4, 0.13)


def style_line_ax(ax, title):
    for s in ax.spines.values():         # all-4 box spines (reference a-c style)
        s.set_color(INK)
        s.set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8, color=INK, which="major")
    ax.tick_params(length=1.8, width=0.6, color=INK, which="minor")
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(XT))
    ax.xaxis.set_major_formatter(FixedFormatter(XTL))
    ax.set_xlabel(r"risk budget $\beta$")
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                fontsize=7.3, fontweight="bold", color=INK)


def e7_markers(ax, ycol):
    ax.plot(E7["beta"].values, E7[ycol].values, linestyle="none", marker="o",
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
            markersize=6.0, zorder=6, label="E7 grid (tested)")


# (a) joint fatal-omission probability vs budget -------------------------------
axA.plot(BETA, curves["fatal_bound"].values, color=INK, linestyle="--",
         linewidth=1.1, zorder=2, label=r"budget $\beta$")
axA.plot(BETA, curves["joint_fatal_omission"].values, color=ACCENT_TEAL, linewidth=1.7,
         solid_capstyle="round", zorder=4, label="empirical probability")
e7_markers(axA, "joint_fatal_omission")
axA.set_yscale("log")
axA.set_ylim(3e-4, 0.2)
axA.set_yticks([1e-3, 1e-2, 1e-1])   # explicit: suppress phantom out-of-range decade tick
axA.set_ylabel("joint fatal-omission probability")
style_line_ax(axA, "(a) Joint fatal omission tracks its budget")
axA.legend(loc="upper left", frameon=False, handlelength=1.5, handletextpad=0.4,
           borderpad=0.2, labelspacing=0.3, bbox_to_anchor=(0.02, 0.99))
# edge crossing at beta=0.000624 (rate 0.000630 > bound 0.000624)
# lifted well clear of the x-axis: text now hangs DOWN from y=1.7e-3 (va="top"),
# so its lowest line bottoms out ~6.7e-4 -- structurally above the 3e-4 axis line
# and its tick marks (iter0 defect: 3rd line struck through the bottom ticks).
axA.annotate("edge crossing\nrate slightly > $\\beta$\n(finite-sample noise)",
             xy=(6.24e-4, 6.30e-4), xycoords="data",
             xytext=(3.6e-3, 1.7e-3), textcoords="data", fontsize=5.8, color=SLATE,
             ha="left", va="top", zorder=7,
             arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=0.6,
                             shrinkA=1.2, shrinkB=2.5))
# moved off the lower-right (the lifted edge-crossing note now occupies that band)
# into the empty triangle above the rate plateau, below the budget diagonal.
axA.annotate(f"$n_{{\\mathrm{{fatal}}}}={N_FATAL}$", xy=(0.96, 0.42),
             xycoords="axes fraction", ha="right", va="center",
             fontsize=6.4, color=INK)

# (b) severity-weighted cost risk vs budget (log-log; bound = beta*kappa) -------
axB.plot(BETA, curves["cost_bound"].values, color=INK, linestyle="--",
         linewidth=1.1, zorder=2, label=r"budget $\beta\,\kappa_{\max}$")
axB.plot(BETA, curves["cost_risk"].values, color=ACCENT_TEAL, linewidth=1.7,
         solid_capstyle="round", zorder=4, label="empirical cost risk")
e7_markers(axB, "cost_risk")
axB.set_yscale("log")
axB.set_ylim(0.45, 260)
axB.set_yticks([1e0, 1e1, 1e2])   # explicit: suppress phantom out-of-range decade tick
axB.set_ylabel("severity-weighted cost risk")
style_line_ax(axB, "(b) Cost risk tracks its budget")
axB.legend(loc="upper left", frameon=False, handlelength=1.5, handletextpad=0.4,
           borderpad=0.2, labelspacing=0.3, bbox_to_anchor=(0.02, 0.99))
axB.annotate("6 low-$\\beta$ edge crossings\n(finite-sample noise)",
             xy=(4.55e-3, 6.85), xycoords="data",
             xytext=(1.3e-3, 55), textcoords="data", fontsize=5.8, color=SLATE,
             ha="left", va="center", zorder=7,
             arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=0.6,
                             shrinkA=1.2, shrinkB=2.5))
axB.annotate(f"$\\kappa_{{\\max}}={KAPPA_MAX}$", xy=(0.98, 0.06),
             xycoords="axes fraction", ha="right", va="bottom",
             fontsize=6.4, color=INK)

# (c) escalation workload vs budget (semilog-x; no bound) -----------------------
axC.plot(BETA, curves["escalated_share"].values, color=HDR_BLUE, linewidth=1.7,
         solid_capstyle="round", zorder=4, label="escalated share")
e7_markers(axC, "escalated_share")
axC.set_ylim(0.0, 0.31)
axC.set_yticks([0.0, 0.1, 0.2, 0.3])
axC.set_ylabel("escalated share")
style_line_ax(axC, "(c) Review workload saturates")
axC.legend(loc="lower right", frameon=False, handlelength=1.5, handletextpad=0.4,
           borderpad=0.2, labelspacing=0.3, bbox_to_anchor=(0.98, 0.02))

# ================================================================ BOTTOM ROW
# Two width heatmaps, EACH with its own colorbar (reference d-i convention),
# sharing one numeric scale so the redistribution is directly comparable.
hours = np.arange(0, 24)
speeds = np.arange(15, 85, 5)                 # bin floors 15..80
h_edges = np.arange(-0.5, 24.5, 1.0)
s_edges = np.arange(15, 90, 5)                 # 15..85, each bin spans [floor, floor+5)
h_cent = hours.astype(float)
s_cent = speeds + 2.5

N_grid = planes.pivot(index="speed_bin", columns="hour", values="n").reindex(
    index=speeds, columns=hours).values
sparse = N_grid < MIN_N


def width_grid(col):
    return planes.pivot(index="speed_bin", columns="hour", values=col).reindex(
        index=speeds, columns=hours).values


def contour_field(Z):
    # NaN-aware ("normalized") smoothing so sparse cells never bleed into the
    # contour geometry, then hard-exclude them (fig2 idiom exactly).
    valid = (~sparse).astype(float)
    Z0 = np.where(sparse, 0.0, np.nan_to_num(Z))
    num = gaussian_filter(Z0, SMOOTH_SIGMA)
    den = gaussian_filter(valid, SMOOTH_SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        Zc = np.where(den > 1e-6, num / den, np.nan)
    return np.where(sparse, np.nan, Zc)


teal_cmap = LinearSegmentedColormap.from_list("seq_teal", SEQ_TEAL)
wnorm = Normalize(vmin=WVMIN, vmax=WVMAX)

MAP_W, MAP_H = 0.300, 0.360
CB_W = 0.017
BOT_Y0 = 0.070
MAP1_X = 0.075
CB1_X = MAP1_X + MAP_W + 0.006
MAP2_X = 0.560
CB2_X = MAP2_X + MAP_W + 0.006

PANELS = [
    ("(d) Marginal spends width evenly", "w_marg", MAP1_X, CB1_X, True),
    ("(e) Declared bands spend width\non the deficit ridge", "w_cc", MAP2_X, CB2_X, False),
]

for title, col, mx, cbx, show_ylabel in PANELS:
    ax = fig.add_axes([mx, BOT_Y0, MAP_W, MAP_H])
    Z = width_grid(col)
    pcm = ax.pcolormesh(h_edges, s_edges, np.nan_to_num(Z, nan=WVMIN),
                        cmap=teal_cmap, norm=wnorm, shading="flat",
                        rasterized=True, zorder=1)
    # sparse cells: translucent white wash + diagonal hatch (fig2/fig5 idiom)
    if sparse.any():
        ys_m, xs_m = np.where(sparse)
        for yi, xi in zip(ys_m, xs_m):
            ax.add_patch(Rectangle(
                (h_edges[xi], s_edges[yi]),
                h_edges[xi + 1] - h_edges[xi], s_edges[yi + 1] - s_edges[yi],
                facecolor="white", alpha=0.55, edgecolor=SLATE, linewidth=0.3,
                hatch="////", zorder=2))
    # INK iso-width contours labeled inline "2"/"3"/"4"
    Zc = contour_field(Z)
    cs = ax.contour(h_cent, s_cent, Zc, levels=CONTOUR_LEVELS, colors=INK,
                    linewidths=1.2, zorder=4)
    ax.clabel(cs, fmt={lv: f"{lv}" for lv in CONTOUR_LEVELS}, inline=True,
              fontsize=7.0, colors=INK)
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
    if show_ylabel:
        ax.set_ylabel("posted speed limit (mph)")
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                fontsize=7.3, fontweight="bold", color=INK, linespacing=1.15)
    # per-panel colorbar with integer contour-level ticks
    cax = fig.add_axes([cbx, BOT_Y0, CB_W, MAP_H])
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_ticks([2.5, 3.0, 3.5, 4.0, 4.5])
    cb.ax.tick_params(labelsize=6.4, length=3, width=0.8, color=INK)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.9)
    cax.set_title("width\n(cat.)", fontsize=6.2, color=INK, pad=5, linespacing=1.0)

# sparse-cell legend (shared meaning across d/e), tucked under panel (d)
sparse_handle = [Rectangle((0, 0), 1, 1, facecolor="white", alpha=0.9,
                           edgecolor=SLATE, linewidth=0.5, hatch="////")]
fig.legend(sparse_handle, ["sparse cell (n < 400, masked)"],
           loc="lower center", bbox_to_anchor=(0.55, 0.002), frameon=False,
           handlelength=1.4, handletextpad=0.5, fontsize=6.4)

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
fig.savefig(OUT / "fig8_risk_price_surface.png", dpi=300)
fig.savefig(OUT / "fig8_risk_price_surface.pdf")
floor_selfcheck(fig, OUT / "floor_selfcheck_iter1.txt")
plt.close(fig)
