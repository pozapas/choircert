"""Figure 3 — Covariate geometry by outcome (O vs K+A corner plot).

Style-transfer of a qBIRD-vs-Bilby dynesty corner plot (L1 reference): a k x k
lower-triangle grid, two overlaid groups with filled 2D density contours off the
diagonal, marginal 1D KDEs on the diagonal with summary-stat panel titles, dashed
reference crosshairs, and a single top-right legend. Our data + our palette.

Standalone and self-contained. The released path uses aggregate density grids and
panel-title summaries under paper/figure_src/data. Restricted row-level covariates
are required only to rebuild the aggregate grids.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",
    "axes.labelpad": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
})

# --- palette (from figstyle.py; the reference is a STYLE anchor, colors are ours) ---
INK = "#1a1a1a"
ACCENT_TEAL = "#11746c"   # O-group line
CORAL = "#c1443c"         # K+A-group line
SLATE = "#697784"         # dashed crosshairs
# Fill families (outer 90% / inner 50%), per LOCKED spec:
O_FILL = ["#a9d3cc", "#5fb0a5"]     # teal light / mid
KA_FILL = ["#f6e0d9", "#e8a08a"]    # SAND / coral-mid
SPINE = "#1a1a1a"  # = INK. L2-class: near-black hairline (#000-#444). iter0 used
                   #   #3a3a3a (soft mid-grey per reviewer); reference corner-box
                   #   frames read near-black, so darkened to INK at heavier hairline.
SPINE_LW = 0.9     # up from 0.7 to match ref's slightly heavier near-black frame.

# === DATA SECTOR (edit here) — file load, not inline literals ===
def find_repo_root(start: Path) -> Path:
    """Final home is paper/figures/<name>.py (repo_root = parents[2]); but walk up
    so the script also renders correctly from the figmirror workdir."""
    for p in [start] + list(start.parents):
        if (p / "paper" / "figure_src" / "data" / "fig03_corner_meta.json").exists():
            return p
    return start.resolve().parents[2]

REPO_ROOT = find_repo_root(Path(__file__).resolve())
DATA = REPO_ROOT / "paper" / "figure_src" / "data"

AGG_PATH = DATA / "fig03_corner_density.npz"
USE_AGGREGATE = AGG_PATH.exists()
aggregate = np.load(AGG_PATH) if USE_AGGREGATE else None
df_o = None if USE_AGGREGATE else pd.read_csv(DATA / "fig03_corner_o.csv")
df_ka = None if USE_AGGREGATE else pd.read_csv(DATA / "fig03_corner_ka.csv")
with open(DATA / "fig03_corner_meta.json") as fh:
    meta = json.load(fh)
# === END DATA SECTOR ===

VARS = ["speed_limit", "hour", "age", "vehicle_age"]
LABELS = {"speed_limit": "speed limit (mph)", "hour": "hour of day",
          "age": "driver age", "vehicle_age": "vehicle age (yr)"}
RANGES = {"speed_limit": (15, 85), "hour": (0, 23), "age": (14, 95),
          "vehicle_age": (0, 40)}
# Off-diagonal panels are drawn on a PADDED range, not the bare data range: a
# KDE fit right up to a hard axis limit gets flush-cut at that edge (density
# doesn't taper to zero exactly at 15 mph or hour 23), which reads as content
# spilling out of the panel. A 6% margin each side lets every joint contour
# taper into whitespace within its own frame.
PAD_FRAC = 0.06
PADDED = {v: (lo - PAD_FRAC * (hi - lo), hi + PAD_FRAC * (hi - lo))
          for v, (lo, hi) in RANGES.items()}
# explicit 3-tick sets kept strictly inside each fixed range (no out-of-range
# ticks that would clip off-canvas)
TICKS = {"speed_limit": [25, 50, 75], "hour": [0, 8, 16], "age": [25, 50, 75],
         "vehicle_age": [0, 15, 30]}
# Marginal histogram bin edges chosen per-variable so bin width never drops below
# the data's discreteness (avoids aliasing spikes: hour/vehicle_age are integer-
# valued, speed_limit is multiples of 5). Reference-crisp steps, no comb artifacts.
BINS = {"speed_limit": np.arange(15, 86, 5),      # 14 bars on the 5-mph grid
        "hour": np.arange(-0.5, 24.5, 1.0),        # 24 unit bars, one per hour
        "age": np.linspace(14, 95, 28),            # width ~3 yr
        "vehicle_age": np.arange(-0.5, 40.5, 2.0)}  # width 2 yr

median_o = meta["median_o"]
median_ka = meta["median_ka"]
iqr_ka = meta["iqr_ka"]

# Subsample for the KDE FIT ONLY (titles use full-sample meta.json numbers).
RNG = np.random.default_rng(20260704)
KDE_N = 15000
def subsample(df):
    if len(df) <= KDE_N:
        return df
    return df.iloc[RNG.choice(len(df), KDE_N, replace=False)]
so, ska = ((None, None) if USE_AGGREGATE else (subsample(df_o), subsample(df_ka)))


def density_levels(Z, fracs=(0.5, 0.9)):
    """Density thresholds enclosing the given mass fractions (highest-density regions)."""
    flat = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [flat[min(np.searchsorted(csum, f), len(flat) - 1)] for f in fracs]


def draw_2d(ax, df, xv, yv, fills, line, group):
    if USE_AGGREGATE:
        prefix = f"grid_{group}_{xv}_{yv}"
        gx = aggregate[f"{prefix}_x"]
        gy = aggregate[f"{prefix}_y"]
        Z = aggregate[f"{prefix}_z"]
        GX, GY = np.meshgrid(gx, gy)
    else:
        x, y = df[xv].to_numpy(), df[yv].to_numpy()
        xr, yr = PADDED[xv], PADDED[yv]
        x = x + RNG.normal(0, 1e-6, x.shape)
        y = y + RNG.normal(0, 1e-6, y.shape)
        kde = gaussian_kde(np.vstack([x, y]), bw_method=0.30)
        gx = np.linspace(*xr, 120)
        gy = np.linspace(*yr, 120)
        GX, GY = np.meshgrid(gx, gy)
        Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    l50, l90 = density_levels(Z)              # l50 (inner) > l90 (outer)
    levels = [l90, l50, Z.max() * 1.0001]
    # lighter fill so the two mass levels stay legible while the crisp outlines
    # (opaque, 1.0pt) carry the contour structure the reviewer asked for.
    ax.contourf(GX, GY, Z, levels=levels, colors=fills, alpha=0.5)
    ax.contour(GX, GY, Z, levels=[l90, l50], colors=line, linewidths=1.0, alpha=1.0)


def draw_1d(ax, xv):
    # Reference marginals are UNFILLED STEP HISTOGRAMS: crisp step outlines only, no
    # fill body. Fills muddied the teal (O) and coral (K+A) into a brown blend where
    # they overlap; unfilled outlines keep each series' color crisp and separable,
    # matching the reference's discrete step-outline diagonal marginals.
    xr = PADDED[xv]
    for group, df, line in (("o", so, ACCENT_TEAL), ("ka", ska, CORAL)):
        if USE_AGGREGATE:
            bins = aggregate[f"hist_{group}_{xv}_edges"]
            h = aggregate[f"hist_{group}_{xv}_height"]
        else:
            bins = BINS[xv]
            vals = df[xv].to_numpy()
            h, _ = np.histogram(vals, bins=bins, density=True)
            h = h / h.max()
        ax.stairs(h, bins, color=line, lw=1.2)   # outline only — no fill
    ax.set_xlim(*xr)
    ax.set_ylim(0, 1.12)


k = len(VARS)
fig, axes = plt.subplots(k, k, figsize=(7.0, 7.0))
fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.085,
                    wspace=0.10, hspace=0.10)

for i in range(k):        # row -> y variable
    for j in range(k):    # col -> x variable
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        for s in ax.spines.values():
            s.set_edgecolor(SPINE)
            s.set_linewidth(SPINE_LW)
        ax.tick_params(direction="out", length=2.6, width=SPINE_LW, color=SPINE,
                       labelcolor=INK)

        if i == j:                                    # diagonal: 1D marginals
            draw_1d(ax, VARS[j])
            ax.yaxis.set_major_locator(NullLocator())  # density axis hidden (as in ref)
            ax.tick_params(left=False)
            ax._marginal = True   # tag: density y-axis carries no rendered labels
            m, lo, hi = median_ka[VARS[j]], iqr_ka[VARS[j]][0], iqr_ka[VARS[j]][1]
            ax.set_title(f"{m:g} [{lo:g}–{hi:g}]", pad=4, color=INK)
        else:                                         # off-diagonal: 2D contours
            xv, yv = VARS[j], VARS[i]
            draw_2d(ax, so, xv, yv, O_FILL, ACCENT_TEAL, "o")
            draw_2d(ax, ska, xv, yv, KA_FILL, CORAL, "ka")   # K+A on top
            ax.axvline(median_o[xv], color=SLATE, ls=(0, (5, 3)), lw=0.9, zorder=5)
            ax.axhline(median_o[yv], color=SLATE, ls=(0, (5, 3)), lw=0.9, zorder=5)
            ax.set_xlim(*PADDED[xv])
            ax.set_ylim(*PADDED[yv])
            ax.set_yticks(TICKS[VARS[i]])
        ax.set_xticks(TICKS[VARS[j]])
        ax.set_xlim(*PADDED[VARS[j]])

        # labels: bottom row gets xlabel/xticklabels; left column gets ylabel/yticklabels
        if i == k - 1:
            ax.set_xlabel(LABELS[VARS[j]])
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(LABELS[VARS[i]])
        elif i != j:
            ax.set_yticklabels([])

# --- single legend, top-right (in the vacated upper-triangle space) ---
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=ACCENT_TEAL, lw=2.2, label="no injury (O)"),
           Line2D([0], [0], color=CORAL, lw=2.2, label="severe or fatal (K+A)")]
leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.955),
                 frameon=True, fancybox=True, borderpad=0.5, labelspacing=0.5,
                 handlelength=1.9, handletextpad=0.6, edgecolor="#cccccc",
                 facecolor="white")
leg.get_frame().set_linewidth(0.8)

# ---------- floor self-check ----------
def assert_no_text_overlap(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        if not ax.axison:           # skip axis('off') upper-triangle panels
            continue
        marg = getattr(ax, "_marginal", False)  # skip unlabeled density y-axis
        ticklabels = list(ax.get_xticklabels())
        if not marg:
            ticklabels += list(ax.get_yticklabels())
        for t in ticklabels:
            if t.get_text() and t.get_visible():
                texts.append(("tick", t))
        if ax.get_title():
            texts.append(("title", ax.title))
        if ax.xaxis.get_label_text():
            texts.append(("xlabel", ax.xaxis.label))
        if ax.yaxis.get_label_text():
            texts.append(("ylabel", ax.yaxis.label))
    # Capture text string AND bbox eagerly per artist (get_window_extent can
    # transiently regenerate lazy ticks; read text first so the record is stable).
    boxes = []
    for k, t in texts:
        s = t.get_text()
        boxes.append((k, s, t.get_window_extent(renderer=r)))
    v = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][2].overlaps(boxes[b][2]):
                v.append(f"{boxes[a][0]}('{boxes[a][1]}') <-> "
                         f"{boxes[b][0]}('{boxes[b][1]}')")
    return v


def assert_no_clipped(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    out = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        marg = getattr(ax, "_marginal", False)
        items = list(ax.get_xticklabels()) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if not marg:
            items += list(ax.get_yticklabels())
        for t in items:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(renderer=r)
            if not (fb.contains(tb.x0, tb.y0) and fb.contains(tb.x1, tb.y1)):
                out.append(f"clipped: '{t.get_text()}'")
    return out


# Run the floor check on a clean canvas-dpi draw BEFORE savefig, so no 180-dpi
# save renderer is left cached to transiently regenerate stale marginal ticks.
ov = assert_no_text_overlap(fig)
cl = assert_no_clipped(fig)

OUT = Path(__file__).resolve().parent / "fig3_corner_geometry.png"
fig.savefig(OUT, dpi=300)
fig.savefig(Path(__file__).resolve().parent / "fig3_corner_geometry.pdf")

report = Path(__file__).resolve().parent / "floor_selfcheck_iter2.txt"
with open(report, "w") as fh:
    fh.write(f"text overlaps: {len(ov)}\n")
    for x in ov:
        fh.write("  " + x + "\n")
    fh.write(f"clipped labels: {len(cl)}\n")
    for x in cl:
        fh.write("  " + x + "\n")
    fh.write("PASS\n" if not ov and not cl else "FAIL\n")
print("floor:", "PASS" if not ov and not cl else f"FAIL ov={len(ov)} cl={len(cl)}")
