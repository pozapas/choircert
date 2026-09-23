"""Figure 1 v3 — "CHOIR as a certified pipeline" (plate notation, notation-first).

v2 RULED 2026-07-11 (FIGURES_PLAN.md §3): zero prose — every glyph is a symbol,
a short formula, or an index; the only word allowed is the operator name "Cert".
v3 = the user's hand-cleaned layout (2026-07-11) ported back into the script,
plus the provenance fix: lambda-hat moved to the top-right REQUIRES the edge
S_i -> lambda-hat, because the CRC threshold is computed from the calibration
scores (it is a calibration-tier object like q-hat, not a fitted-tier one);
(kappa, beta) are declared inputs, tagged like hyperparameters.

All symbols match theory/methods_body.tex (eq:score, eq:qhat, weighted
quantile, eq:expansion, eq:martingale, certificate tuple). Plates carry
replication; arrows carry dependence.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

INK = "#1a1a1a"
HDR_BLUE = "#1f4e79"
CORAL = "#c1443c"
GOLD = "#a87c2a"
SLATE = "#697784"
MIST = "#e6eeec"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "cm", "font.family": "serif",
    "figure.dpi": 300, "savefig.dpi": 300,
})

FS_NODE = 9.2      # node symbols
FS_SMALL = 7.0     # plate indices, hyperparameter tags
FS_FORMULA = 6.1   # short formula annotations

fig = plt.figure(figsize=(7.0, 3.75), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14.3); ax.set_ylim(0, 7.65)
ax.set_aspect("equal"); ax.set_axis_off()

nodes = {}

def node(name, x, y, label, r=0.44, shaded=False, fs=FS_NODE):
    ax.add_patch(Circle((x, y), r, facecolor=MIST if shaded else "white",
                        edgecolor=INK, linewidth=0.9, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=5)
    nodes[name] = (x, y, r)

def edge(a, b, color=INK, ls="-", lw=0.8, shrink_a=None, shrink_b=None,
         zorder=3, rad=0.0):
    xa, ya, ra = nodes[a] if isinstance(a, str) else (*a, 0.0)
    xb, yb, rb = nodes[b] if isinstance(b, str) else (*b, 0.0)
    ra = ra if shrink_a is None else shrink_a
    rb = rb if shrink_b is None else shrink_b
    v = np.array([xb - xa, yb - ya]); L = np.hypot(*v); u = v / L
    p0 = np.array([xa, ya]) + u * (ra + 0.04)
    p1 = np.array([xb, yb]) - u * (rb + 0.04)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=zorder))

def hyper(x, y, label, target, color=INK, fs=FS_SMALL, ls="-"):
    """Tiny tag with a short arrow into a node edge (the nu0,W0 idiom)."""
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=6)
    edge((x, y), target, color=color, ls=ls, lw=0.7, shrink_a=0.17)

def plate(x0, y0, w, h, color, corner, corner_xy, ha="left"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.16",
                                facecolor="none", edgecolor=color,
                                linewidth=1.0, zorder=1))
    ax.text(*corner_xy, corner, ha=ha, va="bottom", fontsize=FS_SMALL,
            color=color, zorder=6)

def formula(x, y, s, fs=FS_FORMULA, color=INK, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, zorder=6)

# ---------------------------------------------------------------- plates
plate(0.55, 1.35, 3.10, 4.30, CORAL, r"$i\in\mathcal{I}_{\mathrm{cal}}$", (0.72, 1.48))
plate(4.25, 1.05, 3.50, 4.60, HDR_BLUE, r"$c=1{:}C$", (4.42, 1.18))
plate(4.95, 1.75, 3.85, 2.90, GOLD, r"$a\in\mathcal{R}$", (5.10, 1.88))
plate(9.40, 0.95, 4.75, 4.90, SLATE, r"$t=1,2,\ldots$", (9.57, 1.08))

# ---------------------------------------------------------------- top tier
node("F", 1.70, 6.45, r"$\hat F(\cdot\,|\,x)$", r=0.52, fs=8.4)
node("c", 3.85, 6.45, r"$\hat c(x)$", r=0.46, fs=8.8)
node("g", 5.90, 6.45, r"$g(x)$", r=0.46, fs=8.8)
node("w", 8.05, 6.45, r"$\hat w(x)$", r=0.46, fs=8.8)
for nm in ("F", "c", "g", "w"):
    x, y, r = nodes[nm]
    hyper(x, 7.32, r"$\mathcal{I}_{\mathrm{tr}}$", nm, color=SLATE)
# unlabeled target covariates -> density ratio (dashed GOLD)
ax.add_patch(Rectangle((9.15, 7.02), 0.44, 0.44, facecolor="white",
                       edgecolor=GOLD, linewidth=0.9, zorder=4))
ax.text(9.37, 7.24, r"$X^{*}$", ha="center", va="center", fontsize=FS_SMALL, zorder=5)
edge((9.22, 7.00), "w", color=GOLD, ls=(0, (3, 2)), shrink_a=0.0)

# lambda-hat: calibration-derived CRC threshold (top-right for layout, but its
# dependence is honest: S_i -> lambda-hat; kappa/beta are declared inputs).
node("lam", 11.35, 6.45, r"$\hat\lambda$", r=0.50, fs=9.2)
hyper(11.35, 7.32, r"$\kappa,\beta$", "lam", color=SLATE)
formula(13.10, 6.45, r"$\mathbb{E}[\kappa(Y)\mathbf{1}\{Y\notin C_{\hat\lambda}\}]\leq\beta$",
        fs=6.0, color=SLATE)

# ---------------------------------------------------------------- calibration
node("XY", 2.40, 4.60, r"$(X_i,\tilde Y_i)$", r=0.58, shaded=True, fs=8.2)
node("S", 1.55, 2.75, r"$S_i$", r=0.42)
formula(2.10, 1.98, r"$s(x,y)=\max\{\hat F(y{-}1|x),\,1{-}\hat F(y|x)\}$", fs=5.9)
edge("F", "S"); edge("XY", "S")
edge("S", "lam", rad=0.16)   # calibration scores feed the CRC threshold

# ---------------------------------------------------------------- product cell
node("q", 6.35, 3.05, r"$\hat q_a$", r=0.46)
formula(6.35, 2.25, r"$\lceil(1{-}\alpha)(n_a{+}1)\rceil$")
edge("S", "q")
edge("c", "q")                                  # gate -> cell quantile
edge("g", "q")                                  # stratum map -> cell quantile
edge("w", "q", color=GOLD, ls=(0, (3, 2)))      # weighted quantile (new stratum)

# ---------------------------------------------------------------- deployment
node("x", 10.30, 4.85, r"$x_t$", r=0.42, shaded=True)
node("Ct", 11.85, 4.35, r"$\tilde C_t$", r=0.46)
node("Cop", 13.30, 4.35, r"$C^{\oplus}_t$", r=0.46)
node("M", 11.00, 1.55, r"$M_t$", r=0.42)
formula(12.40, 5.15, r"$[\tilde a_t,\tilde b_t]$")
formula(11.95, 3.55, r"$\mathbb{P}(\tilde Y_t\in\tilde C_t\mid R{=}a)\geq 1{-}\alpha$",
        fs=6.0, color=SLATE)
formula(13.00, 5.50, r"$\mathbb{P}(Y_t\in C^{\oplus}_t)\geq 1{-}\alpha{-}\delta$",
        fs=6.0, color=SLATE)
hyper(13.70, 3.50, r"$(\mathcal{T},\delta)$", "Cop", color=SLATE)
hyper(12.05, 1.30, r"$\varepsilon$", "M", color=SLATE)
formula(9.95, 2.35, r"$M_t=\prod_{s\leq t}\varepsilon\, p_s^{\varepsilon-1}$")

# certificate rectangle (artifact, not a random object)
CERT_W, CERT_H = 2.25, 1.00
cx, cy = 13.00, 2.30
ax.add_patch(FancyBboxPatch((cx - CERT_W / 2, cy - CERT_H / 2), CERT_W, CERT_H,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4))
ax.text(cx, cy + 0.24, r"$\mathrm{Cert}(a)$", ha="center", va="center",
        fontsize=8.0, zorder=5)
ax.text(cx, cy - 0.23,
        r"$\left(1{-}\alpha,\ \hat\Delta_{\mathrm{emp}}^{\,\mathrm{LCB}},\ \cdot\right)$",
        ha="center", va="center", fontsize=7.4, zorder=5)
nodes["Cert"] = (cx, cy, 0.62)

edge("q", "Ct"); edge("lam", "Ct"); edge("x", "Ct")
edge("Ct", "Cop")
edge("Cop", (13.30, cy + CERT_H / 2), shrink_b=0.0)
edge("x", "M")
hyper(12.15, 0.60, r"$\hat\Delta_{\mathrm{emp}}$",
      (cx - 0.45, cy - CERT_H / 2), color=GOLD)

# monitor feedback below all plates: detection, never correction
FB_Y = 0.38
ax.plot([11.00, 11.00, 2.10], [1.55 - 0.46, FB_Y, FB_Y],
        color=CORAL, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
ax.add_patch(FancyArrowPatch((2.10, FB_Y), (2.10, 1.31), arrowstyle="-|>",
                             mutation_scale=7, linewidth=0.9, color=CORAL,
                             linestyle=(0, (4, 2.5)), shrinkA=0, shrinkB=0, zorder=2))
formula(6.50, 0.58, r"$M_t\geq 1/\alpha_{\mathrm{mon}}$", color=CORAL)

# ================================================================ floor checks
fig.canvas.draw(); renderer = fig.canvas.get_renderer()
texts = [t for t in ax.texts if t.get_text()]
bbs = [(t, t.get_window_extent(renderer)) for t in texts]
overlaps = []
for i, (t, b) in enumerate(bbs):
    for u, b2 in bbs[i + 1:]:
        if b.overlaps(b2):
            overlaps.append((t.get_text(), u.get_text()))
inside = all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and
             b.y1 <= fig.bbox.height for _, b in bbs)
# no-prose gate: after stripping math, only allowed word fragments may remain
ALLOWED = {"Cert", "cal", "tr", "emp", "LCB", "mon", "max"}
prose = []
for t in texts:
    for w in re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", " ", t.get_text())):
        if w not in ALLOWED:
            prose.append((w, t.get_text()))
out = Path(__file__).resolve().parent
status = "PASS" if not overlaps and inside and not prose else "FAIL"
with open(out / "floor_selfcheck_fig1.txt", "w", encoding="utf-8") as f:
    f.write(f"FLOOR SELF-CHECK: {status}\n")
    f.write(f"text_text_overlap: {'PASS' if not overlaps else 'FAIL ' + str(overlaps[:6])}\n")
    f.write(f"all_text_inside_canvas: {'PASS' if inside else 'FAIL'}\n")
    f.write(f"no_prose_gate (allowed: {sorted(ALLOWED)}): "
            f"{'PASS' if not prose else 'FAIL ' + str(prose[:6])}\n")
    f.write(f"text items checked: {len(bbs)}\n")

fig.savefig(out / "fig1_choir_pipeline.png", facecolor="white")
fig.savefig(out / "fig1_choir_pipeline.pdf", facecolor="white")
plt.close(fig)
print(open(out / "floor_selfcheck_fig1.txt", encoding="utf-8").read())
