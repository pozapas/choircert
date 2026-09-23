#!/usr/bin/env python
"""Regenerate all stored figures from result parquets in the CHOIR house style.

Figures are written to paper/figures/ and mirrored to experiments/figures/.
One command regenerates every figure.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

import figstyle as fs

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MIRROR = Path(__file__).resolve().parent / "figures"
MIRROR.mkdir(exist_ok=True)

fs.apply()

KABCO = ["O", "C", "B", "A", "K"]


def read_result(name):
    path = RESULTS / name
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png")
    fig.savefig(MIRROR / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: the conditional-coverage failure (E2) — the motivation figure
# ---------------------------------------------------------------------------
def fig_conditional_failure():
    df = read_result("e2_e3_heterogeneity.parquet")
    alpha = float(df["alpha"].iloc[0])
    marg = df[(df.exp == "E2") & (df.cell_type == "declared")].set_index("cell")
    mond = df[(df.method == "mondrian_declared") &
              (df.cell_type == "declared")].set_index("cell")
    order = ["baseline", "rural_highspeed", "motorcycle", "unrestrained"]
    labels = ["all drivers", "rural high-speed", "motorcyclist", "unrestrained"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    y = np.arange(len(order))[::-1]
    mc = marg.loc[order, "coverage"].values
    mo = mond.loc[order, "coverage"].values

    # left: coverage dumbbell, marginal (coral) -> Mondrian (teal)
    for yi, a, b in zip(y, mc, mo):
        axL.plot([a, b], [yi, yi], color=fs.SLATE, lw=1.2, zorder=1)
    axL.scatter(mc, y, s=58, color=fs.CORAL, zorder=3, label="marginal conformal")
    axL.scatter(mo, y, s=58, color=fs.ACCENT_TEAL, zorder=3,
                label="class-conditional (CHOIR)")
    axL.axvline(1 - alpha, color=fs.INK, ls=(0, (4, 3)), lw=0.9)
    axL.annotate(f"target {1-alpha:.2f}", xy=(1 - alpha, len(order) - 0.5),
                 xytext=(2, 0), textcoords="offset points", fontsize=6.8,
                 color=fs.SLATE, style="italic")
    for yi, v in zip(y, mc):
        axL.annotate(f"{v:.2f}", (v, yi), xytext=(0, -11),
                     textcoords="offset points", ha="center", fontsize=6.6,
                     color=fs.CORAL, fontweight="bold")
    axL.set_yticks(y); axL.set_yticklabels(labels)
    axL.set_xlim(0.35, 1.02); axL.set_xlabel("empirical coverage")
    axL.set_title("a  Marginal conformal undercovers the severe strata")
    axL.legend(loc="lower left", fontsize=6.8)

    # right: per reported-KABCO coverage bars, colored by shortfall
    kab = df[(df.exp == "E2") & (df.cell_type == "kabco")].set_index("cell")
    covk = kab.loc[KABCO, "coverage"].values
    short = np.clip((1 - alpha) - covk, 0, None)
    norm = plt.Normalize(0, max(short.max(), 1e-6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", fs.SEQ_TEAL[::-1])
    bars = axR.bar(KABCO, covk, color=[cmap(norm(s)) for s in short],
                   edgecolor="white", linewidth=0.7)
    axR.axhline(1 - alpha, color=fs.INK, ls=(0, (4, 3)), lw=0.9)
    for b, v in zip(bars, covk):
        axR.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 2), textcoords="offset points", ha="center",
                     fontsize=6.6, color=fs.INK)
    axR.set_ylim(0.7, 1.0); axR.set_xlabel("reported KABCO severity")
    axR.set_ylabel("empirical coverage")
    axR.set_title("b  Undercoverage concentrates on A and K")
    fig.tight_layout()
    save(fig, "fig2_conditional_failure")


# ---------------------------------------------------------------------------
# Fig 4: noise guarantee curves (E4) — structure buys tightness
# ---------------------------------------------------------------------------
def fig_noise_curves():
    df = read_result("e4_noise.parquet")
    alpha = float(df["alpha"].iloc[0])
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    cat = df[df.band == "kabco_catdep"].sort_values("delta")
    const = df[df.band == "constant_b1"].sort_values("delta")
    d = cat["delta"].values

    # guaranteed floor band 1-alpha-delta
    ax.fill_between(d, cat["floor"], 1.0, color=fs.MIST, zorder=0,
                    label="guaranteed region")
    ax.plot(d, cat["floor"], color=fs.INK, lw=1.1, ls=(0, (4, 3)),
            label=r"banded floor $1-\alpha-\delta$")
    # unstructured (generic) floor 1-alpha-eps_tot
    ax.plot(d, cat["floor_unstructured"], color=fs.CORAL, lw=1.4,
            label=r"generic floor $1-\alpha-\varepsilon_{\mathrm{tot}}$")
    # empirical true-label coverage after expansion
    ax.plot(d, cat["coverage"], "o-", color=fs.ACCENT_TEAL, ms=4.5, lw=1.6,
            label="empirical (category-dependent band)")
    ax.plot(const["delta"], const["coverage"], "s--", color=fs.HDR_BLUE, ms=3.8,
            lw=1.1, label="empirical (constant band)")
    ax.set_xlabel(r"declared beyond-band mass $\delta$")
    ax.set_ylabel("coverage on true severity")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Banded structure keeps the guarantee near-nominal")
    ax.legend(loc="lower left", fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig4_noise_curves")


# ---------------------------------------------------------------------------
# Fig 5: temporal timeline + recovery (E5)
# ---------------------------------------------------------------------------
def fig_temporal():
    df = read_result("e5_temporal.parquet")
    alpha = float(df["alpha"].iloc[0])
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    styles = {"unweighted": (fs.CORAL, "o", "unweighted split CP"),
              "county_mondrian_4a": (fs.HDR_BLUE, "s", "county-Mondrian (Thm 4a)"),
              "density_ratio_4b": (fs.ACCENT_TEAL, "^", "density-ratio weighted (Thm 4b)")}
    for method, (c, m, lab) in styles.items():
        g = df[df.method == method].sort_values("year")
        ax.errorbar(g["year"], g["coverage"], yerr=3 * g["se"], fmt=m + "-",
                    color=c, ms=5, lw=1.4, capsize=2.5, label=lab)
    fs.nominal_line(ax, 1 - alpha)
    ax.set_xticks([2024, 2025]); ax.set_xlabel("deployment year")
    ax.set_ylabel("coverage"); ax.set_ylim(0.86, 0.93)
    ax.set_title("Coverage holds across deployment years")
    ax.legend(loc="lower center", ncol=1, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig5_temporal")


# ---------------------------------------------------------------------------
# Fig: model-agnosticism (E1) — validity flat, efficiency varies
# ---------------------------------------------------------------------------
def fig_model_agnostic():
    df = read_result("e1_marginal.parquet")
    g = df[df.alpha == 0.10].copy()
    names = {"histgb": "HistGB", "ordered_logit": "ordered logit",
             "multinomial_lr": "multinomial LR", "lc_logit": "latent-class logit",
             "rp_logit": "random-params logit", "dlcon": "DLCON", "tabpfn": "TabPFN"}
    g = g[g.model.isin(names)].copy()
    g["label"] = g.model.map(names)
    g = g.sort_values("avg_width")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    yy = np.arange(len(g))
    ax.barh(yy, g["avg_width"], color=fs.MIST, edgecolor=fs.ACCENT_TEAL,
            linewidth=1.0, zorder=2, height=0.62)
    for i, (w, cov) in enumerate(zip(g["avg_width"], g["coverage"])):
        ax.annotate(f"{w:.2f}", (w, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=fs.INK, fontweight="bold")
        ax.annotate(f"cov {cov:.3f}", (0.05, i), xytext=(0, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=6.4, color=fs.SLATE)
    ax.set_yticks(yy); ax.set_yticklabels(g["label"])
    ax.set_xlabel("mean prediction-set width (categories)")
    ax.set_xlim(0, g["avg_width"].max() * 1.18)
    ax.set_title("One certificate, seven models: validity fixed, width varies")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_model_agnostic")


if __name__ == "__main__":
    fig_conditional_failure()
    fig_noise_curves()
    fig_temporal()
    fig_model_agnostic()
    print("figures ->", OUT)
