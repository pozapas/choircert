#!/usr/bin/env python
"""Round-3 Task 2 (scoping, not part of the pipeline): pre-declared alpha-frontier
and beta-frontier for three actionable informativeness events, on COMPOSED sets
(class-conditional + band expansion, the e8_composition.py stack).

FROZEN event definitions (do not change after seeing results):
  excl_fatal  : 5 not in C  <=> hi <= 4
  b_or_worse  : min(C) >= 3 <=> lo >= 3
  width_le_3  : |C| <= 3    <=> hi - lo + 1 <= 3

alpha* for (stratum, event) = smallest alpha on the grid with P(event|stratum) >= 0.5.

Partition: declared_partition only (strata=None) -- NOT crossed with rural/urban.
Bands reported: both constant_b1 and kabco_catdep (the paper's efficient/headline arm).

This is a NEW, standalone diagnostic script; it does not modify or overwrite any
existing experiment file or its results parquet.
"""
import time

import numpy as np
import pandas as pd

from common import SEED, RESULTS, load_primary, split_s1, prepared_cdfs, coverage_stats
from choir import CertifiedOrdinal, NoiseModel
from choir.risk import crc_threshold
from choir.crash.costs import usdot_relative, fatal_omission
from e2_e3_heterogeneity import declared_partition
from e4_noise import inject_noise

ALPHA_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BETA_GRID = [0.01, 0.02, 0.05, 0.10]
DELTA = 0.02
MODEL = "histgb"
N_MIN = 1000
STRATA = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]


def events(lo, hi):
    return {
        "excl_fatal": hi <= 4,
        "b_or_worse": lo >= 3,
        "width_le_3": (hi - lo + 1) <= 3,
    }


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)

    rng = np.random.default_rng(SEED + 8)
    yt_ca, eps = inject_noise(y_ca, DELTA, rng)
    part_ca, part_te = declared_partition(ca), declared_partition(te)
    print(f"setup {time.time()-t0:.0f}s; eps_tot={eps:.4f}", flush=True)

    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]
    holder = {"labels": part_ca}
    partition_fn = lambda X: holder["labels"]

    bands = {
        "constant_b1": NoiseModel(K=5, b_plus=1, b_minus=1, delta=DELTA),
        "kabco_catdep": NoiseModel.kabco(delta=DELTA, exact_fatal=True, a_reaches_down=2),
    }

    # ---------------- alpha-frontier on composed sets ----------------
    alpha_rows = []
    for band, nm in bands.items():
        cert = CertifiedOrdinal(base=base, K=5, partition=partition_fn, noise=nm, n_min=N_MIN)
        holder["labels"] = part_ca
        cert.calibrate(X_ca, yt_ca)
        for alpha in ALPHA_GRID:
            holder["labels"] = part_te
            lo, hi = cert.predict_set(X_te, alpha=alpha)
            ev = events(lo, hi)
            for s in STRATA:
                m = part_te == s
                st = coverage_stats(y_te[m], lo[m], hi[m])
                row = {"band": band, "alpha": alpha, "stratum": s,
                       "coverage": st["coverage"], "avg_width": st["avg_width"], "n": st["n"]}
                for ename, evec in ev.items():
                    row[f"P_{ename}"] = float(evec[m].mean())
                alpha_rows.append(row)
        print(f"[{band}] alpha-frontier done ({time.time()-t0:.0f}s)", flush=True)

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df["delta"], alpha_df["model"], alpha_df["seed"] = DELTA, MODEL, SEED
    alpha_df.to_parquet(RESULTS / "adhoc_r3_task2_alpha_frontier.parquet", index=False)

    # alpha* per (band, stratum, event): smallest alpha with P(E|A) >= 0.5
    star_rows = []
    for band in bands:
        for s in STRATA:
            for ename in ("excl_fatal", "b_or_worse", "width_le_3"):
                sub = alpha_df[(alpha_df.band == band) & (alpha_df.stratum == s)] \
                        .sort_values("alpha")
                hits = sub[sub[f"P_{ename}"] >= 0.5]
                astar = float(hits.alpha.iloc[0]) if len(hits) else float("nan")
                star_rows.append({"band": band, "stratum": s, "event": ename,
                                   "alpha_star": astar,
                                   "never_reaches_0.5_on_grid": len(hits) == 0})
    star_df = pd.DataFrame(star_rows)
    star_df.to_parquet(RESULTS / "adhoc_r3_task2_alpha_star.parquet", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("\n--- alpha* (smallest alpha with P(E|A) >= 0.5) ---")
    print(star_df.to_string(index=False))

    # ---------------- beta-frontier for risk control (on composed sets) ----------------
    kappa = usdot_relative()
    kfatal = fatal_omission()
    beta_rows = []
    for band, nm in bands.items():
        cert = CertifiedOrdinal(base=base, K=5, partition=partition_fn, noise=nm, n_min=N_MIN)
        holder["labels"] = part_ca
        cert.calibrate(X_ca, yt_ca)
        for beta in BETA_GRID:
            holder["labels"] = part_te
            lo_c, hi_c = cert.predict_set_risk(X_te, beta=beta, kappa=kappa)
            lo_f, hi_f = cert.predict_set_risk(X_te, beta=beta, kappa=kfatal)
            for s in STRATA:
                m = part_te == s
                miss_c = ~((y_te[m] >= lo_c[m]) & (y_te[m] <= hi_c[m]))
                risk_c = float(np.mean(kappa[y_te[m] - 1] * miss_c))
                w_c = hi_c[m] - lo_c[m] + 1
                miss_f = (y_te[m] == 5) & ~((y_te[m] >= lo_f[m]) & (y_te[m] <= hi_f[m]))
                risk_f = float(miss_f.mean())
                w_f = hi_f[m] - lo_f[m] + 1
                beta_rows.append({
                    "band": band, "beta": beta, "stratum": s, "n": int(m.sum()),
                    "cost_risk": risk_c, "cost_bound": beta * kappa[-1],
                    "cost_holds": bool(risk_c <= beta * kappa[-1]),
                    "cost_avg_width": float(w_c.mean()),
                    "cost_frac_full_scale": float((w_c == 5).mean()),
                    "fatal_risk": risk_f, "fatal_bound": beta,
                    "fatal_holds": bool(risk_f <= beta),
                    "fatal_avg_width": float(w_f.mean()),
                    "fatal_frac_full_scale": float((w_f == 5).mean()),
                    "n_fatal": int((y_te[m] == 5).sum()),
                })
        print(f"[{band}] beta-frontier done ({time.time()-t0:.0f}s)", flush=True)

    beta_df = pd.DataFrame(beta_rows)
    beta_df["delta"], beta_df["model"], beta_df["seed"] = DELTA, MODEL, SEED
    beta_df.to_parquet(RESULTS / "adhoc_r3_task2_beta_frontier.parquet", index=False)
    print("\n--- beta-frontier (risk control on composed sets) ---")
    print(beta_df[["band", "beta", "stratum", "cost_risk", "cost_bound", "cost_holds",
                    "cost_frac_full_scale", "fatal_risk", "fatal_bound", "fatal_holds",
                    "fatal_frac_full_scale", "n"]].to_string(index=False))

    # cross-reference: at alpha=0.10, kabco_catdep, is the composed coverage set
    # full-scale on the same strata where the beta=0.10 risk set is also full-scale?
    print("\n--- cross-reference: alpha=0.10 coverage-set full-scale frac vs "
          "beta=0.10 risk-set full-scale frac (kabco_catdep band) ---")
    a10 = alpha_df[(alpha_df.band == "kabco_catdep") & (alpha_df.alpha == 0.10)]
    b10 = beta_df[(beta_df.band == "kabco_catdep") & (beta_df.beta == 0.10)]
    for s in STRATA:
        cov_row = a10[a10.stratum == s].iloc[0]
        risk_row = b10[b10.stratum == s].iloc[0]
        print(f"{s}: coverage-set P(width_le_3)={cov_row['P_width_le_3']:.6f} "
              f"cost_risk_frac_full_scale={risk_row['cost_frac_full_scale']:.6f} "
              f"fatal_risk_frac_full_scale={risk_row['fatal_frac_full_scale']:.6f}")

    print(f"\nTask2 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
