"""Calibration-draw variability for the final-cell certificate and the risk bounds.

The base models are fit on the S1 training fold and held fixed (their cached CDFs are
reused). The calibration and test folds are pooled and re-split at random into two halves
R times; each repeat recalibrates on one half and evaluates on the other. Because the base
model never sees these records, every repeat is a valid split-conformal experiment, and the
spread across repeats estimates the variability over calibration draws that the single-split
intervals in the paper do not include.
"""
import json
import numpy as np
import pandas as pd
from common import SEED, RESULTS, load_primary, split_s1, prepared_cdfs
from e2_e3_heterogeneity import declared_partition
from choir import cumulative_score, interval_sets, mondrian_calibrate, crc_threshold

ALPHA = 0.10
R_COV = 200
R_RISK = 100
MODELS = ["tabpfn"]
CELLS = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]
BETAS = [0.0005, 0.000624, 0.00078, 0.00097, 0.00121, 0.00151, 0.00188, 0.00234, 0.00292,
         0.00365, 0.00455, 0.01, 0.02, 0.05, 0.10]


def main():
    df = load_primary()
    tr, ca, te = split_s1(df)
    y = np.concatenate([ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()]).astype(int)
    cell = np.concatenate([declared_partition(ca), declared_partition(te)])
    n = len(y)
    rng = np.random.default_rng(SEED + 101)
    perms = [rng.permutation(n) for _ in range(R_COV)]
    rows = []
    for m in MODELS:
        cdf_ca, cdf_te, _ = prepared_cdfs("s1", m, tr, ca, te, X=(None, None, None))
        cdf = np.concatenate([cdf_ca, cdf_te]).astype(np.float64)
        s = cumulative_score(cdf, y)
        for r, p in enumerate(perms):
            a, b = p[: n // 2], p[n // 2:]
            q = mondrian_calibrate(s[a], cell[a], ALPHA)
            lam = np.array([q[c] for c in cell[b]])
            lo, hi = interval_sets(cdf[b], lam)
            cov = (y[b] >= lo) & (y[b] <= hi)
            for c in CELLS:
                mk = cell[b] == c
                rows.append({"model": m, "repeat": r, "cell": c, "coverage": float(cov[mk].mean()),
                             "width": float((hi[mk] - lo[mk] + 1).mean()), "n": int(mk.sum())})
        print(m, "coverage repeats done", flush=True)
    cov = pd.DataFrame(rows)
    cov.to_parquet(RESULTS / "r1_repeated_coverage_tabpfn.parquet", index=False)

    summ = cov.groupby(["model", "cell"]).coverage.agg(
        mean="mean", sd="std", q025=lambda x: x.quantile(0.025), q975=lambda x: x.quantile(0.975),
        min="min").reset_index()
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
