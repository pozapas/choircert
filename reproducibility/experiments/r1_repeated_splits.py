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
MODELS = ["ordered_logit", "multinomial_lr", "lc_logit", "rp_logit", "histgb", "dlcon", "tabpfn"]
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
    cov.to_parquet(RESULTS / "r1_repeated_coverage.parquet", index=False)

    # risk control for the gradient boosting anchor across repeats
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", "histgb", tr, ca, te, X=(None, None, None))
    cdf = np.concatenate([cdf_ca, cdf_te]).astype(np.float64)
    s_all = None
    from choir import score_matrix
    S = score_matrix(cdf)  # (n, K) score of every candidate label
    s_obs = S[np.arange(n), y - 1]
    fatal = (y == 5).astype(float)
    risk_rows = []
    for r, p in enumerate(perms[:R_RISK]):
        a, b = p[: n // 2], p[n // 2:]
        for beta in BETAS:
            lam = crc_threshold(s_obs[a], fatal[a], kappa_max=1.0, beta=beta)
            lo_f, hi_f = interval_sets(cdf[b], lam)
            miss = (y[b] == 5) & ~((y[b] >= lo_f) & (y[b] <= hi_f))
            risk_rows.append({"repeat": r, "beta": beta, "lambda": float(lam),
                              "joint_fatal_omission": float(miss.mean()),
                              "conditional_fatal_omission": float(miss.sum() / max(1, (y[b] == 5).sum()))})
    risk = pd.DataFrame(risk_rows)
    risk.to_parquet(RESULTS / "r1_repeated_risk.parquet", index=False)

    summ = cov.groupby(["model", "cell"]).coverage.agg(
        mean="mean", sd="std", q025=lambda x: x.quantile(0.025), q975=lambda x: x.quantile(0.975),
        min="min").reset_index()
    rs = risk.groupby("beta").agg(mean_joint=("joint_fatal_omission", "mean"),
                                  q975_joint=("joint_fatal_omission", lambda x: x.quantile(0.975)),
                                  mean_cond=("conditional_fatal_omission", "mean")).reset_index()
    rs["bound"] = rs.beta
    rs["mean_below_bound"] = rs.mean_joint <= rs.bound
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string()); print(rs.round(6).to_string())
    (RESULTS / "r1_summary.json").write_text(json.dumps({
        "coverage": summ.to_dict(orient="records"), "risk": rs.to_dict(orient="records")}, indent=1))


if __name__ == "__main__":
    main()
