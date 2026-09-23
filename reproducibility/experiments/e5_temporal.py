#!/usr/bin/env python
"""E5: temporal deployment shift (S2). Train 2017-21, calibrate 2022-23, test 2024-25.

(1) Quantify unweighted split-conformal degradation per test year (a finding itself).
(2) Thm 4a recovery: county-Mondrian calibration (rollup to statewide below n_min) —
    exact under redistribution across observed county strata.
(3) Thm 4b recovery: density-ratio weighted calibration with weights fit on
    calibration covariates vs UNLABELED target-year covariates. The target year is
    split into independent density-ratio-fit and evaluation halves. The output labels
    the TV value as a weakness diagnostic, not as a target-coverage certificate.
"""
import json
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s2, make_encoder, encode,
                    prepared_cdfs, coverage_stats)
from choir import (cumulative_score, interval_sets, split_calibrate,
                   mondrian_calibrate)
from choir.shift import DensityRatioEstimator, tilted_resample, tv_slack_lcb, weighted_thresholds

ALPHA = 0.10
MODEL = "histgb"
N_MIN = 1000


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s2(df)
    print(f"S2 sizes: train={len(tr):,} cal={len(ca):,} test={len(te):,}")

    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    years = te["crash_year"].to_numpy()

    cdf_ca, cdf_te, _ = prepared_cdfs("s2", MODEL, tr, ca, te,
                                      X=(X_tr, X_ca, X_te))
    s_ca = cumulative_score(cdf_ca, y_ca)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    rows = []
    ratio_audit = {
        "source_distribution": "S2 calibration covariates, years 2022-2023",
        "target_distribution": "unlabeled target-year covariates",
        "target_split": "disjoint random fit and evaluation halves within each target year",
        "seed": SEED,
        "years": {},
    }

    # ---- (1) unweighted ----
    qhat = split_calibrate(s_ca, ALPHA)
    lo, hi = interval_sets(cdf_te, qhat)
    for yr in (2024, 2025):
        m = years == yr
        rows.append({"method": "unweighted", "year": yr,
                     **coverage_stats(y_te[m], lo[m], hi[m])})

    # ---- (2) Thm 4a: county-Mondrian with statewide rollup ----
    cnty_ca = ca["Cnty_ID"].astype(str).to_numpy()
    cnty_te = te["Cnty_ID"].astype(str).to_numpy()
    counts = pd.Series(cnty_ca).value_counts()
    big = set(counts[counts >= N_MIN].index)
    cells_ca = np.where(pd.Series(cnty_ca).isin(big), cnty_ca, "STATE")
    cells_te = np.where(pd.Series(cnty_te).isin(big), cnty_te, "STATE")
    q = mondrian_calibrate(s_ca, cells_ca, ALPHA)
    lam = np.array([q.get(c, q["STATE"]) for c in cells_te])
    lo4a, hi4a = interval_sets(cdf_te, lam)
    for yr in (2024, 2025):
        m = years == yr
        rows.append({"method": "county_mondrian_4a", "year": yr,
                     **coverage_stats(y_te[m], lo4a[m], hi4a[m])})

    # ---- (3) Thm 4b: density-ratio weighted + certificate ----
    rng = np.random.default_rng(SEED)
    for yr in (2024, 2025):
        m = years == yr
        # SPLIT DISCIPLINE. The required ratio is target-to-calibration, because it
        # weights calibration scores. The target-year rows are split before fitting.
        # `fit_idx` supplies unlabeled target covariates to the ratio estimator, while
        # `rest` is independent target evaluation data. No result is reported on a row
        # whose covariates helped fit its density-ratio model.
        n_fit = 300_000
        perm = rng.permutation(np.where(m)[0])
        n_fit_eff = min(n_fit, len(perm) // 2)
        fit_idx, rest = perm[:n_fit_eff], perm[n_fit_eff:]
        tgt = X_te[fit_idx]
        from sklearn.linear_model import LogisticRegression
        dre = DensityRatioEstimator(
            clf=LogisticRegression(max_iter=2000, C=1.0, random_state=SEED), clip=50.0,
        ).fit(X_ca, tgt)
        w_ca = dre.weights(X_ca)
        w_te = dre.weights(X_te[rest])
        lam_w = weighted_thresholds(s_ca, w_ca, w_te, ALPHA)
        lo4b, hi4b = interval_sets(cdf_te[rest], lam_w)
        st = coverage_stats(y_te[rest], lo4b, hi4b)
        # Paired like-for-like: the unweighted arm on the very rows the weighted arm is
        # scored on. The headline unweighted / 4a rows above stay on the full year (they
        # fit no density ratio, so they carry no split constraint and keep full-n
        # precision); this column makes the 4b-vs-unweighted contrast row-identical.
        unw_same = coverage_stats(y_te[rest], lo[rest], hi[rest])
        # certificate diagnostic: TV-LCB between target-X and w-tilted calibration-X
        # (diagnostic rows drawn from the half not used to fit w-hat)
        n_diag = min(60_000, len(rest))
        tilt = tilted_resample(X_ca, w_ca, n_diag, rng)
        diag_idx = rest[:n_diag]
        diag = tv_slack_lcb(X_te[diag_idx], tilt,
                            clf=LogisticRegression(max_iter=2000, random_state=SEED),
                            rng=rng)
        rows.append({"method": "density_ratio_4b", "year": yr, **st,
                     "tv_lcb": diag["tv_lcb"], "ba": diag["ba"],
                     "best_case_floor_upper_from_tv_lcb": 1 - ALPHA - diag["tv_lcb"],
                     "unw_same_rows_coverage": unw_same["coverage"],
                     "unw_same_rows_n": unw_same["n"]})
        ratio_audit["years"][str(yr)] = {
            "calibration_source_n": int(len(X_ca)),
            "target_fit_n": int(len(fit_idx)),
            "target_evaluation_n": int(len(rest)),
            "target_fit_and_evaluation_overlap": int(len(np.intersect1d(fit_idx, rest))),
            "ratio": "dP_target_X / dP_calibration_X",
            "diagnostic": (
                "tv_lcb is a lower confidence bound on residual covariate mismatch. "
                "It can certify weakness. It cannot certify target coverage."
            ),
        }
        print(f"4b year {yr}: cov={st['coverage']:.4f} tv_lcb={diag['tv_lcb']:.4f}")

    out = pd.DataFrame(rows)
    out["alpha"] = ALPHA
    out["model"] = MODEL
    out["seed"] = SEED
    out.to_parquet(RESULTS / "e5_temporal.parquet", index=False)
    (RESULTS / "e5_density_ratio_audit.json").write_text(
        json.dumps(ratio_audit, indent=2, sort_keys=True))
    print(out[["method", "year", "coverage", "avg_width", "n"]].to_string(index=False))
    print(f"E5 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
