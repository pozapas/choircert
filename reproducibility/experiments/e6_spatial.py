#!/usr/bin/env python
"""E6: spatial transfer (S3). Calibrate on 200 counties, test on 54 held-out counties.

Per-held-out-county certificates: unweighted coverage, density-ratio weighted coverage,
and the Thm 4b three-number certificate (nominal, TV-slack LCB, weighted coverage).
Output feeds the Texas county map figure (F6).
"""
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s3, make_encoder, encode,
                    prepared_cdfs, coverage_stats)
from choir import cumulative_score, interval_sets, split_calibrate
from choir.shift import DensityRatioEstimator, tilted_resample, tv_slack_lcb, weighted_thresholds

ALPHA = 0.10
MODEL = "histgb"
MIN_COUNTY_TEST = 2000   # certificate granularity floor for the map


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te, holdout = split_s3(df)
    print(f"S3 sizes: train={len(tr):,} cal={len(ca):,} test={len(te):,} "
          f"({len(holdout)} held-out counties)")

    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cnty_te = te["Cnty_ID"].astype(str).to_numpy()

    cdf_ca, cdf_te, _ = prepared_cdfs("s3", MODEL, tr, ca, te,
                                      X=(X_tr, X_ca, X_te))
    s_ca = cumulative_score(cdf_ca, y_ca)
    qhat = split_calibrate(s_ca, ALPHA)
    lo_u, hi_u = interval_sets(cdf_te, qhat)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    from sklearn.linear_model import LogisticRegression
    rows = []
    for cnty in sorted(set(cnty_te)):
        m = cnty_te == cnty
        n_c = int(m.sum())
        row = {"county": cnty, "n_test": n_c, "alpha": ALPHA}
        row.update({f"unw_{k}": v for k, v in
                    coverage_stats(y_te[m], lo_u[m], hi_u[m]).items()})
        if n_c >= MIN_COUNTY_TEST:
            n_fit = min(200_000, len(X_tr))
            src = X_tr[rng.choice(len(X_tr), n_fit, replace=False)]
            # SPLIT DISCIPLINE (Thm 4b hypothesis -- do not regress this).
            # Thm 4b requires w_hat to be FIXED given the training split and the
            # unlabeled target covariates: it may not be a function of the test point
            # X_{n+1} whose interval it weights. So the county's target rows are split
            # once into disjoint halves:
            #   fit_idx -> fits w_hat (these covariates enter the ratio model)
            #   rest    -> prediction + evaluation, and the TV-slack diagnostic
            # The weighted arm must therefore NEVER be scored on fit_idx: a fit_idx row's
            # own covariates shaped the weights it would be evaluated under, which is
            # exactly the dependence Thm 4b forbids. Prediction and the diagnostic may
            # share `rest` -- the diagnostic is label-blind (tv_slack_lcb sees only X)
            # and its estimand d_TV(P*_X, Q_w) needs independence from w_hat's fit
            # sample, which `rest` already provides; it needs nothing from the labels.
            perm = rng.permutation(np.where(m)[0])
            fit_idx, rest = perm[: len(perm) // 2], perm[len(perm) // 2:]
            tgt = X_te[fit_idx]
            dre = DensityRatioEstimator(
                clf=LogisticRegression(max_iter=2000, random_state=SEED), clip=50.0,
            ).fit(src, tgt)
            w_ca = dre.weights(X_ca)
            lam_w = weighted_thresholds(s_ca, w_ca, dre.weights(X_te[rest]), ALPHA)
            lo_w, hi_w = interval_sets(cdf_te[rest], lam_w)
            row.update({f"w_{k}": v for k, v in
                        coverage_stats(y_te[rest], lo_w, hi_w).items()})
            # Paired like-for-like: the unweighted arm restricted to the very rows the
            # weighted arm is scored on. The headline unw_* above stays on the full
            # county (it fits no density ratio, so it has no split constraint and keeps
            # its full-n precision); these unwr_* columns let the weighted-vs-unweighted
            # contrast be read off identical rows rather than across populations.
            row.update({f"unwr_{k}": v for k, v in
                        coverage_stats(y_te[rest], lo_u[rest], hi_u[rest]).items()})
            n_diag = min(20_000, len(rest))
            tilt = tilted_resample(X_ca, w_ca, n_diag, rng)
            diag_idx = rest[:n_diag]
            diag = tv_slack_lcb(X_te[diag_idx], tilt,
                                clf=LogisticRegression(max_iter=2000, random_state=SEED),
                                rng=rng)
            row["tv_lcb"] = diag["tv_lcb"]
            row["cert_floor_lcb"] = 1 - ALPHA - diag["tv_lcb"]
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} counties done ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out["model"] = MODEL
    out["seed"] = SEED
    out.to_parquet(RESULTS / "e6_spatial.parquet", index=False)
    big = out[out.n_test >= MIN_COUNTY_TEST]
    print(f"E6 done in {time.time()-t0:.0f}s; {len(out)} counties, "
          f"{len(big)} certified; unweighted cov range "
          f"[{out.unw_coverage.min():.3f}, {out.unw_coverage.max():.3f}]")


if __name__ == "__main__":
    main()
