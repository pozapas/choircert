#!/usr/bin/env python
"""E1: marginal validity + contiguity on S1 (CHOIR_framework.md 3.4).

The held-out test coverage screen is descriptive only. It does not audit the
split-conformal guarantee because it omits calibration-sample uncertainty and does
not control multiplicity across the model-by-level grid. Package simulation tests
provide the theorem-regime check. This script records observed test performance.
"""
import json
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s1, s1_split_audit,
                    make_encoder, encode, prepared_cdfs, coverage_stats)
from choir import cumulative_score, interval_sets, split_calibrate

ALPHAS = [0.05, 0.10, 0.20]
# dlcon and tabpfn CDFs come from the Colab cache (colab/README.md); prepared_cdfs
# raises on a cache miss for them since they are not in the local registry.
MODELS = ["histgb", "ordered_logit", "multinomial_lr", "dlcon", "tabpfn",
          "lc_logit", "rp_logit"]


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    split_audit = s1_split_audit(df)
    (RESULTS / "s1_split_audit.json").write_text(json.dumps(split_audit, indent=2))
    print(f"S1 sizes: train={len(tr):,} cal={len(ca):,} test={len(te):,} "
          f"(load+split {time.time()-t0:.0f}s)")

    enc = make_encoder().fit(tr)
    X = (encode(enc, tr), encode(enc, ca), encode(enc, te))
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    print(f"design matrix: {X[0].shape[1]} cols (encode {time.time()-t0:.0f}s)", flush=True)

    rows = []
    for name in MODELS:
        cdf_ca, cdf_te, timing = prepared_cdfs("s1", name, tr, ca, te, X=X)
        fit_s, pred_s = timing["fit_s"], timing["predict_s"]
        print(f"{name}: fit {fit_s:.0f}s predict {pred_s:.0f}s "
              f"(cached={timing.get('cached', False)})", flush=True)
        s_ca = cumulative_score(cdf_ca, y_ca)
        for alpha in ALPHAS:
            t1 = time.time()
            qhat = split_calibrate(s_ca, alpha)
            lo, hi = interval_sets(cdf_te, qhat)
            cal_s = time.time() - t1
            st = coverage_stats(y_te, lo, hi)
            # This is retained only as a descriptive, test-only screen. It must not
            # be read as empirical acceptance of the finite-sample guarantee.
            screen_tol = 3 * np.sqrt(alpha * (1 - alpha) / st["n"])
            screen_lower = st["coverage"] >= 1 - alpha - screen_tol
            at_floor = st["avg_width"] <= 1.0 + 1e-9
            screen_upper = (at_floor or st["coverage"] <=
                            1 - alpha + 1 / (len(ca) + 1) + screen_tol)
            rows.append({"model": name, "alpha": alpha, **st, "qhat": qhat,
                         "test_screen_within_band": bool(screen_lower and screen_upper),
                         "test_screen_type": "descriptive_test_only",
                         "singleton_floor": bool(at_floor),
                         "fit_s": fit_s, "predict_s": pred_s,
                         "calibrate_s": cal_s, "seed": SEED})
            print(f"{name} a={alpha}: cov={st['coverage']:.4f} "
                  f"width={st['avg_width']:.2f} "
                  f"test_screen={bool(screen_lower and screen_upper)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "e1_marginal.parquet", index=False)
    (RESULTS / "e1_summary.json").write_text(json.dumps(
        {"test_screen_is_not_acceptance": True,
         "all_test_screens_within_band": bool(out["test_screen_within_band"].all()),
         "theorem_regime_check": "choir/tests/test_theorems.py",
         "total_s": time.time() - t0}, indent=2))
    print(f"E1 done in {time.time()-t0:.0f}s; "
          f"all_test_screens_within_band={out['test_screen_within_band'].all()}")


if __name__ == "__main__":
    main()
