#!/usr/bin/env python
"""E2 + E3: conditional-coverage failure of marginal conformal, and its repair by
heterogeneity-conditional (Mondrian) calibration (Thm 2 / Thm 2b diagnosis).

E2 (Fig 2 data): marginal conformal's coverage within declared safety-relevant strata
(motorcycle drivers, unrestrained, rural high-speed) and per reported KABCO category.
Expectation from Thm 2b(iii): undercoverage exactly on the hard strata.

E3: Mondrian on (a) the declared-strata partition and (b) a KMeans-8 latent proxy
partition restores per-cell coverage >= 1-alpha; report the honest set-size price.
Partition sources are fit/defined on the training split only (split discipline).
"""
import json
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s1, s1_split_audit,
                    make_encoder, encode, prepared_cdfs, coverage_stats)
from choir import (cumulative_score, interval_sets, split_calibrate,
                   mondrian_calibrate)

ALPHA = 0.10
MODEL = "histgb"

# Only "None" records a driver observed to be using no restraint. "Not Applicable" is
# an inapplicability code (it is the restraint value carried by motorcycle/non-standard
# occupants, for whom a restraint field does not apply) and "Other (Explain In
# Narrative)" is a residual bucket whose content lives in the narrative field, which is
# excluded at ingest. Neither is evidence of an unrestrained driver, so neither belongs
# in this stratum.
UNRESTRAINED = {"None"}


def declared_partition(df: pd.DataFrame) -> np.ndarray:
    """Priority-rule partition from pre-declared safety strata (fixed, covariate-only)."""
    moto = (df["Prsn_Type_ID"] == "Driver Of Motorcycle Type Vehicle").to_numpy()
    unre = df["Prsn_Rest_ID"].isin(UNRESTRAINED).to_numpy()
    rural_hs = ((df["Rural_Fl"] == "Y") & (df["speed_limit"] >= 55)).to_numpy()
    out = np.full(len(df), "baseline", dtype=object)
    out[rural_hs] = "rural_highspeed"
    out[unre] = "unrestrained"
    out[moto] = "motorcycle"
    return out


def _counts(labels: np.ndarray) -> dict:
    values, counts = np.unique(labels.astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()

    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te,
                                      X=(X_tr, X_ca, X_te))
    s_ca = cumulative_score(cdf_ca, y_ca)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    part_ca = declared_partition(ca)
    part_te = declared_partition(te)

    # KMeans latent proxy fit on TRAIN design matrix only
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=3, random_state=SEED).fit(
        X_tr[np.random.default_rng(SEED).choice(len(X_tr), 500_000, replace=False)])
    km_ca = km.predict(X_ca)
    km_te = km.predict(X_te)

    # This sidecar is the complete fixed-partition ledger needed to reproduce the
    # certified cells. It is intentionally separate from coverage results.
    partition_audit = {
        "split": s1_split_audit(df),
        "declared_partition": {
            "type": "priority_rule",
            "priority_high_to_low": ["motorcycle", "unrestrained", "rural_highspeed", "baseline"],
            "motorcycle": "Prsn_Type_ID == 'Driver Of Motorcycle Type Vehicle'",
            "unrestrained": "Prsn_Rest_ID in ['None']",
            "rural_highspeed": "Rural_Fl == 'Y' and speed_limit >= 55",
            "baseline": "all remaining rows",
            "calibration_counts": _counts(part_ca),
            "test_counts": _counts(part_te),
        },
        "kmeans8_partition": {
            "fit_split": "S1 training only",
            "features": "common.make_encoder() output fitted on S1 training",
            "n_clusters": 8,
            "n_init": 3,
            "random_state": SEED,
            "fit_rows": int(min(500_000, len(X_tr))),
            "sampling": "uniform without replacement from S1 training",
            "calibration_counts": _counts(km_ca),
            "test_counts": _counts(km_te),
        },
    }
    (RESULTS / "e2_e3_partition_audit.json").write_text(
        json.dumps(partition_audit, indent=2, sort_keys=True))

    # DLCON latent-class gate (the econometric-bridge partition; Colab cache)
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parent / "cache"
    dlcon_cls = cache_dir / "colab_outputs" / "s1_dlcon_classes.npz"
    dl_ca = dl_te = None
    if dlcon_cls.exists():
        zc = np.load(dlcon_cls)
        dl_ca, dl_te = zc["cls_ca"], zc["cls_te"]

    # classical latent-class ordered-logit gate (fit_econ.py cache)
    lc_cls = cache_dir / "s1_lc_classes.npz"
    lc_ca = lc_te = None
    if lc_cls.exists():
        zl = np.load(lc_cls)
        lc_ca, lc_te = zl["cls_ca"], zl["cls_te"]

    rows = []

    # ---- E2: marginal conformal, stratum-wise audit ----
    qhat = split_calibrate(s_ca, ALPHA)
    lo, hi = interval_sets(cdf_te, qhat)
    for stratum in np.unique(part_te):
        m = part_te == stratum
        rows.append({"exp": "E2", "method": "marginal", "cell_type": "declared",
                     "cell": stratum, **coverage_stats(y_te[m], lo[m], hi[m])})
    for k in range(1, 6):
        m = y_te == k
        rows.append({"exp": "E2", "method": "marginal", "cell_type": "kabco",
                     "cell": "OCBAK"[k - 1], **coverage_stats(y_te[m], lo[m], hi[m])})

    # ---- E3: Mondrian repairs, three partitions ----
    partitions = [("declared", part_ca, part_te), ("kmeans8", km_ca, km_te)]
    if dl_ca is not None:
        partitions.append(("dlcon_gate", dl_ca, dl_te))
    if lc_ca is not None:
        partitions.append(("lc_logit_gate", lc_ca, lc_te))
    for pname, p_ca, p_te in partitions:
        q = mondrian_calibrate(s_ca, p_ca, ALPHA)
        lam = np.array([q[c] for c in p_te])
        lo_m, hi_m = interval_sets(cdf_te, lam)
        for cell in np.unique(p_te):
            m = p_te == cell
            rows.append({"exp": "E3", "method": f"mondrian_{pname}",
                         "cell_type": pname, "cell": str(cell),
                         **coverage_stats(y_te[m], lo_m[m], hi_m[m])})
        rows.append({"exp": "E3", "method": f"mondrian_{pname}", "cell_type": "overall",
                     "cell": "ALL", **coverage_stats(y_te, lo_m, hi_m)})
    rows.append({"exp": "E2", "method": "marginal", "cell_type": "overall",
                 "cell": "ALL", **coverage_stats(y_te, lo, hi)})

    out = pd.DataFrame(rows)
    out["alpha"] = ALPHA
    out["model"] = MODEL
    out["seed"] = SEED
    out.to_parquet(RESULTS / "e2_e3_heterogeneity.parquet", index=False)
    print(out[["exp", "method", "cell", "coverage", "avg_width", "n"]].to_string(index=False))
    print(f"E2/E3 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
