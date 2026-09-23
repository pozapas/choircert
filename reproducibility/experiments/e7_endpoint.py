#!/usr/bin/env python
"""E7 completed-record endpoint audit.

This script uses the same S1 folds, frozen safety cells, and reported-label conformal
thresholds as the main analysis. It reports a descriptive A-or-K endpoint rule for the
raw reported-label interval and the kabco-main-v1 sensitivity expansion.
"""

import gc
import json
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s1, s1_split_audit,
                    prepared_cdfs)
from choir import cumulative_score, interval_sets, mondrian_calibrate
from choir.noise import NoiseModel
from e2_e3_heterogeneity import declared_partition

ALPHA = 0.10
DELTA = 0.02
MODEL = "histgb"
MAP_VERSION = "kabco-main-v1"


def _metrics(y, lo, hi, mask):
    y_m = np.asarray(y)[mask]
    lo_m = np.asarray(lo)[mask]
    hi_m = np.asarray(hi)[mask]
    flag = hi_m >= 4
    severe = y_m >= 4
    missed = severe & ~flag
    return {
        "n": int(len(y_m)),
        "n_severe_AK": int(severe.sum()),
        "n_flagged_endpoint_AK": int(flag.sum()),
        "n_severe_omitted": int(missed.sum()),
        "workload": float(flag.mean()),
        "conditional_severe_omission": float(missed.sum() / severe.sum()),
        "joint_severe_omission": float(missed.mean()),
        "avg_width": float(np.mean(hi_m.astype(np.int16) - lo_m.astype(np.int16) + 1)),
    }


def main():
    t0 = time.time()
    df = load_primary()
    split_audit = s1_split_audit(df)
    tr, ca, te = split_s1(df)
    y_ca = ca["y_kabco"].to_numpy(dtype=np.int8)
    y_te = te["y_kabco"].to_numpy(dtype=np.int8)
    cell_ca = declared_partition(ca)
    cell_te = declared_partition(te)
    cdf_ca, cdf_te, timing = prepared_cdfs("s1", MODEL, tr, ca, te)
    del df, tr, ca, te
    gc.collect()

    scores = cumulative_score(cdf_ca, y_ca)
    q = mondrian_calibrate(scores, cell_ca, ALPHA)
    thresholds = np.array([q[cell] for cell in cell_te])
    lo_raw, hi_raw = interval_sets(cdf_te, thresholds)
    lo_raw = lo_raw.astype(np.int8, copy=False)
    hi_raw = hi_raw.astype(np.int8, copy=False)

    noise = NoiseModel.kabco(delta=DELTA, exact_fatal=False, a_reaches_down=2)
    lo_exp, hi_exp = noise.expand(lo_raw, hi_raw)
    lo_exp = lo_exp.astype(np.int8, copy=False)
    hi_exp = hi_exp.astype(np.int8, copy=False)

    rows = []
    for set_type, lo, hi in (
        ("raw_reported_interval", lo_raw, hi_raw),
        ("expanded_sensitivity_interval", lo_exp, hi_exp),
    ):
        for cell in [*np.unique(cell_te), "overall"]:
            mask = (np.ones(len(y_te), dtype=bool) if cell == "overall"
                    else cell_te == cell)
            rows.append({
                "set_type": set_type,
                "final_cell": cell,
                "endpoint_rule": "flag when upper endpoint is A or K",
                "outcome": "observed police-reported sampled-driver KABCO",
                "scope": "completed-record descriptive audit",
                **_metrics(y_te, lo, hi, mask),
            })

    out = pd.DataFrame(rows)
    out["alpha"] = ALPHA
    out["delta"] = DELTA
    out["model"] = MODEL
    out["map_version"] = MAP_VERSION
    out["seed"] = SEED
    out.to_parquet(RESULTS / "e7_endpoint_audit.parquet", index=False)

    audit = {
        "split": split_audit,
        "model": MODEL,
        "cached_prediction_timing": timing,
        "alpha": ALPHA,
        "delta": DELTA,
        "compatibility_map_version": MAP_VERSION,
        "endpoint_rule": "flag when upper endpoint is A or K",
        "workload_denominator": "all test records",
        "conditional_severe_omission_denominator": "all observed A or K test outcomes",
        "joint_severe_omission_denominator": "all test records",
        "interpretation": "completed-record descriptive audit",
    }
    (RESULTS / "e7_endpoint_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True))
    print(out.to_string(index=False))
    print(f"E7 endpoint audit done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
