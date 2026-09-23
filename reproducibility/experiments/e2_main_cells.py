#!/usr/bin/env python
"""Main 3-model by 4-cell reported-label audit."""

import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

from common import (SEED, RESULTS, load_primary, split_s1, s1_split_audit,
                    prepared_cdfs)
from choir import cumulative_score, interval_sets, mondrian_calibrate, split_calibrate
from e2_e3_heterogeneity import declared_partition

ALPHA = 0.10
FAMILY_ERROR = 0.05
MODELS = ["ordered_logit", "histgb", "dlcon"]
CELLS = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]


def clopper_pearson_simultaneous(successes, n, comparisons):
    """Two-sided interval with Bonferroni control for one predeclared family."""
    tail = FAMILY_ERROR / comparisons / 2
    lo = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - tail, successes + 1, n - successes))
    return lo, hi


def summarize(y, lo, hi, mask, comparisons):
    y_m = y[mask]
    lo_m = lo[mask]
    hi_m = hi[mask]
    covered = (y_m >= lo_m) & (y_m <= hi_m)
    successes = int(covered.sum())
    n = int(len(y_m))
    ci_lo, ci_hi = clopper_pearson_simultaneous(successes, n, comparisons)
    return {
        "n": n,
        "covered": successes,
        "coverage": float(successes / n),
        "simultaneous_ci_low": ci_lo,
        "simultaneous_ci_high": ci_hi,
        "avg_width": float(np.mean(hi_m.astype(np.int16) - lo_m.astype(np.int16) + 1)),
    }


def main():
    t0 = time.time()
    df = load_primary()
    split_audit = s1_split_audit(df)
    tr, ca, te = split_s1(df)
    y_ca = ca["y_kabco"].to_numpy(dtype=np.int8)
    y_te = te["y_kabco"].to_numpy(dtype=np.int8)
    cell_tr = declared_partition(tr)
    cell_ca = declared_partition(ca)
    cell_te = declared_partition(te)
    cell_counts = {
        fold: {cell: int(np.sum(labels == cell)) for cell in CELLS}
        for fold, labels in (("training", cell_tr), ("calibration", cell_ca),
                             ("test", cell_te))
    }

    rows = []
    comparisons = len(MODELS) * len(CELLS)
    for model in MODELS:
        cdf_ca, cdf_te, timing = prepared_cdfs("s1", model, tr, ca, te)
        scores = cumulative_score(cdf_ca, y_ca)

        q_cell = mondrian_calibrate(scores, cell_ca, ALPHA)
        thresholds = np.array([q_cell[cell] for cell in cell_te])
        lo_cell, hi_cell = interval_sets(cdf_te, thresholds)

        q_global = split_calibrate(scores, ALPHA)
        lo_global, hi_global = interval_sets(cdf_te, q_global)

        for method, lo, hi in (
            ("final_cell", lo_cell, hi_cell),
            ("marginal_diagnostic", lo_global, hi_global),
        ):
            for cell in CELLS:
                mask = cell_te == cell
                rows.append({
                    "model": model,
                    "method": method,
                    "final_cell": cell,
                    "alpha": ALPHA,
                    "family_error": FAMILY_ERROR,
                    "family_comparisons": comparisons,
                    "interval_error": FAMILY_ERROR / comparisons,
                    "fit_s": timing["fit_s"],
                    "predict_s": timing["predict_s"],
                    "cache_used": timing["cached"],
                    "seed": SEED,
                    **summarize(y_te, lo, hi, mask, comparisons),
                })

    del df, tr, ca, te
    gc.collect()
    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "e2_main_cells.parquet", index=False)
    audit = {
        "split": split_audit,
        "cell_rule": {
            "priority_high_to_low": ["motorcycle", "unrestrained",
                                      "rural_highspeed", "baseline"],
            "motorcycle": "Prsn_Type_ID == 'Driver Of Motorcycle Type Vehicle'",
            "unrestrained": "Prsn_Rest_ID == 'None' among remaining records",
            "rural_highspeed": "Rural_Fl == 'Y' and speed_limit >= 55 among remaining records",
            "baseline": "all remaining records",
            "training_rollup_threshold": 1000,
            "raw_to_final_mapping": {cell: cell for cell in CELLS},
        },
        "cell_counts": cell_counts,
        "simultaneous_policy": {
            "family": "three main models by four final cells",
            "comparisons": comparisons,
            "family_error": FAMILY_ERROR,
            "method": "two-sided Clopper-Pearson with Bonferroni correction",
        },
    }
    (RESULTS / "e2_main_cells_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True))
    print(out.to_string(index=False))
    print(f"E2 main-cell audit done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
