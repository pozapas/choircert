#!/usr/bin/env python
"""Descriptive final-cell temporal stress test.

Histogram gradient boosting is trained on 2017 through 2021 records. Final-cell
thresholds are calibrated on 2022 through 2023 records and evaluated separately in
2024 and 2025. The test is descriptive because the random-split exchangeability
guarantee does not extend across calendar-time shift without an added assumption.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import beta

from common import RESULTS, SEED, load_primary, prepared_cdfs, split_s2
from e2_e3_heterogeneity import declared_partition
from choir import cumulative_score, interval_sets, mondrian_calibrate

ALPHA = 0.10
FAMILY_ERROR = 0.05
MODEL = "histgb"
YEARS = (2024, 2025)
CELLS = ("baseline", "motorcycle", "rural_highspeed", "unrestrained")


def simultaneous_interval(successes: int, n: int, comparisons: int):
    tail = FAMILY_ERROR / comparisons / 2
    lo = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - tail, successes + 1, n - successes))
    return lo, hi


def main():
    df = load_primary()
    tr, ca, te = split_s2(df)
    y_ca = ca["y_kabco"].to_numpy(dtype=np.int8)
    y_te = te["y_kabco"].to_numpy(dtype=np.int8)
    cell_ca = declared_partition(ca)
    cell_te = declared_partition(te)
    years = te["crash_year"].to_numpy(dtype=np.int16)

    cdf_ca, cdf_te, timing = prepared_cdfs("s2", MODEL, tr, ca, te)
    scores = cumulative_score(cdf_ca, y_ca)
    q_cell = mondrian_calibrate(scores, cell_ca, ALPHA)
    thresholds = np.array([q_cell[cell] for cell in cell_te])
    lo, hi = interval_sets(cdf_te, thresholds)

    rows = []
    comparisons = len(YEARS) * len(CELLS)
    for year in YEARS:
        for cell in CELLS:
            mask = (years == year) & (cell_te == cell)
            covered = (y_te[mask] >= lo[mask]) & (y_te[mask] <= hi[mask])
            successes = int(covered.sum())
            n = int(mask.sum())
            ci_lo, ci_hi = simultaneous_interval(successes, n, comparisons)
            rows.append({
                "year": year,
                "final_cell": cell,
                "n": n,
                "covered": successes,
                "coverage": float(successes / n),
                "simultaneous_ci_low": ci_lo,
                "simultaneous_ci_high": ci_hi,
                "avg_width": float(np.mean(hi[mask] - lo[mask] + 1)),
                "alpha": ALPHA,
                "model": MODEL,
                "seed": SEED,
                "family_error": FAMILY_ERROR,
                "family_comparisons": comparisons,
                "cache_used": timing["cached"],
            })

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "e6_temporal_cells.parquet", index=False)
    audit = {
        "purpose": "descriptive temporal stress test without an exchangeability claim",
        "training_years": [2017, 2018, 2019, 2020, 2021],
        "calibration_years": [2022, 2023],
        "test_years": list(YEARS),
        "model": MODEL,
        "alpha": ALPHA,
        "seed": SEED,
        "cell_priority": ["motorcycle", "unrestrained", "rural_highspeed", "baseline"],
        "simultaneous_policy": {
            "family": "two target years by four final cells",
            "comparisons": comparisons,
            "family_error": FAMILY_ERROR,
            "method": "two-sided Clopper-Pearson with Bonferroni correction",
        },
    }
    (RESULTS / "e6_temporal_cells_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
