#!/usr/bin/env python
"""E4 semi-synthetic audit of the reporting-aware KABCO expansion.

The observed CRIS category is the pseudo-true outcome Y. A fully specified transition
rule generates Y_tilde on both calibration and test folds. Calibration uses Y_tilde,
while the expanded interval is evaluated against the pre-injection category Y.
"""
import time
import gc
import json

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s1, prepared_cdfs,
                    coverage_stats)
from choir import cumulative_score, interval_sets, mondrian_calibrate
from choir.noise import NoiseModel
from e2_e3_heterogeneity import declared_partition

ALPHA = 0.10
MODEL = "histgb"
DELTAS = [0.0, 0.01, 0.02, 0.05]
K = 5
P_UNDER_ADJ = 0.20
P_OVER_ADJ = 0.10
NOISE_SEED = SEED + 4


def inject_noise(y, delta, rng, exact_fatal=False):
    """Banded misclassification: adjacent confusion (within band) + beyond-band mass
    delta (jump of 2 downward in report = under-reporting by 2). Fatal exact if set.

    NOTE (declared-gate arm): `delta` here is the CONDITIONAL gate probability applied
    only to rows eligible to jump two categories down (y in {3, 4} when exact_fatal).
    The REALIZED beyond-band mass P(Y not in T(Ytilde)) is therefore delta * P(y in
    {3,4}) ~= delta * 0.078, not delta. Assumption N is an inequality, so the floor
    1-alpha-delta stays valid, but this arm cannot stress it. See
    inject_noise_realized() for the arm where realized mass == declared delta.
    """
    y = y.copy()
    yt = y.copy()
    movable = np.ones(len(y), bool)
    if exact_fatal:
        movable &= y != 5
    u = rng.uniform(size=len(y))
    down = movable & (u < P_UNDER_ADJ) & (y > 1)               # report understates by 1
    yt[down] = y[down] - 1
    up = movable & (u >= P_UNDER_ADJ) & (u < P_UNDER_ADJ + P_OVER_ADJ) & (y < (4 if exact_fatal else 5))
    yt[up] = y[up] + 1                                         # report overstates by 1
    v = rng.uniform(size=len(y))
    far = movable & (v < delta) & (y > 2)                      # beyond band: understate by 2
    yt[far] = y[far] - 2
    eps_tot = float((yt != y).mean())
    return yt, eps_tot


def inject_noise_realized(y, delta, rng, exact_fatal=False, cells=None):
    """Apply one recordwise reporting kernel to any fold.

    Every B, A, or K record independently enters the far-under-reporting branch with
    probability delta. Therefore, the cellwise population beyond-map probability is
    delta times the severe-category share and is no larger than delta.
    """
    y = y.copy()
    yt = y.copy()
    movable = np.ones(len(y), bool)
    if exact_fatal:
        movable &= y != 5
    eligible = movable & (y > 2)
    cells = np.asarray(cells if cells is not None else np.repeat("overall", len(y)))
    far = eligible & (rng.uniform(size=len(y)) < delta)
    u = rng.uniform(size=len(y))
    remaining = ~far
    down = remaining & movable & (u < P_UNDER_ADJ) & (y > 1)
    yt[down] = y[down] - 1
    up = (remaining & movable & (u >= P_UNDER_ADJ) &
          (u < P_UNDER_ADJ + P_OVER_ADJ) &
          (y < (4 if exact_fatal else 5)))
    yt[up] = y[up] + 1
    yt[far] = y[far] - 2
    cell_info = {}
    for cell in np.unique(cells):
        in_cell = cells == cell
        n_cell = int(in_cell.sum())
        n_eligible = int((eligible & in_cell).sum())
        n_far = int((far & in_cell).sum())
        cell_info[str(cell)] = {
            "n": n_cell,
            "n_eligible": n_eligible,
            "far_gate_probability": float(delta),
            "n_beyond_realized": n_far,
            "realized_beyond_mass": float(n_far / n_cell),
            "population_upper_bound": float(delta),
        }
    eps_tot = float((yt != y).mean())
    return yt, eps_tot, cell_info


def beyond_band_mass(y, yt, nm):
    """Realized P(Y not in T(Ytilde)) under the declared compatibility map of `nm`.

    T(k) = [k - down_k, k + up_k] (clipped to the scale): the truths a report k is
    declared compatible with. This is the quantity Assumption N bounds by delta.
    """
    y = np.asarray(y)
    yt = np.asarray(yt)
    if nm.tmap:
        down = np.array([nm.tmap[k][0] for k in range(1, nm.K + 1)])
        up = np.array([nm.tmap[k][1] for k in range(1, nm.K + 1)])
        loT = np.maximum(yt - down[yt - 1], 1)
        hiT = np.minimum(yt + up[yt - 1], nm.K)
    else:
        loT = np.maximum(yt - nm.b_plus, 1)
        hiT = np.minimum(yt + nm.b_minus, nm.K)
    return float(((y < loT) | (y > hiT)).mean())


def transition_counts(y, yt):
    """Return the complete 5 by 5 pseudo-true-to-generated-report count matrix."""
    out = np.zeros((K, K), dtype=np.int64)
    np.add.at(out, (np.asarray(y) - 1, np.asarray(yt) - 1), 1)
    return out.tolist()


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cells_ca, cells_te = declared_partition(ca), declared_partition(te)
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)
    del df, tr, ca, te
    gc.collect()
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    rows = []
    audit = {
        "regime": "semi_synthetic",
        "pseudo_true_outcome": "observed CRIS y_kabco",
        "generated_reported_outcome": "synthetic y_tilde",
        "compatibility_map_version": "kabco-main-v1",
        "compatibility_map": {str(k): list(v) for k, v in
                              NoiseModel.kabco(exact_fatal=False,
                                               a_reaches_down=2).tmap.items()},
        "alpha": ALPHA,
        "noise_seed": NOISE_SEED,
        "cell_rule": "declared covariate-only safety cells",
        "transition_algorithm": {
            "far_branch": "each B, A, or K record independently reports two categories lower with probability delta",
            "adjacent_under_branch": "among remaining records above O, report one category lower when U < 0.20",
            "adjacent_over_branch": "among remaining records below K, report one category higher when 0.20 <= U < 0.30",
            "unchanged_branch": "all other records",
            "branch_priority": ["far", "adjacent_under", "adjacent_over", "unchanged"],
            "population_cell_bound": "P(outside map | cell) = delta * P(Y in {B,A,K} | cell) <= delta",
        },
        "deltas": {},
    }
    for j, delta in enumerate(DELTAS):
        rng_ca = np.random.default_rng(NOISE_SEED + 2 * j)
        rng_te = np.random.default_rng(NOISE_SEED + 2 * j + 1)
        yt_ca, eps_ca, info_ca = inject_noise_realized(
            y_ca, delta, rng_ca, cells=cells_ca)
        yt_te, eps_te, info_te = inject_noise_realized(
            y_te, delta, rng_te, cells=cells_te)
        audit["deltas"][str(delta)] = {
            "calibration": {
                "cell_injection": info_ca,
                "transition_counts": transition_counts(y_ca, yt_ca),
            },
            "test": {
                "cell_injection": info_te,
                "transition_counts": transition_counts(y_te, yt_te),
            },
        }
        s_ca = cumulative_score(cdf_ca, yt_ca)
        qhat = mondrian_calibrate(s_ca, cells_ca, ALPHA)
        thresholds = np.array([qhat[cell] for cell in cells_te])
        lo, hi = interval_sets(cdf_te, thresholds)
        lo, hi = lo.astype(np.int8, copy=False), hi.astype(np.int8, copy=False)
        reported_stats = coverage_stats(yt_te, lo, hi)

        nm_main = NoiseModel.kabco(delta=delta, exact_fatal=False,
                                   a_reaches_down=2)
        nm_const = NoiseModel(K=K, b_plus=1, b_minus=1, delta=delta)
        for band, nm in (("kabco_main", nm_main), ("constant_b1", nm_const)):
            lo_e, hi_e = nm.expand(lo, hi)
            for cell in [*np.unique(cells_te), "overall"]:
                mask_te = np.ones(len(y_te), dtype=bool) if cell == "overall" else cells_te == cell
                mask_ca = np.ones(len(y_ca), dtype=bool) if cell == "overall" else cells_ca == cell
                st = coverage_stats(y_te[mask_te], lo_e[mask_te], hi_e[mask_te])
                rep = coverage_stats(yt_te[mask_te], lo[mask_te], hi[mask_te])
                rows.append({
                    "arm": "realized_delta",
                    "band": band,
                    "final_cell": cell,
                    "compatibility_map_version": "kabco-main-v1" if band == "kabco_main" else "constant-b1-v1",
                    "delta": delta,
                    "eps_tot_calibration": eps_ca,
                    "eps_tot_test": eps_te,
                    "beyond_map_calibration": beyond_band_mass(
                        y_ca[mask_ca], yt_ca[mask_ca], nm),
                    "beyond_map_test": beyond_band_mass(
                        y_te[mask_te], yt_te[mask_te], nm),
                    "reported_coverage": rep["coverage"],
                    "reported_avg_width": rep["avg_width"],
                    "floor": 1 - ALPHA - delta,
                    **st,
                })

        st = coverage_stats(y_te, lo, hi)
        rows.append({
            "arm": "realized_delta", "band": "none", "final_cell": "overall",
            "compatibility_map_version": "none", "delta": delta,
            "eps_tot_calibration": eps_ca, "eps_tot_test": eps_te,
            "beyond_map_calibration": beyond_band_mass(y_ca, yt_ca, nm_main),
            "beyond_map_test": beyond_band_mass(y_te, yt_te, nm_main),
            "reported_coverage": reported_stats["coverage"],
            "reported_avg_width": reported_stats["avg_width"], "floor": np.nan, **st,
        })
        print(f"delta={delta}: calibration beyond-map mass="
              f"{beyond_band_mass(y_ca, yt_ca, nm_main):.5f}; "
              f"test beyond-map mass={beyond_band_mass(y_te, yt_te, nm_main):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out["alpha"] = ALPHA
    out["model"] = MODEL
    out["seed"] = NOISE_SEED
    out.to_parquet(RESULTS / "e4_noise.parquet", index=False)
    (RESULTS / "e4_outcome_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True))
    print(out[["band", "final_cell", "delta", "beyond_map_calibration", "beyond_map_test",
               "reported_coverage", "floor", "coverage", "avg_width",
               ]].to_string(index=False))
    ok = out[(out.band == "kabco_main") & (out.final_cell != "overall")]
    accept = bool((ok["coverage"] >= ok["floor"] - 3 * ok["se"]).all())
    print(f"E4 main-map bound holds at every delta: {accept}")
    print(f"E4 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
