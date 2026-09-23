#!/usr/bin/env python
"""E8: the composition theorem in action on S1.

This is a semi-synthetic outcome experiment. The observed CRIS KABCO label is the
pseudo-true outcome Y. Synthetic reported labels Y_tilde are generated from Y only
on the calibration fold. The test fold is evaluated against its observed KABCO label
as pseudo-true Y. This does not validate unobserved medical severity in Texas.

The audit reports coverage only for the final calibration cell after rollup. It also
writes the leaf-to-final-cell ledger, the full partition definition, and the injected
outcome process. A leaf label is descriptive after rollup and is not a certified
conditioning event.
"""
import json
import time

import numpy as np
import pandas as pd

from common import (SEED, RESULTS, load_primary, split_s1, s1_split_audit,
                    prepared_cdfs, coverage_stats)
from choir import CertifiedOrdinal, NoiseModel
from e2_e3_heterogeneity import declared_partition
from e4_noise import inject_noise, beyond_band_mass

ALPHA = 0.10
DELTA = 0.02
MODEL = "histgb"
N_MIN = 1000


def _noise_map(nm: NoiseModel) -> dict:
    if nm.tmap:
        return {str(k): {"down": int(v[0]), "up": int(v[1])}
                for k, v in nm.tmap.items()}
    return {"constant_band": {"down": int(nm.b_plus), "up": int(nm.b_minus)}}


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)

    rng = np.random.default_rng(SEED + 8)
    # Outcome construction is explicit. y_kabco is not a medical reference label.
    # It is the pseudo-true Y used to make a controlled label-noise experiment.
    yt_ca, eps = inject_noise(y_ca, DELTA, rng, exact_fatal=True)

    part_ca, part_te = declared_partition(ca), declared_partition(te)
    strat_ca = np.where(ca["Rural_Fl"].to_numpy() == "Y", "rural", "urban")
    strat_te = np.where(te["Rural_Fl"].to_numpy() == "Y", "rural", "urban")

    # base model as a callable backed by the cached CDFs (keyed by matrix identity)
    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]

    # partition callable keyed by identity of the input matrix (train-split discipline:
    # declared_partition is a fixed covariate rule, no fitting involved)
    holder = {"labels": part_ca}
    cert = CertifiedOrdinal(
        base=base, K=5,
        partition=lambda X: holder["labels"],
        noise=NoiseModel(K=5, b_plus=1, b_minus=1, delta=DELTA),
        n_min=N_MIN,
    )
    cert.calibrate(X_ca, yt_ca, strata=strat_ca)
    holder["labels"] = part_te
    print(f"stack calibrated ({time.time()-t0:.0f}s); eps_tot={eps:.3f}")

    # Two declared compatibility maps on the SAME calibration (noise enters only at
    # expansion): the constant band b=1, and the category-dependent KABCO map with
    # exact fatal. The constant band is uninformative on the hard strata (it certifies
    # the whole scale); the category-dependent map is the efficient variant (Remark 3.2).
    # "none" is a diagnostic arm, not a certified one: it reports the informativeness
    # of the sets BEFORE expansion, which localizes whether vacuity is caused by the
    # compatibility map or was already present in the calibrated base sets.
    bands = {
        "constant_b1": NoiseModel(K=5, b_plus=1, b_minus=1, delta=DELTA),
        "kabco_catdep": NoiseModel.kabco(delta=DELTA, exact_fatal=True,
                                         a_reaches_down=2),
        "none": None,
    }

    # The calibration partitions are fixed before score calibration. The sidecar
    # records both each original leaf and the final cell that owns its threshold.
    partition_audit = {
        "split": s1_split_audit(df),
        "partition_definition": {
            "product": "declared safety class x Rural_Fl rural/urban",
            "class_priority_high_to_low": ["motorcycle", "unrestrained", "rural_highspeed", "baseline"],
            "motorcycle": "Prsn_Type_ID == 'Driver Of Motorcycle Type Vehicle'",
            "unrestrained": "Prsn_Rest_ID == 'None'",
            "rural_highspeed": "Rural_Fl == 'Y' and speed_limit >= 55",
            "baseline": "all remaining rows",
            "stratum": "rural if Rural_Fl == 'Y', otherwise urban",
            "n_min": N_MIN,
            "rollup_order": ["product leaf", "declared safety class", "global"],
        },
        "outcome_construction": {
            "regime": "semi_synthetic",
            "pseudo_true_outcome": "observed CRIS y_kabco on calibration and test folds",
            "synthetic_reported_outcome": "inject_noise(pseudo_true_outcome) on calibration fold only",
            "test_evaluation_outcome": "observed CRIS y_kabco as pseudo-true outcome",
            "noise_seed": SEED + 8,
            "exact_fatal_in_injection": True,
            "calibration_error_rate": float(eps),
            "fatal_labels_changed_in_calibration": int(np.sum((y_ca == 5) & (yt_ca != 5))),
            "declared_delta": DELTA,
        },
        "leaf_to_final_calibration_cell": cert.partition_audit(alpha=ALPHA),
        "noise_compatibility_maps": {},
    }
    for band, nm in bands.items():
        if nm is not None:
            partition_audit["noise_compatibility_maps"][band] = {
                "map": _noise_map(nm),
                "declared_delta": nm.delta,
                "realized_beyond_compatibility_mass_calibration":
                    beyond_band_mass(y_ca, yt_ca, nm),
            }
    (RESULTS / "e8_partition_outcome_audit.json").write_text(
        json.dumps(partition_audit, indent=2, sort_keys=True))

    def informativeness(lo_m, hi_m):
        """Mean width hides the problem: report how often the set is the FULL scale."""
        w = hi_m - lo_m + 1
        return {"frac_full_scale": float((w == 5).mean()),
                "frac_width_le3": float((w <= 3).mean()),
                "mean_width": float(w.mean())}

    rows = []
    for band, nm in bands.items():
        cert.noise = nm
        # the "none" arm carries no expansion, so the Thm-6 floor does not apply to it
        floor = np.nan if nm is None else 1 - ALPHA - DELTA
        lo, hi = cert.predict_set(X_te, alpha=ALPHA, strata=strat_te)
        resolution = cert.resolved_cells(X_te, alpha=ALPHA, strata=strat_te)

        def emit(mask, final_cell, rollup_level, final_n_cal):
            st = coverage_stats(y_te[mask], lo[mask], hi[mask])
            holds = (None if nm is None
                     else bool(st["coverage"] >= floor - 3 * st["se"]))
            rows.append({"band": band, "final_calibration_cell": final_cell,
                         "rollup_level": rollup_level, "final_n_cal": final_n_cal,
                         "coverage_conditioning": "final_calibration_cell",
                         "outcome_regime": "semi_synthetic_pseudo_true_kabco",
                         **st,
                         **informativeness(lo[mask], hi[mask]), "floor": floor,
                         "holds": holds})

        for final_cell in np.unique(resolution["final_cell"]):
            m = resolution["final_cell"] == final_cell
            i = np.flatnonzero(m)[0]
            emit(m, str(final_cell), str(resolution["rollup_level"][i]),
                 int(resolution["final_n_cal"][i]))
        emit(np.ones(len(y_te), bool), "overall:ALL", "overall", len(y_ca))
        print(f"[{band}] predicted ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out["alpha"], out["delta"], out["model"], out["seed"] = ALPHA, DELTA, MODEL, SEED
    out.to_parquet(RESULTS / "e8_composition.parquet", index=False)
    print(out[["band", "final_calibration_cell", "rollup_level", "n", "coverage", "floor",
               "avg_width", "frac_full_scale", "frac_width_le3", "holds"]].to_string(index=False))
    certified = out[out.band != "none"]
    print(f"E8 done in {time.time()-t0:.0f}s; all certified cells hold: "
          f"{bool(certified['holds'].all())}")


if __name__ == "__main__":
    main()
