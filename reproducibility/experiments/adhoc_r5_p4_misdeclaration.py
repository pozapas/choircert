"""Round 5, Task 2: the P4 mis-declaration falsification, run at alpha in {0.10, 0.20, 0.30}.

Supervisor's point (SUPERVISOR_RESPONSE_2 sec 6): the mis-declaration test cannot bite
where the composed set is the full scale, because a true label can never EXIT a full-scale
set, so coverage is pinned at 1.000 regardless of how badly the band is mis-declared. At
alpha = 0.10 the hard-stratum sets are full-scale, so the test is toothless there. Raising
alpha pulls the sets off the ceiling, at which point a mis-declared band CAN push true-label
coverage through the declared floor. This script demonstrates exactly that, with no new data.

Design. The calibration labels carry an injected beyond-band process at the largest
realized mass the two-down design admits (`inject_noise_realized` with a high request; the
realized value is recorded, not assumed). The certificate DECLARES the band at
delta_declared = 0.02, so the promised floor is 1 - alpha - 0.02. The realized beyond-band
mass is larger, so the declaration is wrong on purpose. We then measure true-label coverage
of the composed set per stratum at each alpha and compare it to the DECLARED floor. A drop
below the declared floor is the guarantee failing under mis-declaration -- the thing the
original always-passing experiment could never show.

This is the stage-one demonstration only. The full three-arm design (benign / adversarially
coupled / published-rate) on informative strata is a separate, larger task.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from common import (SEED, RESULTS, load_primary, split_s1, prepared_cdfs,  # noqa: E402
                    coverage_stats)
from choir import CertifiedOrdinal, NoiseModel  # noqa: E402
from e2_e3_heterogeneity import declared_partition  # noqa: E402
from e4_noise import inject_noise_realized  # noqa: E402

MODEL = "histgb"
DELTA_DECLARED = 0.02          # what the certificate claims (band +/-1, floor 1-a-0.02)
DELTA_REQUEST = 0.30           # requested realized beyond-band mass (clips at feasibility)
ALPHAS = [0.10, 0.20, 0.30]
N_MIN = 1000


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)

    # MIS-DECLARATION: inject a large realized beyond-band mass into the calibration labels
    rng = np.random.default_rng(SEED + 54)
    yt_ca, eps_realized, info = inject_noise_realized(y_ca, DELTA_REQUEST, rng)
    print(f"injection: requested delta={DELTA_REQUEST}, realized eps_tot={eps_realized:.4f}, "
          f"p_eligible={info['p_eligible']:.4f}, feasible={info['delta_feasible']}, "
          f"delta_max={info['delta_max']:.4f}")

    part_ca, part_te = declared_partition(ca), declared_partition(te)
    strat_ca = np.where(ca["Rural_Fl"].to_numpy() == "Y", "rural", "urban")
    strat_te = np.where(te["Rural_Fl"].to_numpy() == "Y", "rural", "urban")

    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]

    holder = {"labels": part_ca}
    cert = CertifiedOrdinal(
        base=base, K=5,
        partition=lambda X: holder["labels"],
        noise=NoiseModel(K=5, b_plus=1, b_minus=1, delta=DELTA_DECLARED),
        n_min=N_MIN,
    )
    cert.calibrate(X_ca, yt_ca, strata=strat_ca)     # calibrated on the NOISY reported labels
    holder["labels"] = part_te
    print(f"calibrated on mis-declared labels ({time.time()-t0:.0f}s)")

    # realized beyond-band mass per stratum on the TEST split (what the declaration got wrong,
    # cell by cell). T(k) = [k-1, k+1] declared; realized = P(Y_true not in [yt-1, yt+1]).
    # We measure it directly from the injected test labels.
    yt_te, eps_te, _ = inject_noise_realized(y_te, DELTA_REQUEST,
                                             np.random.default_rng(SEED + 55))
    band_lo = np.clip(yt_te - 1, 1, 5)
    band_hi = np.clip(yt_te + 1, 1, 5)
    beyond = (y_te < band_lo) | (y_te > band_hi)

    rows = []
    for alpha in ALPHAS:
        floor = 1 - alpha - DELTA_DECLARED
        lo, hi = cert.predict_set(X_te, alpha=alpha, strata=strat_te)
        width = hi - lo + 1

        def emit(mask, strat):
            st = coverage_stats(y_te[mask], lo[mask], hi[mask])
            rows.append({
                "alpha": alpha, "stratum": strat, "n": int(mask.sum()),
                "declared_floor": floor,
                "coverage_true": st["coverage"], "se": st["se"],
                "breaches_declared_floor": bool(st["coverage"] < floor),
                "frac_full_scale": float((width[mask] == 5).mean()),
                "mean_width": float(width[mask].mean()),
                "realized_beyond_band": float(beyond[mask].mean()),
            })

        for c in ("motorcycle", "unrestrained", "rural_highspeed", "baseline"):
            m = part_te == c
            if m.sum():
                emit(m, c)
        emit(np.ones(len(y_te), bool), "ALL")
        print(f"[alpha={alpha}] floor={floor:.2f} predicted ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out["delta_declared"], out["eps_realized"], out["model"] = DELTA_DECLARED, eps_realized, MODEL
    out.to_parquet(RESULTS / "adhoc_r5_p4_misdeclaration.parquet", index=False)
    cols = ["alpha", "stratum", "n", "declared_floor", "coverage_true",
            "breaches_declared_floor", "frac_full_scale", "mean_width", "realized_beyond_band"]
    print("\n" + out[cols].to_string(index=False))
    print(f"\nDone in {time.time()-t0:.0f}s. "
          f"declared delta={DELTA_DECLARED}, realized eps_tot={eps_realized:.4f}.")
    print("Read: where frac_full_scale ~ 1.0, coverage cannot breach (sets absorb the "
          "violation); where sets come off the ceiling, breaches become possible.")


if __name__ == "__main__":
    main()
