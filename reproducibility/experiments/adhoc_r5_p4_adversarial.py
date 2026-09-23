"""Round 5: the adversarially-coupled arm of the P4 mis-declaration falsification.

Reuses the structure of adhoc_r5_p4_misdeclaration.py and the composition stack from
e8_composition.py: a cached base model, declared_partition (imported, not reimplemented),
CertifiedOrdinal with a DECLARED band NoiseModel(K=5, b_plus=1, b_minus=1, delta=0.02).

MODEL = "ordered_logit" (narrow base sets -- baseline/rural_highspeed are genuinely
informative there, unlike histgb, so this is where an adversarial arm can actually bite).

Three arms, injected into the CALIBRATION labels (the certificate always DECLARES the same
band, delta=0.02; only what actually generated the labels changes):

1. benign        -- e4_noise.inject_noise_realized (two-DOWN jump on severe records,
                     y in {3,4}). This is the mechanic the earlier misdeclaration run showed
                     cannot bite: injectable mass concentrates on strata whose composed sets
                     are already full-scale, so a violation can never exit the set.
2. adversarial   -- inject_adversarial (two-UP jump on LOW-severity records, y in {1,2}),
                     gate renormalized so realized beyond-band mass ~= delta_target=0.10 on
                     the calibration set. This decouples injectable mass from stratum
                     severity: over-reporting lands where truth is mild, which is exactly
                     where the informative (narrow) strata live.
3. published-rate -- inject_published_rate: a documented APPROXIMATION of the Burdett et al.
                     linkage numbers, using the same two-UP mechanic with per-true-category
                     gates instead of one global delta_target:
                       true O(1) -> reported B(3) [+2], gate = 0.004  (Burdett's O
                         beyond-adjacent rate, repurposed to the over-reporting direction)
                       true C(2) -> reported A(4) [+2], gate = 0.382  (the A-to-MAIS1 rate;
                         C->A is the only +2 transition landing on a report of A, so the
                         headline 38.2% number attaches here)
                       true B(3): EXCLUDED. B+2 = K(5) would fabricate a fatality, which
                         violates "leaves fatalities exact" -- the one hard constraint every
                         arm in this file respects. Burdett's B beyond-adjacent rate (2.0%)
                         and C's own beyond-adjacent rate (1.7%) are therefore NOT used: the
                         prompt's spec attaches 38.2% to a "true records" population that,
                         read literally as *true*-A, cannot receive a +2 jump without landing
                         outside the KABCO scale, and attaches it separately to "O/C/B
                         beyond-adjacent at 0.4/1.7/2.0%" -- two numbers for one cell (C) that
                         do not reconcile. Resolved by keeping the larger, headline number
                         (38.2%, explicitly named as "the A-> MAIS1 rate") for C->A, and
                         dropping the smaller, superseded number (1.7%) for the same cell.
                     This is a documented approximation, not a claim to reconstruct Burdett's
                     actual forward generative kernel (which is not identified from the
                     published posterior rates alone).

All three arms are evaluated at alpha in {0.10, 0.20, 0.30}, declared floor 1-alpha-0.02, on
the four declared strata plus ALL. Fatalities are exact in every arm (checked, not assumed).

This is a NEW, standalone diagnostic script; it does not modify experiments/e4_noise.py,
experiments/e8_composition.py, experiments/e2_e3_heterogeneity.py, or
experiments/adhoc_r5_p4_misdeclaration.py.
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
from e4_noise import inject_noise_realized, beyond_band_mass  # noqa: E402

MODEL = "ordered_logit"
DELTA_DECLARED = 0.02
BENIGN_DELTA_REQUEST = 0.30       # matches the earlier misdeclaration run, for comparability
ADVERSARIAL_DELTA_TARGET = 0.10
ALPHAS = [0.10, 0.20, 0.30]
N_MIN = 1000
ADJ_UP = 0.20
ADJ_DOWN = 0.10
STRATA = ["motorcycle", "unrestrained", "rural_highspeed", "baseline"]

# Published-rate arm: per-true-category two-up gate (see module docstring for the reasoning).
PUBLISHED_GATES = {1: 0.004, 2: 0.382}   # true O -> +2 -> B ; true C -> +2 -> A


def inject_adversarial(y, delta_target, rng, exact_fatal=True):
    """Two-UP over-reporting jump on LOW-severity truths (y in {1,2}), gate renormalized
    so realized beyond-band mass ~= delta_target on THIS array. Fatalities exact (the
    eligible set never includes y>=3, so this mechanic alone cannot touch y=5; the adjacent
    ADJ_UP/ADJ_DOWN noise below excludes true fatalities from movement exactly as
    e4_noise.inject_noise does, so exact_fatal is respected end to end).
    """
    y = y.copy()
    yt = y.copy()
    movable = np.ones(len(y), bool)
    if exact_fatal:
        movable &= y != 5
    u = rng.uniform(size=len(y))
    down = movable & (u < ADJ_UP) & (y > 1)
    yt[down] = y[down] - 1
    up = movable & (u >= ADJ_UP) & (u < ADJ_UP + ADJ_DOWN) & (y < (4 if exact_fatal else 5))
    yt[up] = y[up] + 1

    eligible = movable & (y <= 2)
    p_elig = float(eligible.mean())
    p_req = delta_target / p_elig if p_elig > 0 else np.inf
    feasible = bool(p_req <= 1.0)
    p_gate = float(min(p_req, 1.0))
    v = rng.uniform(size=len(y))
    far = eligible & (v < p_gate)
    yt[far] = y[far] + 2
    eps_tot = float((yt != y).mean())
    return yt, eps_tot, {"p_eligible": p_elig, "p_gate": p_gate,
                        "delta_feasible": feasible, "delta_max": p_elig}


def inject_published_rate(y, rng, exact_fatal=True):
    """Documented approximation of the Burdett linkage numbers; see module docstring."""
    y = y.copy()
    yt = y.copy()
    movable = np.ones(len(y), bool)
    if exact_fatal:
        movable &= y != 5
    u = rng.uniform(size=len(y))
    down = movable & (u < ADJ_UP) & (y > 1)
    yt[down] = y[down] - 1
    up = movable & (u >= ADJ_UP) & (u < ADJ_UP + ADJ_DOWN) & (y < (4 if exact_fatal else 5))
    yt[up] = y[up] + 1

    v = rng.uniform(size=len(y))
    far = np.zeros(len(y), bool)
    for true_cat, gate in PUBLISHED_GATES.items():
        m = movable & (y == true_cat) & (v < gate)
        far |= m
    yt[far] = y[far] + 2
    eps_tot = float((yt != y).mean())
    p_eligible = {c: float((movable & (y == c)).mean()) for c in PUBLISHED_GATES}
    return yt, eps_tot, {"gates": PUBLISHED_GATES, "p_eligible_by_cat": p_eligible}


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, timing = prepared_cdfs("s1", MODEL, tr, ca, te)
    cached = bool(timing.get("cached", False))
    print(f"setup {time.time()-t0:.0f}s; {MODEL} cached={cached}", flush=True)
    if not cached:
        print("*** WARNING: model was NOT cache-loaded -- had to fit fresh. ***", flush=True)

    part_ca, part_te = declared_partition(ca), declared_partition(te)
    strat_ca = np.where(ca["Rural_Fl"].to_numpy() == "Y", "rural", "urban")
    strat_te = np.where(te["Rural_Fl"].to_numpy() == "Y", "rural", "urban")

    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]
    holder = {"labels": part_ca}
    nm_declared = NoiseModel(K=5, b_plus=1, b_minus=1, delta=DELTA_DECLARED)

    def run_arm(name, yt_ca, yt_te_for_measurement, info):
        # fatal-exactness check, every arm
        fatal_broken_ca = bool(np.any((y_ca == 5) & (yt_ca != 5)) or
                               np.any((y_ca != 5) & (yt_ca == 5)))
        cert = CertifiedOrdinal(base=base, K=5, partition=lambda X: holder["labels"],
                                noise=nm_declared, n_min=N_MIN)
        holder["labels"] = part_ca
        cert.calibrate(X_ca, yt_ca, strata=strat_ca)
        holder["labels"] = part_te

        realized_bbm_by_stratum = {}
        for s in STRATA + ["ALL"]:
            m = part_te == s if s != "ALL" else np.ones(len(y_te), bool)
            realized_bbm_by_stratum[s] = beyond_band_mass(y_te[m], yt_te_for_measurement[m],
                                                           nm_declared)

        rows = []
        for alpha in ALPHAS:
            floor = 1 - alpha - DELTA_DECLARED
            lo, hi = cert.predict_set(X_te, alpha=alpha, strata=strat_te)
            width = hi - lo + 1
            for s in STRATA + ["ALL"]:
                m = part_te == s if s != "ALL" else np.ones(len(y_te), bool)
                st = coverage_stats(y_te[m], lo[m], hi[m])
                rows.append({
                    "arm": name, "alpha": alpha, "stratum": s, "n": int(m.sum()),
                    "declared_floor": floor, "coverage_true": st["coverage"], "se": st["se"],
                    "breaches_declared_floor": bool(st["coverage"] < floor),
                    "frac_full_scale": float((width[m] == 5).mean()),
                    "mean_width": float(width[m].mean()),
                    "realized_beyond_band_mass": realized_bbm_by_stratum[s],
                })
        print(f"[{name}] fatal_exact_violated={fatal_broken_ca} calibrated+predicted "
              f"({time.time()-t0:.0f}s)", flush=True)
        return rows, fatal_broken_ca

    all_rows = []
    arm_info = {}

    # ---- arm 1: benign (existing inject_noise_realized, two-down, severe records) ----
    rng_ca = np.random.default_rng(SEED + 54)
    yt_ca_benign, eps_ca, info_benign = inject_noise_realized(y_ca, BENIGN_DELTA_REQUEST, rng_ca)
    rng_te = np.random.default_rng(SEED + 55)
    yt_te_benign, eps_te, _ = inject_noise_realized(y_te, BENIGN_DELTA_REQUEST, rng_te)
    print(f"benign injection: requested={BENIGN_DELTA_REQUEST} realized_ca={eps_ca:.6f} "
          f"realized_te={eps_te:.6f} feasible={info_benign['delta_feasible']} "
          f"delta_max={info_benign['delta_max']:.6f}", flush=True)
    rows, fb = run_arm("benign", yt_ca_benign, yt_te_benign, info_benign)
    all_rows += rows
    arm_info["benign"] = {"requested": BENIGN_DELTA_REQUEST, "realized_eps_ca": eps_ca,
                          "realized_eps_te": eps_te, **info_benign, "fatal_exact_violated": fb}

    # ---- arm 2: adversarial (two-up, low-severity, delta_target=0.10) ----
    rng_ca = np.random.default_rng(SEED + 56)
    yt_ca_adv, eps_ca, info_adv = inject_adversarial(y_ca, ADVERSARIAL_DELTA_TARGET, rng_ca)
    rng_te = np.random.default_rng(SEED + 57)
    yt_te_adv, eps_te, _ = inject_adversarial(y_te, ADVERSARIAL_DELTA_TARGET, rng_te)
    print(f"adversarial injection: requested={ADVERSARIAL_DELTA_TARGET} realized_ca={eps_ca:.6f} "
          f"realized_te={eps_te:.6f} feasible={info_adv['delta_feasible']} "
          f"delta_max={info_adv['delta_max']:.6f}", flush=True)
    rows, fb = run_arm("adversarial", yt_ca_adv, yt_te_adv, info_adv)
    all_rows += rows
    arm_info["adversarial"] = {"requested": ADVERSARIAL_DELTA_TARGET, "realized_eps_ca": eps_ca,
                               "realized_eps_te": eps_te, **info_adv, "fatal_exact_violated": fb}

    # ---- arm 3: published-rate (documented approximation) ----
    rng_ca = np.random.default_rng(SEED + 58)
    yt_ca_pub, eps_ca, info_pub = inject_published_rate(y_ca, rng_ca)
    rng_te = np.random.default_rng(SEED + 59)
    yt_te_pub, eps_te, _ = inject_published_rate(y_te, rng_te)
    print(f"published-rate injection: realized_ca={eps_ca:.6f} realized_te={eps_te:.6f} "
          f"gates={info_pub['gates']} p_eligible={info_pub['p_eligible_by_cat']}", flush=True)
    rows, fb = run_arm("published-rate", yt_ca_pub, yt_te_pub, info_pub)
    all_rows += rows
    arm_info["published-rate"] = {"realized_eps_ca": eps_ca, "realized_eps_te": eps_te,
                                  **info_pub, "fatal_exact_violated": fb}

    out = pd.DataFrame(all_rows)
    out["delta_declared"], out["model"] = DELTA_DECLARED, MODEL
    out.to_parquet(RESULTS / "adhoc_r5_p4_adversarial.parquet", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda x: f"{x:.10f}")
    cols = ["arm", "alpha", "stratum", "n", "declared_floor", "coverage_true",
            "breaches_declared_floor", "frac_full_scale", "mean_width",
            "realized_beyond_band_mass"]
    print("\n" + out[cols].to_string(index=False))

    print("\n=== fatal-exactness check per arm (must be all False) ===")
    for arm, info in arm_info.items():
        print(f"  {arm}: fatal_exact_violated={info['fatal_exact_violated']}")

    print(f"\nDone in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
