#!/usr/bin/env python
"""Round-3 Task 3 (scoping, not part of the pipeline): per-cell delta sensitivity.

The composition subtracts delta CELL-WISE (floor 1-alpha-delta_c per cell), but the
KABCO-MAIS linkage studies report POPULATION-level agreement. This checks whether a
single population delta understates the per-cell delta on the strata that matter.

Two deliverables, kept separate because they answer different questions:

(A) ROBUST: realized beyond-band mass per declared cell under the PAPER'S ACTUAL
    injection exactly as e8_composition.py runs it (inject_noise, single global
    delta=0.02 applied uniformly, NOT swept). This is well-defined regardless of any
    coupling assumption and directly answers "does the population-delta assumption
    hold up cell-by-cell under what the paper already does."

(B) THE SWEEP: delta_c in {0.02, 0.05, 0.10, 0.20} for one cell at a time (other
    cells held at 0.02), using a per-cell renormalized injection (mirrors
    inject_noise_realized but with a per-row, cell-dependent gate probability) so the
    REALIZED beyond-band mass in the swept cell actually tracks delta_c (declared-gate
    delta does NOT: realized ~= delta * P(eligible) ~= delta * 0.078, not delta).
    IMPORTANT MECHANIC: NoiseModel.expand() does not depend on delta at all (only the
    tmap/band shape does) -- delta only ever enters (i) the injection gate and (ii)
    the coverage floor 1-alpha-delta_c used for the audit. Widening delta_c does NOT
    widen the certified set; it only changes what floor that cell's coverage is
    audited against. Per-cell eligibility P(eligible) = P(y>2, y!=5) differs by cell,
    so a requested delta_c can be infeasible (exceeds that cell's cap) -- reported.

This is a NEW, standalone diagnostic script; it does not modify or overwrite any
existing experiment file or its results parquet.
"""
import time

import numpy as np
import pandas as pd

from common import SEED, RESULTS, load_primary, split_s1, prepared_cdfs, coverage_stats
from choir import CertifiedOrdinal, NoiseModel
from e2_e3_heterogeneity import declared_partition
from e4_noise import inject_noise, beyond_band_mass

ALPHA = 0.10
BASE_DELTA = 0.02
SWEEP_DELTAS = [0.02, 0.05, 0.10, 0.20]
MODEL = "histgb"
N_MIN = 1000
STRATA = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]
ADJ_UP = 0.20
ADJ_DOWN = 0.10


def inject_noise_percell(y, cell, delta_by_cell, rng, exact_fatal=True):
    """Per-cell renormalized injection: gate probability computed PER CELL so that
    realized beyond-band mass in cell c tracks delta_by_cell[c] (up to that cell's
    feasibility cap). Adjacent within-band noise (ADJ_UP/ADJ_DOWN) is unswept and
    applied uniformly, matching e4_noise.inject_noise_realized's structure.
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

    eligible = movable & (y > 2)
    v = rng.uniform(size=len(y))
    far = np.zeros(len(y), bool)
    cell_info = {}
    for c in np.unique(cell):
        mc = cell == c
        elig_c = eligible & mc
        p_elig = float(elig_c.mean()) if mc.sum() > 0 else 0.0
        p_elig_cond = float(elig_c.sum() / mc.sum()) if mc.sum() > 0 else 0.0
        d_c = delta_by_cell[c]
        p_req = d_c / p_elig_cond if p_elig_cond > 0 else np.inf
        feasible = bool(p_req <= 1.0)
        p_gate = float(min(p_req, 1.0))
        far |= elig_c & (v < p_gate)
        cell_info[c] = {"p_eligible_cond": p_elig_cond, "p_gate": p_gate,
                        "delta_feasible": feasible, "delta_requested": d_c,
                        "delta_max": p_elig_cond}
    yt[far] = y[far] - 2
    return yt, cell_info


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)
    part_ca, part_te = declared_partition(ca), declared_partition(te)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]
    holder = {"labels": part_ca}
    partition_fn = lambda X: holder["labels"]
    nm = NoiseModel.kabco(delta=BASE_DELTA, exact_fatal=True, a_reaches_down=2)

    # ---------------- (A) robust: paper's actual injection, per-cell realized mass ----------------
    rng_a = np.random.default_rng(SEED + 8)  # identical seed/stream to e8_composition.py
    yt_ca_paper, eps_paper = inject_noise(y_ca, BASE_DELTA, rng_a)
    a_rows = []
    for c in STRATA:
        m = part_ca == c
        bbm = beyond_band_mass(y_ca[m], yt_ca_paper[m], nm)
        a_rows.append({"cell": c, "n_cal": int(m.sum()),
                        "realized_beyond_band_mass": bbm,
                        "declared_population_delta": BASE_DELTA,
                        "ratio_realized_over_declared": bbm / BASE_DELTA if BASE_DELTA else np.nan})
    a_global = beyond_band_mass(y_ca, yt_ca_paper, nm)
    a_df = pd.DataFrame(a_rows)
    a_df.to_parquet(RESULTS / "adhoc_r3_task3a_realized_mass_paper_injection.parquet", index=False)
    print("\n--- (A) realized beyond-band mass per cell, PAPER'S actual injection "
          f"(global delta={BASE_DELTA}) ---")
    print(a_df.to_string(index=False))
    print(f"global (all cells) realized beyond-band mass: {a_global:.6f} "
          f"(declared population delta: {BASE_DELTA})")

    # ---------------- (B) the sweep ----------------
    b_rows = []
    feas_rows = []
    for target_cell in STRATA:
        for d_c in SWEEP_DELTAS:
            delta_by_cell = {c: (d_c if c == target_cell else BASE_DELTA) for c in STRATA}
            rng_b = np.random.default_rng(SEED + 8)  # fresh, same base stream each sweep point
            yt_ca, cell_info = inject_noise_percell(y_ca, part_ca, delta_by_cell, rng_b)

            cert = CertifiedOrdinal(base=base, K=5, partition=partition_fn, noise=nm, n_min=N_MIN)
            holder["labels"] = part_ca
            cert.calibrate(X_ca, yt_ca)
            holder["labels"] = part_te
            lo, hi = cert.predict_set(X_te, alpha=ALPHA)

            m = part_te == target_cell
            st = coverage_stats(y_te[m], lo[m], hi[m])
            floor_c = 1 - ALPHA - d_c
            breach = st["coverage"] < floor_c
            info = cell_info[target_cell]
            realized_bbm = beyond_band_mass(y_ca[part_ca == target_cell],
                                             yt_ca[part_ca == target_cell], nm)
            b_rows.append({
                "target_cell": target_cell, "delta_c_requested": d_c,
                "floor_1_minus_alpha_minus_delta_c": floor_c,
                "realized_coverage": st["coverage"], "se": st["se"],
                "breaches_floor": bool(breach),
                "breach_margin": (floor_c - st["coverage"]) if breach else 0.0,
                "avg_width": st["avg_width"], "n_test": st["n"],
                "delta_feasible": info["delta_feasible"],
                "p_eligible_cond": info["p_eligible_cond"],
                "delta_max_this_cell": info["delta_max"],
                "p_gate_applied": info["p_gate"],
                "realized_beyond_band_mass": realized_bbm,
            })
            feas_rows.append({"cell": target_cell, "p_eligible_cond": info["p_eligible_cond"]})
        print(f"[{target_cell}] sweep done ({time.time()-t0:.0f}s)", flush=True)

    b_df = pd.DataFrame(b_rows)
    b_df.to_parquet(RESULTS / "adhoc_r3_task3b_percell_delta_sweep.parquet", index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    print("\n--- (B) per-cell delta_c sweep (others held at 0.02) ---")
    print(b_df[["target_cell", "delta_c_requested", "floor_1_minus_alpha_minus_delta_c",
                "realized_coverage", "breaches_floor", "breach_margin",
                "delta_feasible", "p_eligible_cond", "delta_max_this_cell",
                "realized_beyond_band_mass"]].to_string(index=False))

    print("\n--- per-cell eligibility ceiling (P(y in {3,4}), the max REALIZED beyond-band "
          "mass this two-down injection design can ever deliver in that cell) ---")
    for c in STRATA:
        p = [r["p_eligible_cond"] for r in feas_rows if r["cell"] == c][0]
        print(f"  {c}: delta_max = {p:.6f}")

    breaching = b_df[b_df.breaches_floor]
    print(f"\n{len(breaching)} of {len(b_df)} (cell, delta_c) combinations breach their floor.")
    if len(breaching):
        print(breaching[["target_cell", "delta_c_requested", "breach_margin"]].to_string(index=False))

    print(f"\nTask3 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
