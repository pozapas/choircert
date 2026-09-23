#!/usr/bin/env python
"""Round-3 Task 2 supplement: exact frac_full_scale for the composed coverage set
at alpha=0.10, kabco_catdep, per declared stratum -- to directly compare against
the beta=0.10 risk-set frac_full_scale already computed in
adhoc_r3_task2_beta_frontier.parquet. Standalone, writes nothing that overwrites
existing files.
"""
import numpy as np

from common import SEED, load_primary, split_s1, prepared_cdfs
from choir import CertifiedOrdinal, NoiseModel
from e2_e3_heterogeneity import declared_partition
from e4_noise import inject_noise

DELTA = 0.02
MODEL = "histgb"
N_MIN = 1000
STRATA = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]


def main():
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)
    rng = np.random.default_rng(SEED + 8)
    yt_ca, _ = inject_noise(y_ca, DELTA, rng)
    part_ca, part_te = declared_partition(ca), declared_partition(te)

    X_ca, X_te = np.empty((len(ca), 0)), np.empty((len(te), 0))
    cdf_by_id = {id(X_ca): cdf_ca, id(X_te): cdf_te}
    base = lambda X: cdf_by_id[id(X)]
    holder = {"labels": part_ca}
    partition_fn = lambda X: holder["labels"]
    nm = NoiseModel.kabco(delta=DELTA, exact_fatal=True, a_reaches_down=2)
    cert = CertifiedOrdinal(base=base, K=5, partition=partition_fn, noise=nm, n_min=N_MIN)
    cert.calibrate(X_ca, yt_ca)
    holder["labels"] = part_te
    lo, hi = cert.predict_set(X_te, alpha=0.10)
    w = hi - lo + 1
    print("alpha=0.10 kabco_catdep composed coverage-set frac_full_scale (width==5), per stratum:")
    for s in STRATA:
        m = part_te == s
        print(f"  {s}: frac_full_scale={float((w[m]==5).mean()):.6f} "
              f"mean_width={float(w[m].mean()):.6f} n={int(m.sum())}")


if __name__ == "__main__":
    main()
