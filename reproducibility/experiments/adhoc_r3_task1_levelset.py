#!/usr/bin/env python
"""Round-3 Task 1 (scoping, not part of the pipeline): plug-in estimate of the
level-set floor N(x, tau) for the new set-size-lower-bound theorem.

Per (model, declared stratum), on the S1 test split:
  mean_oracle_width  - mean smallest CONTIGUOUS interval whose predicted mass >= tau.
  frac_width_ge_4    - fraction of records where that width >= 4 (headline number).
  frac_width_eq_5    - fraction at full scale.
  mean_N_levelset    - mean |{y : phat(y|x) > t}| at the single t (per model, stratum)
                       making the model's own mean predicted mass at that level 0.90
                       (non-contiguous level-set form, distinct from the interval form).
  n

Also a marginal (covariate-free) reference: phat(y) = TRAIN-split label frequency
within the stratum, applied identically to every record (deterministic per stratum).

This is a NEW, standalone diagnostic script. It does not modify or overwrite any
existing experiment file or its results parquet.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, load_primary, split_s1, prepared_cdfs
from e2_e3_heterogeneity import declared_partition

TAU = 0.90
MODELS = ["histgb", "ordered_logit", "multinomial_lr", "dlcon", "tabpfn",
          "lc_logit", "rp_logit"]
STRATA = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]


def pmf_from_cdf(cdf: np.ndarray) -> np.ndarray:
    return np.diff(cdf, axis=1, prepend=np.zeros((cdf.shape[0], 1)))


def oracle_width(pmf: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Smallest contiguous interval width (1..K) whose pmf mass >= tau, per row."""
    n, K = pmf.shape
    cs = np.cumsum(pmf, axis=1)  # cs[:, j] = mass(1..j+1)
    width = np.full(n, K, dtype=np.int8)
    found = np.zeros(n, dtype=bool)
    for w in range(1, K + 1):
        best = np.zeros(n)
        for lo in range(1, K - w + 2):
            hi = lo + w - 1
            hi_c = cs[:, hi - 1]
            lo_c = cs[:, lo - 2] if lo > 1 else 0.0
            best = np.maximum(best, hi_c - lo_c)
        newly = (~found) & (best >= tau - 1e-9)
        width[newly] = w
        found |= newly
    return width.astype(int)


def levelset_threshold(pmf: np.ndarray, target: float = TAU,
                        iters: int = 60) -> tuple[float, float]:
    """Bisect t in [0, 1] so mean_i sum_{y: pmf[i,y] > t} pmf[i,y] == target.

    f(t) = mean predicted mass captured by the strict level set is non-increasing
    in t (f(0-)=~1, f(1)=0), so bisection is well-defined even though f is a step
    function. Returns (t, achieved_f(t)).
    """
    def f(t):
        mask = pmf > t
        return float((pmf * mask).sum(axis=1).mean())

    lo_t, hi_t = 0.0, 1.0
    if f(lo_t) < target:  # degenerate: even t=0 doesn't reach target (shouldn't happen)
        return 0.0, f(lo_t)
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        if f(mid) >= target:
            lo_t = mid
        else:
            hi_t = mid
    return lo_t, f(lo_t)


def mean_N_levelset(pmf: np.ndarray, t: float) -> float:
    return float((pmf > t).sum(axis=1).mean())


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    part_te = declared_partition(te)
    part_tr = declared_partition(tr)
    y_tr = tr["y_kabco"].to_numpy()
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    rows = []
    per_model_frac_ge4 = {s: {} for s in STRATA}

    for model in MODELS:
        cdf_ca, cdf_te, timing = prepared_cdfs("s1", model, tr, ca, te)
        if not timing.get("cached", False):
            print(f"WARNING: {model} was NOT cached, had to fit fresh — unexpected", flush=True)
        pmf_te = pmf_from_cdf(cdf_te)
        ow = oracle_width(pmf_te)
        for s in STRATA:
            m = part_te == s
            n = int(m.sum())
            if n == 0:
                continue
            ow_s = ow[m]
            t_star, f_t = levelset_threshold(pmf_te[m])
            row = {
                "model": model, "stratum": s,
                "mean_oracle_width": float(ow_s.mean()),
                "frac_width_ge_4": float((ow_s >= 4).mean()),
                "frac_width_eq_5": float((ow_s == 5).mean()),
                "levelset_t": t_star, "levelset_f_at_t": f_t,
                "mean_N_levelset": mean_N_levelset(pmf_te[m], t_star),
                "n": n,
            }
            rows.append(row)
            per_model_frac_ge4[s][model] = row["frac_width_ge_4"]
        print(f"{model} done ({time.time()-t0:.0f}s)", flush=True)

    # Marginal (covariate-free) reference: phat(y) = TRAIN label freq within stratum.
    for s in STRATA:
        mtr = part_tr == s
        mte = part_te == s
        n_tr = int(mtr.sum())
        n_te = int(mte.sum())
        if n_tr == 0 or n_te == 0:
            continue
        freq = np.bincount(y_tr[mtr], minlength=6)[1:6].astype(float)
        freq = freq / freq.sum()
        pmf_marg = np.tile(freq, (1, 1))  # (1, 5) — identical for every record
        ow = oracle_width(pmf_marg)
        t_star, f_t = levelset_threshold(pmf_marg)
        rows.append({
            "model": "MARGINAL_reference", "stratum": s,
            "mean_oracle_width": float(ow[0]),
            "frac_width_ge_4": float(ow[0] >= 4),
            "frac_width_eq_5": float(ow[0] == 5),
            "levelset_t": t_star, "levelset_f_at_t": f_t,
            "mean_N_levelset": mean_N_levelset(pmf_marg, t_star),
            "n": n_te,
            "note": "deterministic (single pmf applied to all n_te records; n is the "
                    "test-stratum count for context, not a sample size for this metric)",
        })

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "adhoc_r3_task1_levelset.parquet", index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(out[["model", "stratum", "mean_oracle_width", "frac_width_ge_4",
               "frac_width_eq_5", "mean_N_levelset", "n"]].to_string(index=False))

    print("\n--- agreement across the 7 base models: spread of frac_width_ge_4 per stratum ---")
    for s in STRATA:
        vals = per_model_frac_ge4[s]
        arr = np.array(list(vals.values()))
        print(f"{s}: min={arr.min():.6f} max={arr.max():.6f} range={arr.max()-arr.min():.6f} "
              f"per-model={vals}")

    print(f"\nTask1 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
