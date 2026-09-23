#!/usr/bin/env python
"""Round-4 (scoping, not part of the pipeline): does E2/E3's headline undercoverage
finding hold across all seven base models, or is it a histgb artifact?

Reproduces E2 (marginal conformal, per-stratum audit) and E3 (Mondrian on the
declared partition, the repair arm) EXACTLY as experiments/e2_e3_heterogeneity.py
computes them -- same declared_partition (imported, not reimplemented), same ALPHA,
same S1 split -- but looped over all seven E1 base models instead of hardcoding
histgb. No fitting: every model's CDF must load from cache via prepared_cdfs.

This is a NEW, standalone diagnostic script. It does not modify or overwrite
experiments/e2_e3_heterogeneity.py or its results parquet.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, load_primary, split_s1, make_encoder, encode, prepared_cdfs, coverage_stats
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate
from e2_e3_heterogeneity import declared_partition

ALPHA = 0.10
MODELS = ["histgb", "ordered_logit", "multinomial_lr", "dlcon", "tabpfn",
          "lc_logit", "rp_logit"]
STRATA = ["baseline", "motorcycle", "rural_highspeed", "unrestrained"]


def width_stats(lo, hi):
    w = hi - lo + 1
    return {"frac_width_ge_4": float((w >= 4).mean()),
            "frac_full_scale": float((w == 5).mean())}


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    part_ca, part_te = declared_partition(ca), declared_partition(te)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    e2_rows, e3_rows, model_rows = [], [], []

    for model in MODELS:
        cdf_ca, cdf_te, timing = prepared_cdfs("s1", model, tr, ca, te)
        cached = bool(timing.get("cached", False))
        if not cached:
            print(f"*** WARNING: {model} was NOT cache-loaded -- had to fit fresh. "
                  f"This invalidates the round's premise. ***", flush=True)

        s_ca = cumulative_score(cdf_ca, y_ca)

        # ---- E2: marginal conformal ----
        qhat = split_calibrate(s_ca, ALPHA)
        lo, hi = interval_sets(cdf_te, qhat)
        for s in STRATA:
            m = part_te == s
            st = coverage_stats(y_te[m], lo[m], hi[m])
            e2_rows.append({"model": model, "stratum": s, **st, **width_stats(lo[m], hi[m])})
        st_all = coverage_stats(y_te, lo, hi)
        e2_rows.append({"model": model, "stratum": "ALL", **st_all, **width_stats(lo, hi)})

        # ---- E3: Mondrian on the declared partition (no band expansion -- these
        # ARE the "base, pre-expansion" sets the paper's diagnostic arm refers to) ----
        q = mondrian_calibrate(s_ca, part_ca, ALPHA)
        lam = np.array([q[c] for c in part_te])
        lo_m, hi_m = interval_sets(cdf_te, lam)
        for s in STRATA:
            m = part_te == s
            st = coverage_stats(y_te[m], lo_m[m], hi_m[m])
            e3_rows.append({"model": model, "stratum": s, "qhat": q[s], **st,
                            **width_stats(lo_m[m], hi_m[m])})
        st_all_m = coverage_stats(y_te, lo_m, hi_m)
        e3_rows.append({"model": model, "stratum": "ALL", "qhat": np.nan, **st_all_m,
                        **width_stats(lo_m, hi_m)})

        model_rows.append({"model": model, "cached": cached,
                           "marginal_qhat": qhat,
                           "marginal_coverage_ALL": st_all["coverage"],
                           "marginal_width_ALL": st_all["avg_width"],
                           "mondrian_coverage_ALL": st_all_m["coverage"],
                           "mondrian_width_ALL": st_all_m["avg_width"]})
        print(f"{model}: cached={cached} ({time.time()-t0:.0f}s)", flush=True)

    e2_df = pd.DataFrame(e2_rows)
    e3_df = pd.DataFrame(e3_rows)
    model_df = pd.DataFrame(model_rows)
    e2_df.to_parquet(RESULTS / "adhoc_r4_e2_marginal.parquet", index=False)
    e3_df.to_parquet(RESULTS / "adhoc_r4_e3_mondrian.parquet", index=False)
    model_df.to_parquet(RESULTS / "adhoc_r4_model_summary.parquet", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda x: f"{x:.10f}")

    print("\n=== E2 (marginal conformal), per (model, stratum) ===")
    print(e2_df[e2_df.stratum != "ALL"][
        ["model", "stratum", "coverage", "avg_width", "frac_width_ge_4",
         "frac_full_scale", "n"]].to_string(index=False))

    print("\n=== E3 (Mondrian, declared partition = base pre-expansion sets), per (model, stratum) ===")
    print(e3_df[e3_df.stratum != "ALL"][
        ["model", "stratum", "coverage", "avg_width", "qhat", "frac_width_ge_4",
         "frac_full_scale", "n"]].to_string(index=False))

    print("\n=== per-model summary (marginal qhat, ALL-row coverage/width for both arms) ===")
    print(model_df.to_string(index=False))

    print("\n=== Q1/Q3: coverage on unrestrained and motorcycle (E2, marginal), all 7 models ===")
    for s in ("unrestrained", "motorcycle"):
        sub = e2_df[(e2_df.stratum == s)]
        vals = dict(zip(sub.model, sub.coverage))
        arr = np.array(list(vals.values()))
        print(f"{s}: min={arr.min():.10f} max={arr.max():.10f} range={arr.max()-arr.min():.10f}")
        for m, v in vals.items():
            print(f"    {m}: {v:.10f}")

    print("\n=== Q2: per-model ranking of the 4 strata by E2 coverage (ascending = hardest first) ===")
    for m in MODELS:
        sub = e2_df[(e2_df.model == m) & (e2_df.stratum != "ALL")].sort_values("coverage")
        order = " < ".join(f"{r.stratum}({r.coverage:.4f})" for r in sub.itertuples())
        print(f"{m}: {order}")

    print(f"\nTask (round 4) done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
