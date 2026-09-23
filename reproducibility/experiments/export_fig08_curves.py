#!/usr/bin/env python
"""Fig 8 data prep (a-c): "Risk control and the price surface" (FIGURES_PLAN.md
sec 6.5). Extends E7's beta grid so panels (a-c) show real variation instead of
the flat 4-point line in `experiments/results/e7_risk.parquet` (BETAS =
[0.01,0.02,0.05,0.10], where crc_threshold already collapses to lambda=0 for
both guarantees at every point -- verified 2026-07-11 while planning this
figure). Recomputes E7's EXACT recipe (same model/costs/split), just at a
wider, log-spaced beta grid reaching down to where lambda leaves 0, reusing the
cached S1 histgb CDFs -- no model refit.

Per FIGURES_PLAN.md sec 6.5's beta-grid ruling (2026-07-11), the grid is
declared BY RULE (log-spaced, fixed before looking at the curves; no knots
added or removed afterward), and the four original E7 beta values are kept as
distinct, separately-flagged points (`is_e7_grid` column) so E7's 4-point grid
remains the experiment of record and the dense curve is legible as the same
estimator evaluated finely, not a different one.

Escalation share ("min C >= 3", i.e. frac(lo >= 3)) is computed from the
cost_risk guarantee's own lambda at each beta, exactly as in e7_risk.py -- not
a separate threshold.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import SEED, load_primary, split_s1, prepared_cdfs
from choir import cumulative_score, interval_sets
from choir.risk import crc_threshold
from choir.crash.costs import usdot_relative, fatal_omission

MODEL = "histgb"
OUT = Path(__file__).resolve().parent.parent / "paper" / "figure_src" / "data"
RESULTS_OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

# Log-spaced grid, declared BY RULE per the plan's honesty conditions: 25
# points from 5e-4 to 0.10, fixed before inspecting any curve. E7's original
# 4 points (0.01/0.02/0.05/0.10) are unioned in explicitly so they are exact
# members of the grid (not just "nearby" logspace points) and can be flagged.
E7_BETAS = [0.01, 0.02, 0.05, 0.10]
_LOGSPACE = np.logspace(np.log10(5e-4), np.log10(0.10), 25)
BETAS = sorted(set(np.round(_LOGSPACE, 6).tolist()) | set(E7_BETAS))


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, timing = prepared_cdfs("s1", MODEL, tr, ca, te)
    s_ca = cumulative_score(cdf_ca, y_ca)
    print(f"setup ({'cached' if timing['cached'] else 'fit fresh'})  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    kappa = usdot_relative()
    kfatal = fatal_omission()
    n_fatal = int((y_te == 5).sum())

    rows = []
    for beta in BETAS:
        # cost_risk guarantee (Thm 5a) -- also the source of the "min C >= 3"
        # escalation share (Lemma 1: interval decision rules are monotone).
        lam = crc_threshold(s_ca, kappa[y_ca - 1], kappa_max=kappa[-1], beta=beta)
        lo, hi = interval_sets(cdf_te, lam)
        miss = ~((y_te >= lo) & (y_te <= hi))
        cost_risk = float(np.mean(kappa[y_te - 1] * miss))
        escalated_share = float(np.mean(lo >= 3))

        # fatal-omission guarantee (Cor 5c) -- separate lambda, separate costs.
        lam_f = crc_threshold(s_ca, kfatal[y_ca - 1], kappa_max=1.0, beta=beta)
        lo_f, hi_f = interval_sets(cdf_te, lam_f)
        fatal_mask = y_te == 5
        fatal_omitted = fatal_mask & ~((y_te >= lo_f) & (y_te <= hi_f))
        joint_fatal_omission = float(fatal_omitted.mean())
        conditional_fatal_omission = float(fatal_omitted.sum() / fatal_mask.sum())

        rows.append({
            "beta": beta,
            "is_e7_grid": beta in E7_BETAS,
            "lambda_cost": lam, "lambda_fatal": lam_f,
            "joint_fatal_omission": joint_fatal_omission, "fatal_bound": beta,
            "conditional_fatal_omission": conditional_fatal_omission,
            "joint_omission_numerator": int(fatal_omitted.sum()),
            "cost_risk": cost_risk, "cost_bound": beta * kappa[-1],
            "escalated_share": escalated_share,
        })
        print(f"beta={beta:.4f}: joint fatal omission {joint_fatal_omission:.5f} "
              f"(bound {beta:.4f}; conditional {conditional_fatal_omission:.5f}, "
              f"lam_f={lam_f:.3f})  cost {cost_risk:.3f} (bound "
              f"{beta*kappa[-1]:.1f}, lam={lam:.3f})  escalated {escalated_share:.3f}",
              flush=True)

    out = pd.DataFrame(rows)
    out["n_test"] = len(y_te)
    out["n_fatal"] = n_fatal
    out["kappa_max"] = kappa[-1]
    out.to_csv(OUT / "fig08_curves.csv", index=False, float_format="%.6f")
    out.to_csv(RESULTS_OUT / "e7_risk_dense.csv", index=False, float_format="%.6f")
    print(f"fig08_curves.csv -> {OUT}  [{time.time()-t0:.0f}s total]")
    print(f"e7_risk_dense.csv -> {RESULTS_OUT}")
    print(f"n_fatal={n_fatal}, n_test={len(y_te)}, kappa_max={kappa[-1]}")

    # Honesty condition (v): report the slack region and the knee as findings,
    # not smoothed over -- computed from the data, not asserted.
    slack_cost = out[out["lambda_cost"] == 0.0]
    slack_fatal = out[out["lambda_fatal"] == 0.0]
    knee_cost = out[out["lambda_cost"] > 0.0]["beta"].max() if (out["lambda_cost"] > 0).any() else None
    knee_fatal = out[out["lambda_fatal"] > 0.0]["beta"].max() if (out["lambda_fatal"] > 0).any() else None
    print(f"cost_risk slack (lambda=0) for beta >= {slack_cost['beta'].min():.4f}; "
          f"knee (last beta with lambda>0): {knee_cost}")
    print(f"fatal_omission slack (lambda=0) for beta >= {slack_fatal['beta'].min():.4f}; "
          f"knee (last beta with lambda>0): {knee_fatal}")
    over_cost = out[out["cost_risk"] > out["cost_bound"]]
    over_fatal = out[out["joint_fatal_omission"] > out["fatal_bound"]]
    print(f"cost_risk bound EXCEEDED at beta in {over_cost['beta'].tolist()}")
    print(f"fatal_omission bound EXCEEDED at beta in {over_fatal['beta'].tolist()}")


if __name__ == "__main__":
    main()
