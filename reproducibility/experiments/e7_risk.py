#!/usr/bin/env python
"""E7: severity-cost risk control (Thm 5a) and fatal-omission guarantee (Cor 5c).

(1) Sweep beta; verify empirical severity-weighted omission risk <= beta * kappa_max.
(2) Fatal-omission: kappa = 1{y=5}; verify the JOINT probability
    P(Y=5, Y not in set) <= beta using the full test-fold denominator. Report the
    conditional omission rate among fatal cases separately as a descriptive measure.
(3) Operating-point table for the triage narrative: fraction of cases whose interval
    lies in {A,K} ("escalate") and fraction with min >= B, per beta. Descriptive
    decision framing only; all statements are coverage/risk statements.
"""
import time

import numpy as np
import pandas as pd

from common import SEED, RESULTS, load_primary, split_s1, prepared_cdfs
from choir import cumulative_score, interval_sets
from choir.risk import crc_threshold
from choir.crash.costs import usdot_relative, fatal_omission

BETAS = [0.01, 0.02, 0.05, 0.10]
MODEL = "histgb"


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    cdf_ca, cdf_te, _ = prepared_cdfs("s1", MODEL, tr, ca, te)
    s_ca = cumulative_score(cdf_ca, y_ca)
    print(f"setup {time.time()-t0:.0f}s", flush=True)

    kappa = usdot_relative()
    kfatal = fatal_omission()
    rows = []
    for beta in BETAS:
        # (1) severity-weighted cost risk
        lam = crc_threshold(s_ca, kappa[y_ca - 1], kappa_max=kappa[-1], beta=beta)
        lo, hi = interval_sets(cdf_te, lam)
        miss = ~((y_te >= lo) & (y_te <= hi))
        risk = float(np.mean(kappa[y_te - 1] * miss))
        # (3) operating points (interval decision rules; Lemma 1 makes them monotone)
        escalate = float(np.mean(lo >= 4))          # set within {A, K}
        b_or_worse = float(np.mean(lo >= 3))
        rows.append({"guarantee": "cost_risk", "beta": beta, "lambda": lam,
                     "risk": risk, "bound": beta * kappa[-1],
                     "holds": bool(risk <= beta * kappa[-1]),
                     "avg_width": float(np.mean(hi - lo + 1)),
                     "frac_escalate_AK": escalate, "frac_min_B": b_or_worse,
                     "n": len(y_te)})

        # (2) fatal-omission (Cor 5c: exact-fatal recording assumed for K reports)
        lam_f = crc_threshold(s_ca, kfatal[y_ca - 1], kappa_max=1.0, beta=beta)
        lo_f, hi_f = interval_sets(cdf_te, lam_f)
        fatal_mask = y_te == 5
        fatal_omitted_mask = fatal_mask & ~((y_te >= lo_f) & (y_te <= hi_f))
        joint_numerator = int(fatal_omitted_mask.sum())
        n_test = int(len(y_te))
        n_fatal = int(fatal_mask.sum())
        fatal_joint_omission = joint_numerator / n_test
        fatal_conditional_omission = (joint_numerator / n_fatal if n_fatal else np.nan)
        rows.append({"guarantee": "fatal_omission", "beta": beta, "lambda": lam_f,
                     "risk": fatal_joint_omission, "bound": beta,
                     "risk_definition": "joint P(Y=5 and Y not in set)",
                     "joint_omission_numerator": joint_numerator,
                     "joint_omission_denominator": n_test,
                     "conditional_fatal_omission": fatal_conditional_omission,
                     "conditional_fatal_omission_denominator": n_fatal,
                     "holds": bool(fatal_joint_omission <= beta),
                     "avg_width": float(np.mean(hi_f - lo_f + 1)),
                     "n": n_test, "n_fatal": n_fatal})
        print(f"beta={beta}: cost risk {risk:.2f} <= {beta*kappa[-1]:.1f}; "
              f"joint fatal omission {fatal_joint_omission:.5f} <= {beta}; "
              f"conditional among fatalities {fatal_conditional_omission:.5f}")

    out = pd.DataFrame(rows)
    out["alpha_model"] = MODEL
    out["seed"] = SEED
    out.to_parquet(RESULTS / "e7_risk.parquet", index=False)
    print(f"E7 done in {time.time()-t0:.0f}s; all bounds hold: {out['holds'].all()}")


if __name__ == "__main__":
    main()
