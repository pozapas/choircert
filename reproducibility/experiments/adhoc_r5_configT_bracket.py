"""Round 5, Task 3 (laptop first-read): the Config P -> Config T achieved-width bracket.

Supervisor's licensed framing (SUPERVISOR_RESPONSE_2 sec 4): achieved width is a finite-sample
UPPER bound on the floor; enriching the feature set (Config P subset Config T) can only tighten
that upper end. This measures width_P vs width_T per declared stratum, marginal and Mondrian, at
alpha=0.10, for the CPU-feasible base models. It is the first read that decides whether the deep
models (DLCON, TabPFN) are worth a Colab run.

Config T = Config P + config_T_extra (pipeline/features.json): FHE_Collsn_ID, Harm_Evnt_ID,
Obj_Struck_ID, Veh_Damage_Severity1_Id, Prsn_Airbag_ID, Prsn_Ejct_ID, num_units. These are the
EMS/triage-observable fields, declared pre-registered in CHOIR_framework.md 3.1.

No cache is written or read (config-T CDFs must NOT land under config-P cache names). Fits are
inline. This does not modify common.py or any existing experiment.
"""
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "experiments")
from common import (SEED, RESULTS, CONT, CAT, BOOL, META, split_s1,  # noqa: E402
                    make_encoder, encode, coverage_stats, MODEL_REGISTRY)
from choir import (cdf_from_proba, cumulative_score, interval_sets,  # noqa: E402
                   split_calibrate, mondrian_calibrate)
from common import SNAPSHOT  # noqa: E402
from e2_e3_heterogeneity import declared_partition  # noqa: E402

ALPHA = 0.10
MODELS = ["histgb", "ordered_logit"]   # reweighted + clean linear; multinomial_lr dropped
                                       # (lbfgs on the wide config-T matrix is pathologically slow;
                                       # two models suffice for the P->T width bracket)
T_CAT = ["FHE_Collsn_ID", "Harm_Evnt_ID", "Obj_Struck_ID", "Veh_Damage_Severity1_Id",
         "Prsn_Airbag_ID", "Prsn_Ejct_ID"]
T_CONT = ["num_units"]
STRATA = ["motorcycle", "unrestrained", "rural_highspeed", "baseline"]


def load_with_T():
    cols = list(dict.fromkeys(META + CONT + CAT + BOOL + T_CAT + T_CONT))
    df = (pl.scan_parquet(SNAPSHOT).filter(pl.col("primary_row")).select(cols)
          .collect().to_pandas())
    df["y_kabco"] = df["y_kabco"].astype(np.int8)
    for c in CONT + T_CONT:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    for c in BOOL:
        df[c] = df[c].astype(np.float32)
    for c in CAT + T_CAT:
        df[c] = df[c].fillna("Missing").replace("", "Missing").astype("category")
    return df


def make_encoder_T():
    """Config-T encoder: same recipe as make_encoder() but with the extra fields."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cont = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                        sparse_output=False, dtype=np.float32)
    return ColumnTransformer(
        [("cont", cont, CONT + T_CONT), ("cat", cat, CAT + T_CAT),
         ("bool", "passthrough", BOOL)],
        verbose_feature_names_out=False)


def fit_cdfs(model_name, X_tr, y_tr, X_ca, X_te):
    t = time.time()
    model = MODEL_REGISTRY[model_name]().fit(X_tr, y_tr)
    cdf_ca = cdf_from_proba(model.predict_proba(X_ca)).astype(np.float32)
    cdf_te = cdf_from_proba(model.predict_proba(X_te)).astype(np.float32)
    return cdf_ca, cdf_te, time.time() - t


def run_suite(cdf_ca, cdf_te, y_ca, y_te, part_ca, part_te):
    """Marginal + Mondrian(declared) achieved coverage/width per stratum at ALPHA."""
    s_ca = cumulative_score(cdf_ca, y_ca)
    out = {}
    # marginal
    qhat = split_calibrate(s_ca, ALPHA)
    lo, hi = interval_sets(cdf_te, qhat)
    # mondrian on declared partition
    qmap = mondrian_calibrate(s_ca, part_ca, ALPHA)
    qte = np.array([qmap[p] for p in part_te])
    lo_m, hi_m = interval_sets(cdf_te, qte)
    for arm, (L, H) in {"marginal": (lo, hi), "mondrian": (lo_m, hi_m)}.items():
        w = H - L + 1
        for s in STRATA:
            m = part_te == s
            st = coverage_stats(y_te[m], L[m], H[m])
            out[(arm, s)] = {"coverage": st["coverage"], "avg_width": float(w[m].mean()),
                             "frac_full_scale": float((w[m] == 5).mean()), "n": int(m.sum())}
    return out


MAX_TRAIN = 800_000   # subsample train for the slow linear models on the wide config-T matrix;
                      # 800k is ample for the achieved-width comparison and keeps fits to minutes.


def main():
    t0 = time.time()
    df = load_with_T()
    tr, ca, te = split_s1(df)
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(n=MAX_TRAIN, random_state=SEED).reset_index(drop=True)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    part_ca, part_te = declared_partition(ca), declared_partition(te)
    print(f"loaded+split n_tr={len(tr):,} n_ca={len(ca):,} n_te={len(te):,} "
          f"({time.time()-t0:.0f}s)")

    encoders = {"P": make_encoder().fit(tr), "T": make_encoder_T().fit(tr)}
    X = {cfg: (encode(e, tr), encode(e, ca), encode(e, te)) for cfg, e in encoders.items()}
    print(f"encoded: P dim={X['P'][0].shape[1]}, T dim={X['T'][0].shape[1]} "
          f"({time.time()-t0:.0f}s)")

    rows = []
    for model_name in MODELS:
        for cfg in ("P", "T"):
            Xtr, Xca, Xte = X[cfg]
            cdf_ca, cdf_te, fit_s = fit_cdfs(model_name, Xtr, tr["y_kabco"].to_numpy(),
                                             Xca, Xte)
            res = run_suite(cdf_ca, cdf_te, y_ca, y_te, part_ca, part_te)
            for (arm, s), v in res.items():
                rows.append({"model": model_name, "config": cfg, "arm": arm,
                             "stratum": s, "fit_s": round(fit_s, 1), **v})
            print(f"  [{model_name}/{cfg}] fit {fit_s:.0f}s ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "adhoc_r5_configT_bracket.parquet", index=False)

    # the bracket: Mondrian achieved width, P vs T, per stratum, per model
    print("\n=== Mondrian achieved width: Config P -> Config T (the bracket upper end) ===")
    piv = (out[out.arm == "mondrian"]
           .pivot_table(index=["stratum", "model"], columns="config",
                        values="avg_width"))
    piv["delta_T_minus_P"] = piv["T"] - piv["P"]
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Mondrian coverage (must stay ~0.90 both configs) ===")
    covp = (out[out.arm == "mondrian"]
            .pivot_table(index=["stratum", "model"], columns="config", values="coverage"))
    print(covp.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nDone in {time.time()-t0:.0f}s. Full table -> "
          f"experiments/results/adhoc_r5_configT_bracket.parquet")


if __name__ == "__main__":
    main()
