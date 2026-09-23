#!/usr/bin/env python
"""Fig 4 data prep: "The heterogeneity atlas" (FIGURES_PLAN.md sec 6.1).

Builds a frozen 120,000-row UMAP embedding of the S1 calibration split's encoded
design matrix, tagged with the declared KMeans-8 partition, the conformal score,
the marginal- and class-conditional (Mondrian) prediction-set widths at
alpha=0.10, the reported KABCO label, and the declared-stratum name -- everything
the four Fig-4 panels need, computed once and frozen to parquet so the figure
script never touches the raw snapshot or refits anything.

Shared prerequisites per FIGURES_PLAN.md sec 6: histgb base model on S1 (via
common.prepared_cdfs), KMeans-8 fit on the training split only (same procedure
as experiments/e2_e3_heterogeneity.py, replicated here for an identical partition).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import SEED, load_primary, split_s1, make_encoder, encode, prepared_cdfs
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate

ALPHA = 0.10
MODEL = "histgb"
N_EMBED = 120_000
OUT = Path(__file__).resolve().parent.parent / "paper" / "figure_src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Kept in sync with e2_e3_heterogeneity.py by hand: only "None" records an
# unrestrained driver. "Not Applicable" is an inapplicability code (it is what
# motorcycle/non-standard occupants carry) and "Other (Explain In Narrative)" is a
# residual bucket resolved only in the excluded narrative field.
UNRESTRAINED = {"None"}


def declared_partition(df: pd.DataFrame) -> np.ndarray:
    """Same priority-rule partition as e2_e3_heterogeneity.py (kept in sync by hand)."""
    moto = (df["Prsn_Type_ID"] == "Driver Of Motorcycle Type Vehicle").to_numpy()
    unre = df["Prsn_Rest_ID"].isin(UNRESTRAINED).to_numpy()
    rural_hs = ((df["Rural_Fl"] == "Y") & (df["speed_limit"] >= 55)).to_numpy()
    out = np.full(len(df), "baseline", dtype=object)
    out[rural_hs] = "rural_highspeed"
    out[unre] = "unrestrained"
    out[moto] = "motorcycle"
    return out


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    print(f"split: tr={len(tr)} ca={len(ca)} te={len(te)}  [{time.time()-t0:.0f}s]", flush=True)

    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_ca = ca["y_kabco"].to_numpy()
    print(f"encoded: X_tr={X_tr.shape} X_ca={X_ca.shape}  [{time.time()-t0:.0f}s]", flush=True)

    cdf_ca, cdf_te, timing = prepared_cdfs("s1", MODEL, tr, ca, te, X=(X_tr, X_ca, X_te))
    print(f"cdfs ready ({'cached' if timing['cached'] else 'fit fresh'})  [{time.time()-t0:.0f}s]",
          flush=True)
    s_ca = cumulative_score(cdf_ca, y_ca)

    # Declared KMeans-8 partition: fit on a 500k training subsample, EXACT same
    # recipe as e2_e3_heterogeneity.py so this is the same partition used there.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=3, random_state=SEED).fit(
        X_tr[np.random.default_rng(SEED).choice(len(X_tr), 500_000, replace=False)])
    km_ca = km.predict(X_ca)
    print(f"kmeans8 fit  [{time.time()-t0:.0f}s]", flush=True)

    # Marginal + class-conditional (Mondrian) thresholds, fit on the FULL
    # calibration fold (not just the UMAP subsample below).
    qhat = split_calibrate(s_ca, ALPHA)
    q_mondrian = mondrian_calibrate(s_ca, km_ca, ALPHA)
    lam_mondrian = np.array([q_mondrian[c] for c in km_ca])

    lo_m, hi_m = interval_sets(cdf_ca, qhat)
    width_marginal = (hi_m - lo_m + 1).astype(np.int16)
    lo_c, hi_c = interval_sets(cdf_ca, lam_mondrian)
    width_mondrian = (hi_c - lo_c + 1).astype(np.int16)

    stratum = declared_partition(ca)

    # Frozen 120k-row seeded subsample of the calibration split -- the embedding
    # is fit ONLY on this subsample (per plan), not on the full calibration fold.
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(ca), size=min(N_EMBED, len(ca)), replace=False)

    import umap
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.25, random_state=SEED)
    embedding = reducer.fit_transform(X_ca[sub])
    print(f"umap fit on {len(sub)} rows  [{time.time()-t0:.0f}s]", flush=True)

    out = pd.DataFrame({
        "u1": embedding[:, 0].astype(np.float32),
        "u2": embedding[:, 1].astype(np.float32),
        "kmeans8": km_ca[sub].astype(np.int8),
        "score": s_ca[sub].astype(np.float32),
        "width_marginal": width_marginal[sub],
        "width_mondrian": width_mondrian[sub],
        "y_kabco": y_ca[sub].astype(np.int8),
        "stratum": stratum[sub],
    })
    out.to_parquet(OUT / "fig04_embedding.parquet", index=False)
    print(out.describe(include="all").to_string())
    print(f"fig04_embedding.parquet -> {OUT}  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
