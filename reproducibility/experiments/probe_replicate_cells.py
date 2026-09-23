"""Feasibility probe for the level-set impossibility result (fast version).

Question: within the motorcyclist / unrestrained strata, do exactly-replicated
recorded-covariate patterns exist in enough volume to estimate p(y|x) model-free?

Any predictor is a sigma(X)-measurable function, so two records with identical x get
identical predictions. Within an exact x-cell the empirical label frequency is therefore
a model-free estimate of the oracle conditional law, and no base model can beat it.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from common import load_primary, CONT, CAT, BOOL  # noqa: E402

FEATS = CONT + CAT + BOOL

df = load_primary()
print(f"total primary rows: {len(df):,}", flush=True)

moto = (df["Prsn_Type_ID"] == "Driver Of Motorcycle Type Vehicle").to_numpy()
unre = df["Prsn_Rest_ID"].isin({"None"}).to_numpy()
rural_hs = ((df["Rural_Fl"] == "Y") & (df["speed_limit"] >= 55)).to_numpy()
print(f"motorcycle  : {moto.sum():,}", flush=True)
print(f"unrestrained: {unre.sum():,}", flush=True)
print(f"rural_hs    : {rural_hs.sum():,}", flush=True)
print(f"feature vector: {len(FEATS)} fields "
      f"({len(CONT)} cont, {len(CAT)} cat, {len(BOOL)} bool)", flush=True)


def cellkey(sub: pd.DataFrame) -> np.ndarray:
    """Integer code per exact covariate pattern. NaN is its own level."""
    codes = np.empty((len(sub), len(FEATS)), dtype=np.int64)
    for j, c in enumerate(FEATS):
        s = sub[c]
        if str(s.dtype) == "category":
            s = s.astype(str)
        # factorize maps NaN -> -1, which is a distinct level, exactly what we want
        codes[:, j] = pd.factorize(s, use_na_sentinel=True)[0]
    # hash rows to a single key
    order = np.lexsort(codes.T[::-1])
    srt = codes[order]
    newgrp = np.ones(len(srt), dtype=bool)
    if len(srt) > 1:
        newgrp[1:] = (srt[1:] != srt[:-1]).any(axis=1)
    gid_sorted = np.cumsum(newgrp) - 1
    gid = np.empty(len(sub), dtype=np.int64)
    gid[order] = gid_sorted
    return gid


for name, mask in [("motorcycle", moto), ("unrestrained", unre), ("rural_highspeed", rural_hs)]:
    sub = df.loc[mask, FEATS + ["y_kabco"]].reset_index(drop=True)
    gid = cellkey(sub)
    y = sub["y_kabco"].to_numpy()
    sizes = np.bincount(gid)
    print(f"\n=== {name}: n={len(sub):,} ===", flush=True)
    print(f"  distinct exact x-patterns : {len(sizes):,}", flush=True)
    for thr in (2, 5, 10, 30, 100):
        sel = sizes >= thr
        print(f"  patterns with n_x >= {thr:<3d}  : {sel.sum():>7,}   "
              f"covering {sizes[sel].sum():>7,} records "
              f"({100*sizes[sel].sum()/len(sub):5.2f}% of stratum)", flush=True)
    print(f"  max cell size             : {sizes.max():,}", flush=True)

    # label spread within replicated cells: the model-free signal
    for thr in (10, 30):
        big = np.flatnonzero(sizes >= thr)
        if len(big) == 0:
            print(f"  [n_x>={thr}] no cells", flush=True)
            continue
        keep = np.isin(gid, big)
        g, yy = gid[keep], y[keep]
        # distinct labels observed per cell
        tab = pd.crosstab(g, yy)
        nuniq = (tab > 0).sum(axis=1)
        # empirical N(x,t) at a few levels t, model-free
        frac = tab.div(tab.sum(axis=1), axis=0)
        line = f"  [n_x>={thr}] cells={len(big):,} records={keep.sum():,} " \
               f"mean#labels={nuniq.mean():.3f}"
        for t in (0.0, 0.05, 0.10, 0.20):
            Nt = (frac > t).sum(axis=1)
            line += f" | meanN(t={t:.2f})={Nt.mean():.3f}"
        print(line, flush=True)
