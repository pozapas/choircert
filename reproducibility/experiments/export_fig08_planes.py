#!/usr/bin/env python
"""Fig 8 data prep (d,e): "The price surface" (FIGURES_PLAN.md sec 6.5).

Builds the (hour x 5-mph speed-limit) WIDTH grid on the S1 TEST split -- the
same plane as Fig 2/5 -- for marginal split-conformal vs the declared
speed-band x time-band Mondrian partition, alpha=0.10, histgb base. Coverage
columns (cov_marg, cov_cc) are also included per the exported schema (cheap
to compute alongside width, useful for cross-checking against
fig05_surface.csv's cov_marginal/cov_declared), even though the Fig 8 render
only plots width -- the coverage story lives in Fig 5, per FIGURES_PLAN.md
sec 6.5's 2026-07-11 revision ("do not duplicate them here").

Reuses the EXACT declared speed x time partition recipe from
export_fig05_surface.py (same bands, same rollup rule, same fit-on-calibration
discipline) -- no KMeans-8 here, Fig 8 only needs the marginal/declared pair.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import load_primary, split_s1, make_encoder, encode, prepared_cdfs
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate

_local_snapshot = os.environ.get("WM_LOCAL_SNAPSHOT")
if _local_snapshot:
    common.SNAPSHOT = Path(_local_snapshot)

ALPHA = 0.10
MODEL = "histgb"
MIN_N = 400
OUT = Path(__file__).resolve().parent.parent / "paper" / "figure_src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SPEED_BINS = np.arange(15, 90, 5)
SPEED_LABELS = SPEED_BINS[:-1]


def bin_speed(speed: pd.Series) -> pd.Series:
    return pd.cut(speed, bins=SPEED_BINS, labels=SPEED_LABELS, right=False)


# Same declared speed x time partition as export_fig05_surface.py (kept in
# sync by hand -- identical bands/rollup rule, so the two figures' declared
# partitions are the same object).
SPEED_BAND_EDGES = [0, 35, 55, 70, 200]
SPEED_BAND_LABELS = ["lt35", "35_50", "55_65", "ge70"]
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}
MIN_N_GROUP = 1000


def speed_band(speed: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(SPEED_BAND_EDGES, speed, side="right") - 1
    idx = np.clip(idx, 0, len(SPEED_BAND_LABELS) - 1)
    return np.array(SPEED_BAND_LABELS)[idx]


def time_band(hour: np.ndarray) -> np.ndarray:
    return np.where(np.isin(hour, list(NIGHT_HOURS)), "night", "day")


def declared_speedtime_group(speed: np.ndarray, hour: np.ndarray, min_n: int = MIN_N_GROUP,
                              fit_groups: np.ndarray | None = None) -> np.ndarray:
    sb = speed_band(speed)
    tb = time_band(hour)
    combined = np.char.add(np.char.add(sb, "_"), tb)

    basis_sb = speed_band(fit_groups[0]) if fit_groups is not None else sb
    basis_tb = time_band(fit_groups[1]) if fit_groups is not None else tb
    basis_combined = np.char.add(np.char.add(basis_sb, "_"), basis_tb)
    counts = pd.Series(basis_combined).value_counts()
    rollup = {label: (label if counts.get(label, 0) >= min_n else label.rsplit("_", 1)[0])
              for label in set(basis_combined)}
    return np.array([rollup.get(c, c.rsplit("_", 1)[0]) for c in combined])


def main():
    t0 = time.time()
    df = load_primary()
    tr, ca, te = split_s1(df)
    print(f"split: tr={len(tr)} ca={len(ca)} te={len(te)}  [{time.time()-t0:.0f}s]", flush=True)

    enc = make_encoder().fit(tr)
    X_tr, X_ca, X_te = encode(enc, tr), encode(enc, ca), encode(enc, te)
    y_ca, y_te = ca["y_kabco"].to_numpy(), te["y_kabco"].to_numpy()
    print(f"encoded: X_tr={X_tr.shape} X_ca={X_ca.shape} X_te={X_te.shape}  [{time.time()-t0:.0f}s]",
          flush=True)

    cdf_ca, cdf_te, timing = prepared_cdfs("s1", MODEL, tr, ca, te, X=(X_tr, X_ca, X_te))
    print(f"cdfs ready ({'cached' if timing['cached'] else 'fit fresh'})  [{time.time()-t0:.0f}s]",
          flush=True)
    s_ca = cumulative_score(cdf_ca, y_ca)

    qhat = split_calibrate(s_ca, ALPHA)

    hour_ca = ca["hour"].to_numpy()
    speed_ca = ca["speed_limit"].to_numpy()
    hour_te = te["hour"].to_numpy()
    speed_te = te["speed_limit"].to_numpy()
    group_ca = declared_speedtime_group(speed_ca, hour_ca, fit_groups=(speed_ca, hour_ca))
    group_te = declared_speedtime_group(speed_te, hour_te, fit_groups=(speed_ca, hour_ca))
    q_declared = mondrian_calibrate(s_ca, group_ca, ALPHA)
    lam_te_declared = np.array([q_declared[c] for c in group_te])

    lo_m, hi_m = interval_sets(cdf_te, qhat)
    lo_d, hi_d = interval_sets(cdf_te, lam_te_declared)

    cov_marg = (y_te >= lo_m) & (y_te <= hi_m)
    cov_cc = (y_te >= lo_d) & (y_te <= hi_d)
    width_marg = (hi_m - lo_m + 1).astype(np.float64)
    width_cc = (hi_d - lo_d + 1).astype(np.float64)

    te = te.reset_index(drop=True)
    grid_df = pd.DataFrame({
        "hour": te["hour"].to_numpy(),
        "speed_limit": te["speed_limit"].to_numpy(),
        "cov_marg": cov_marg,
        "cov_cc": cov_cc,
        "w_marg": width_marg,
        "w_cc": width_cc,
    })
    grid_df = grid_df[grid_df["speed_limit"] > 0].copy()
    grid_df["speed_bin"] = bin_speed(grid_df["speed_limit"])
    grid_df = grid_df.dropna(subset=["speed_bin"])
    grid_df["speed_bin"] = grid_df["speed_bin"].astype(int)

    planes = (
        grid_df.groupby(["hour", "speed_bin"])
        .agg(n=("cov_marg", "size"),
             cov_marg=("cov_marg", "mean"),
             cov_cc=("cov_cc", "mean"),
             w_marg=("w_marg", "mean"),
             w_cc=("w_cc", "mean"))
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [range(24), SPEED_LABELS.tolist()], names=["hour", "speed_bin"]
    )
    planes = planes.set_index(["hour", "speed_bin"]).reindex(full_index).reset_index()
    planes["n"] = planes["n"].fillna(0).astype(int)
    planes.to_csv(OUT / "fig08_planes.csv", index=False, float_format="%.6f", na_rep="nan")
    print(f"fig08_planes.csv -> {OUT}  [{time.time()-t0:.0f}s total]")

    valid = planes.dropna(subset=["w_marg"])
    valid = valid[valid["n"] >= MIN_N]
    print(f"w_marg range (n>={MIN_N}): {valid['w_marg'].min():.3f}-{valid['w_marg'].max():.3f}")
    print(f"w_cc range (n>={MIN_N}): {valid['w_cc'].min():.3f}-{valid['w_cc'].max():.3f}")
    widest = valid.loc[valid["w_marg"].idxmax()]
    print(f"widest marginal cell: hour={int(widest.hour)} speed_bin={int(widest.speed_bin)} "
          f"n={int(widest.n)} w_marg={widest.w_marg:.3f} w_cc={widest.w_cc:.3f}")


if __name__ == "__main__":
    main()
