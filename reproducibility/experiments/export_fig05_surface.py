#!/usr/bin/env python
"""Fig 5 data prep: "The coverage deficit surface" (FIGURES_PLAN.md sec 6.2).

Builds the (hour x 5-mph speed-limit) coverage grid on the S1 TEST split --
the same plane as Fig 2 -- for THREE calibrations at alpha=0.10, histgb base:
  - marginal split-conformal (no conditioning),
  - class-conditional Mondrian on the declared KMeans-8 partition (a generic
    latent partition, per FIGURES_PLAN.md sec 6 shared prerequisite 2),
  - class-conditional Mondrian on a pre-declared speed-band x time-band
    partition (engineering-natural bands fixed a priori: speed <35 / 35-50 /
    55-65 / >=70 mph x time 22:00-05:59 / 06:00-21:59, n_min=1000 rollup).
The third layer conditions on the plane's own coordinates, so by construction
(Mondrian validity per cell) it closes the deficit on this exact plane; the
KMeans-8 layer narrows but does not close it, since it is not targeted at the
hour x speed interaction -- both facts are real findings the figure reports,
not a redesign of the plan's partition rule (no partition is swapped, only a
third one is added; see FIGURES_PLAN.md discussion 2026-07-11).

Also exports the wall marginals (coverage by speed limit alone, coverage by
hour alone) for all three methods, so the figure script never touches the raw
snapshot or refits anything.

Shared prerequisites per FIGURES_PLAN.md sec 6: histgb base model on S1 (via
common.prepared_cdfs), KMeans-8 fit on the training split only (same recipe as
experiments/e2_e3_heterogeneity.py and experiments/export_fig04_embedding.py).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import SEED, load_primary, split_s1, make_encoder, encode, prepared_cdfs
from choir import cumulative_score, interval_sets, split_calibrate, mondrian_calibrate

# OneDrive Files-On-Demand placeholders break polars' scan_parquet on this
# machine (it routes through a cloud-file code path that times out even once
# the file is hydrated). If WM_LOCAL_SNAPSHOT is set, read from that plain
# local copy instead of the OneDrive-synced path.
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
    """Pre-declared speed-band x time-band partition, fixed a priori (not tuned
    to outcomes). Rolls a (speed_band, time_band) cell up to its speed_band
    alone if the CALIBRATION fold has fewer than min_n rows in that cell --
    the rollup decision is made once, on fit_groups (the calibration split),
    and then applied identically to any other split via the returned labels
    for that split's own (speed, hour)."""
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

    # Declared KMeans-8 partition: fit on a 500k training subsample, EXACT same
    # recipe as e2_e3_heterogeneity.py / export_fig04_embedding.py.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=3, random_state=SEED).fit(
        X_tr[np.random.default_rng(SEED).choice(len(X_tr), 500_000, replace=False)])
    km_ca = km.predict(X_ca)
    km_te = km.predict(X_te)
    print(f"kmeans8 fit  [{time.time()-t0:.0f}s]", flush=True)

    qhat = split_calibrate(s_ca, ALPHA)
    q_mondrian = mondrian_calibrate(s_ca, km_ca, ALPHA)
    lam_te = np.array([q_mondrian[c] for c in km_te])

    # Pre-declared speed-band x time-band partition (fixed a priori; encodes
    # the plane's own coordinates). Rollup decision fit on the CALIBRATION
    # fold only, then applied to calibration and test alike.
    hour_ca = ca["hour"].to_numpy()
    speed_ca = ca["speed_limit"].to_numpy()
    hour_te = te["hour"].to_numpy()
    speed_te = te["speed_limit"].to_numpy()
    group_ca = declared_speedtime_group(speed_ca, hour_ca, fit_groups=(speed_ca, hour_ca))
    group_te = declared_speedtime_group(speed_te, hour_te, fit_groups=(speed_ca, hour_ca))
    group_counts = pd.Series(group_ca).value_counts()
    print(f"declared speed x time groups (calibration n): {group_counts.to_dict()}", flush=True)
    q_declared = mondrian_calibrate(s_ca, group_ca, ALPHA)
    lam_te_declared = np.array([q_declared[c] for c in group_te])

    lo_m, hi_m = interval_sets(cdf_te, qhat)
    lo_c, hi_c = interval_sets(cdf_te, lam_te)
    lo_d, hi_d = interval_sets(cdf_te, lam_te_declared)

    cov_marginal = (y_te >= lo_m) & (y_te <= hi_m)
    cov_mondrian = (y_te >= lo_c) & (y_te <= hi_c)
    cov_declared = (y_te >= lo_d) & (y_te <= hi_d)

    te = te.reset_index(drop=True)
    grid_df = pd.DataFrame({
        "hour": te["hour"].to_numpy(),
        "speed_limit": te["speed_limit"].to_numpy(),
        "cov_marginal": cov_marginal,
        "cov_mondrian": cov_mondrian,
        "cov_declared": cov_declared,
    })
    grid_df = grid_df[grid_df["speed_limit"] > 0].copy()
    grid_df["speed_bin"] = bin_speed(grid_df["speed_limit"])
    grid_df = grid_df.dropna(subset=["speed_bin"])
    grid_df["speed_bin"] = grid_df["speed_bin"].astype(int)

    surface = (
        grid_df.groupby(["hour", "speed_bin"])
        .agg(n=("cov_marginal", "size"),
             cov_marginal=("cov_marginal", "mean"),
             cov_mondrian=("cov_mondrian", "mean"),
             cov_declared=("cov_declared", "mean"))
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [range(24), SPEED_LABELS.tolist()], names=["hour", "speed_bin"]
    )
    surface = surface.set_index(["hour", "speed_bin"]).reindex(full_index).reset_index()
    surface["n"] = surface["n"].fillna(0).astype(int)
    surface.to_csv(OUT / "fig05_surface.csv", index=False, float_format="%.6f", na_rep="nan")
    print(f"fig05_surface.csv -> {OUT}  [{time.time()-t0:.0f}s]")

    # Wall marginals: coverage vs speed limit alone, coverage vs hour alone.
    by_speed = (
        grid_df.groupby("speed_bin")
        .agg(n=("cov_marginal", "size"), cov_marginal=("cov_marginal", "mean"),
             cov_mondrian=("cov_mondrian", "mean"), cov_declared=("cov_declared", "mean"))
        .reindex(SPEED_LABELS.tolist())
        .reset_index()
        .rename(columns={"speed_bin": "bin"})
    )
    by_speed.insert(0, "axis", "speed_limit")

    by_hour = (
        grid_df.groupby("hour")
        .agg(n=("cov_marginal", "size"), cov_marginal=("cov_marginal", "mean"),
             cov_mondrian=("cov_mondrian", "mean"), cov_declared=("cov_declared", "mean"))
        .reindex(range(24))
        .reset_index()
        .rename(columns={"hour": "bin"})
    )
    by_hour.insert(0, "axis", "hour")

    walls = pd.concat([by_speed, by_hour], ignore_index=True)
    walls.to_csv(OUT / "fig05_walls.csv", index=False, float_format="%.6f", na_rep="nan")
    print(f"fig05_walls.csv -> {OUT}  [{time.time()-t0:.0f}s total]")

    # Deepest-pocket annotation (n >= MIN_N cells only, matching the figure's mask rule).
    surf_valid = surface.dropna(subset=["cov_marginal"])
    surf_valid = surf_valid[surf_valid["n"] >= MIN_N]
    deepest = surf_valid.loc[(surf_valid["cov_marginal"] - 0.90).idxmin()]
    print(
        f"deepest pocket: hour={int(deepest.hour)} speed_bin={int(deepest.speed_bin)} "
        f"n={int(deepest.n)} cov_marginal={deepest.cov_marginal:.4f} "
        f"cov_mondrian={deepest.cov_mondrian:.4f} cov_declared={deepest.cov_declared:.4f} "
        f"delta_marginal={deepest.cov_marginal - 0.90:.4f}"
    )


if __name__ == "__main__":
    main()
