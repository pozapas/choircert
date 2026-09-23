#!/usr/bin/env python
"""E9: scale/runtime accounting.

(1) Harvest fit/predict wall times per base model from the CDF cache and
    calibration times from the E1 results.
(2) Microbenchmark the certification layer itself (score + quantile + intervals)
    at n = 1e5 .. 5e6 to substantiate the O(n log n) claim.
Hardware note recorded: 2015 4-core i7, 16 GB (models); DLCON/TabPFN on Colab T4.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, CACHE, SEED
from choir import cumulative_score, interval_sets, split_calibrate

rows = []
for f in sorted(CACHE.glob("s1_*.npz")):
    if "classes" in f.name:
        continue
    z = np.load(f)
    rows.append({"model": f.stem.replace("s1_", ""),
                 "fit_s": float(z["fit_s"]), "predict_s": float(z["predict_s"]),
                 "n_cal": int(z["cdf_ca"].shape[0]), "n_test": int(z["cdf_te"].shape[0])})
models = pd.DataFrame(rows)

e1 = pd.read_parquet(RESULTS / "e1_marginal.parquet")
cal = e1.groupby("model", as_index=False)["calibrate_s"].mean()
models = models.merge(cal, on="model", how="left")

# certification-layer scaling: synthetic CDFs, one calibration+prediction pass
rng = np.random.default_rng(SEED)
bench = []
for n in [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
    cdf4 = np.sort(rng.uniform(0, 1, (n, 4)).astype(np.float32), axis=1)
    cdf = np.concatenate([cdf4, np.ones((n, 1), np.float32)], axis=1)
    y = rng.integers(1, 6, n)
    t = time.time()
    s = cumulative_score(cdf, y)
    q = split_calibrate(s, 0.1)
    lo, hi = interval_sets(cdf, q)
    bench.append({"n": n, "layer_s": time.time() - t})
    print(f"n={n:>9,}: certification layer {bench[-1]['layer_s']:.2f}s", flush=True)
bench = pd.DataFrame(bench)

models.to_parquet(RESULTS / "e9_models_runtime.parquet", index=False)
bench.to_parquet(RESULTS / "e9_layer_scaling.parquet", index=False)
print(models.to_string(index=False))
print(bench.to_string(index=False))
