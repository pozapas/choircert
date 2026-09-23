"""Build aggregate density grids for the Figure 3 corner display.

The input CSV files contain de-identified sampled covariate rows and stay in the
restricted research workspace. The output NPZ contains only normalized marginal
histograms and smooth two-dimensional density grids. It is safe to distribute
with the reproducibility supplement.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
VARS = ["speed_limit", "hour", "age", "vehicle_age"]
RANGES = {
    "speed_limit": (15, 85),
    "hour": (0, 23),
    "age": (14, 95),
    "vehicle_age": (0, 40),
}
PAD_FRAC = 0.06
PADDED = {
    name: (lo - PAD_FRAC * (hi - lo), hi + PAD_FRAC * (hi - lo))
    for name, (lo, hi) in RANGES.items()
}
BINS = {
    "speed_limit": np.arange(15, 86, 5),
    "hour": np.arange(-0.5, 24.5, 1.0),
    "age": np.linspace(14, 95, 28),
    "vehicle_age": np.arange(-0.5, 40.5, 2.0),
}
RNG = np.random.default_rng(20260704)
KDE_N = 15000


def subsample(frame):
    if len(frame) <= KDE_N:
        return frame
    return frame.iloc[RNG.choice(len(frame), KDE_N, replace=False)]


def density_grid(frame, xv, yv):
    x = frame[xv].to_numpy() + RNG.normal(0, 1e-6, len(frame))
    y = frame[yv].to_numpy() + RNG.normal(0, 1e-6, len(frame))
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.30)
    gx = np.linspace(*PADDED[xv], 120)
    gy = np.linspace(*PADDED[yv], 120)
    grid_x, grid_y = np.meshgrid(gx, gy)
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    return gx.astype("float32"), gy.astype("float32"), z.astype("float32")


def main():
    frames = {
        "o": subsample(pd.read_csv(HERE / "data" / "fig03_corner_o.csv")),
        "ka": subsample(pd.read_csv(HERE / "data" / "fig03_corner_ka.csv")),
    }
    out = {}
    for group, frame in frames.items():
        for name in VARS:
            edges = BINS[name]
            h, _ = np.histogram(frame[name].to_numpy(), bins=edges, density=True)
            out[f"hist_{group}_{name}_edges"] = edges.astype("float32")
            out[f"hist_{group}_{name}_height"] = (h / h.max()).astype("float32")

    for i, yv in enumerate(VARS):
        for j, xv in enumerate(VARS):
            if j >= i:
                continue
            for group, frame in frames.items():
                gx, gy, z = density_grid(frame, xv, yv)
                prefix = f"grid_{group}_{xv}_{yv}"
                out[f"{prefix}_x"] = gx
                out[f"{prefix}_y"] = gy
                out[f"{prefix}_z"] = z

    target = HERE / "data" / "fig03_corner_density.npz"
    np.savez_compressed(target, **out)
    print(f"wrote {target} with {len(out)} aggregate arrays")


if __name__ == "__main__":
    main()
