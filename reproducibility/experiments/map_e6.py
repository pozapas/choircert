#!/usr/bin/env python
"""F6: Texas county map of E6 spatial-transfer results (the money figure).

Held-out counties colored by unweighted per-county coverage; counties with a
transfer certificate (n >= 2000) hatched; calibration counties in light gray.
Uses pyshp + matplotlib polygons (no GDAL dependency).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
SHP = ROOT / "data" / "interim" / "shapes" / "cb_2023_us_county_20m.shp"
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"

e6 = pd.read_parquet(RESULTS / "e6_spatial.parquet")
cov = dict(zip(e6["county"].str.upper(), e6["unw_coverage"]))
certified = set(e6.loc[e6["n_test"] >= 2000, "county"].str.upper())
alpha = float(e6["alpha"].iloc[0])

sf = shapefile.Reader(str(SHP))
fields = [f[0] for f in sf.fields[1:]]
i_state = fields.index("STATEFP")
i_name = fields.index("NAME")

polys, colors, hatches = [], [], []
norm = Normalize(vmin=0.4, vmax=1.0)
cmap = plt.get_cmap("RdYlGn")

for shp, rec in zip(sf.shapes(), sf.records()):
    if rec[i_state] != "48":
        continue
    name = str(rec[i_name]).upper()
    pts = np.array(shp.points)
    parts = list(shp.parts) + [len(pts)]
    c = cov.get(name)
    for j in range(len(parts) - 1):
        polys.append(pts[parts[j]:parts[j + 1]])
        colors.append(cmap(norm(c)) if c is not None else (0.92, 0.92, 0.92, 1.0))
        hatches.append("//" if name in certified else None)

fig, ax = plt.subplots(figsize=(5.4, 5.0))
pc = PolyCollection(polys, facecolors=colors, edgecolors="white", linewidths=0.3)
ax.add_collection(pc)
# hatch layer for certified counties
cert_polys = [p for p, h in zip(polys, hatches) if h]
if cert_polys:
    ax.add_collection(PolyCollection(cert_polys, facecolors="none",
                                     edgecolors="black", linewidths=0.4, hatch="///"))
ax.autoscale()
ax.set_aspect(1.2)  # approx lat/lon aspect at Texas latitudes
ax.axis("off")
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label(f"held-out county coverage (nominal {1-alpha:.2f})", fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_title("Spatial transfer without certification: per-county coverage\n"
             "(gray = calibration counties; hatched = certificate issued, n≥2000)",
             fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "f_e6_county_map.pdf", dpi=200)
fig.savefig(FIGURES / "f_e6_county_map.png", dpi=200)
print("held-out counties drawn:", len({n for n in cov if cov[n] is not None}),
      "| certified:", len(certified))
print("saved", FIGURES / "f_e6_county_map.pdf")
