"""Per-cell beyond-band mass implied by the cited linkage rates (Section 5.9)."""
import json, numpy as np
from common import load_primary, split_s1
from e2_e3_heterogeneity import declared_partition
df = load_primary(); tr, ca, te = split_s1(df)
y = te["y_kabco"].to_numpy().astype(int); c = declared_partition(te)
under = {1: 0.004, 2: 0.017, 3: 0.020, 4: 0.0, 5: 0.0}  # beyond-adjacent under-reporting
A_over = 0.382                                          # A reported, two levels down
out = {}
for cell in ["baseline", "motorcycle", "rural_highspeed", "unrestrained", "ALL"]:
    m = np.ones(len(y), bool) if cell == "ALL" else (c == cell)
    sh = {k: float((y[m] == k).mean()) for k in range(1, 6)}
    d_cat = sum(sh[k] * under[k] for k in range(1, 6))
    out[cell] = {"delta_category_dependent": d_cat, "delta_constant_band": d_cat + sh[4] * A_over}
print(json.dumps(out, indent=2))
