"""Predictive quality of the seven cached base models on the S1 test fold."""
import json
import numpy as np
from common import RESULTS, load_primary, split_s1, prepared_cdfs

MODELS = ["ordered_logit", "multinomial_lr", "lc_logit", "rp_logit", "histgb", "dlcon", "tabpfn"]
df = load_primary()
tr, ca, te = split_s1(df)
y = te["y_kabco"].to_numpy().astype(int)
ind = (np.arange(1, 5)[None, :] >= y[:, None]).astype(float)  # 1{y <= k}, k=1..4
out = {}
for m in MODELS:
    _, cdf, _ = prepared_cdfs("s1", m, tr, ca, te, X=(None, None, None))
    cdf = cdf.astype(float)
    rps = float(((cdf[:, :4] - ind) ** 2).sum(1).mean())
    p = np.diff(np.concatenate([np.zeros((len(cdf), 1)), cdf], 1), axis=1)
    ll = float(-np.log(np.clip(p[np.arange(len(y)), y - 1], 1e-12, 1)).mean())
    out[m] = {"rps": rps, "log_loss": ll}
    print(m, round(rps, 4), round(ll, 4), flush=True)
(RESULTS / "q1_quality.json").write_text(json.dumps(out, indent=2))
