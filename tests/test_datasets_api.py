"""Demo dataset loads and the sklearn-idiomatic API runs end to end."""

import numpy as np

from choir import CertifiedOrdinal, NoiseModel
from choir.datasets import DEMO_COLUMNS, load_demo


def test_demo_loads():
    rows, y, cols = load_demo()
    assert len(rows) == len(y) > 1000
    assert set(y) <= {1, 2, 3, 4, 5}
    assert 5 in set(y)  # fatalities present for the fatal-omission demo
    assert cols == [c for c in DEMO_COLUMNS if c != "y"]


def _encode(rows, cols):
    cats = {c: sorted({r[c] for r in rows}) for c in cols if isinstance(rows[0][c], str)}
    X = []
    for r in rows:
        v = []
        for c in cols:
            if c in cats:
                v += [1.0 if r[c] == lv else 0.0 for lv in cats[c]]
            else:
                v.append(float(r[c]))
        X.append(v)
    return np.array(X)


def test_certified_ordinal_end_to_end():
    rows, y, cols = load_demo()
    y = np.array(y)
    X = _encode(rows, cols)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, ca, te = np.split(idx, [int(0.5 * len(y)), int(0.75 * len(y))])

    from sklearn.ensemble import HistGradientBoostingClassifier
    base = HistGradientBoostingClassifier(max_iter=80, random_state=0)
    cert = CertifiedOrdinal(base=base, K=5, noise=NoiseModel.kabco(delta=0.02))
    cert.fit(X[tr], y[tr]).calibrate(X[ca], y[ca])
    lo, hi = cert.predict_set(X[te], alpha=0.10)
    assert np.all(lo >= 1) and np.all(hi <= 5) and np.all(lo <= hi)  # contiguous, valid range
    cov = np.mean((y[te] >= lo) & (y[te] <= hi))
    assert cov >= 0.85  # near nominal (true-label, expanded); loose for the small demo
