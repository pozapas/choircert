#!/usr/bin/env python
"""choir demo (runs in seconds, no external data): wrap a model, certify it.

Simulated crash-like world: two heterogeneity classes (benign vs severe-prone),
KABCO-style 5-level ordinal outcome, banded reporting noise on labels.
"""
import numpy as np

from choir import CertifiedOrdinal, NoiseModel

rng = np.random.default_rng(42)
K = 5


def draw(n):
    cls = rng.integers(0, 2, n)                       # latent heterogeneity class
    x = rng.normal(0, 1, n) * np.where(cls == 0, 0.6, 2.0)
    u = x[:, None] + rng.logistic(0, 1, (n, 1))
    y = (1 + (u > np.array([-1.5, 0.0, 1.0, 2.2])).sum(axis=1)).astype(int)
    X = np.column_stack([x, cls])
    return X, y


def report(y, n):                                     # banded under/over-reporting
    yt = y.copy()
    up = (rng.uniform(size=n) < 0.2) & (y < K)
    yt[up] += 1
    return yt


class OrderedLogitLike:
    """Stand-in base model: any object with fit / predict_proba works."""
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        x = np.asarray(X)[:, 0]
        cdf4 = 1 / (1 + np.exp(-(np.array([-1.2, 0.3, 1.4, 2.0])[None, :] - 0.9 * x[:, None])))
        cdf = np.concatenate([cdf4, np.ones((len(x), 1))], axis=1)
        cdf = np.maximum.accumulate(cdf, axis=1)
        return np.diff(np.concatenate([np.zeros((len(x), 1)), cdf], axis=1), axis=1)


X_tr, y_tr = draw(5000)
X_cal, y_cal = draw(20000)
X_new, y_new = draw(20000)
yt_cal = report(y_cal, len(y_cal))                    # calibration sees reported labels

cert = CertifiedOrdinal(
    base=OrderedLogitLike(),
    partition=lambda X: np.asarray(X)[:, 1].astype(int),   # latent class (fit upstream)
    noise=NoiseModel(K=5, b_plus=1, b_minus=0, delta=0.0), # declared band
    n_min=500,
)
cert.fit(X_tr, y_tr).calibrate(X_cal, yt_cal)

lo, hi = cert.predict_set(X_new, alpha=0.10)          # contiguous KABCO intervals
for c in cert.certificate(alpha=0.10):
    print(f"cell {c.cell}: n_cal={c.n_cal}, coverage floor >= {c.floor:.3f}")

cov = np.mean((y_new >= lo) & (y_new <= hi))
print(f"empirical TRUE-label coverage: {cov:.3f}  (floor 0.900)")
print(f"mean set width: {np.mean(hi - lo + 1):.2f} of {K} categories")
