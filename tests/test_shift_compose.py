"""Tests for the shift module (Thm 4a rollup, 4b diagnostics, DriftMonitor) and the
composition API (Thm 6): additive slack budget on product cells."""

import numpy as np

from choir.core.scores import cumulative_score
from choir.core.intervals import interval_sets
from choir.shift import (
    rollup_map, apply_rollup, DensityRatioEstimator, tilted_resample, tv_slack_lcb,
    weighted_thresholds, DriftMonitor, conformal_pvalue,
)
from choir.noise import NoiseModel
from choir.compose import CertifiedOrdinal

K = 5


def make_world(rng, n, shift=0.0, scale=1.0):
    x = rng.normal(0, 1, n) * scale + shift
    cuts = np.array([-1.5, 0.0, 1.0, 2.2])
    u = x[:, None] + rng.logistic(0, 1, (n, 1))
    y = (1 + (u > cuts).sum(axis=1)).astype(int)
    return x, y


def model_cdf(x, rng):
    mcuts = np.array([-1.2, 0.3, 1.4, 2.0])
    cdf4 = 1 / (1 + np.exp(-(mcuts[None, :] - 0.8 * x[:, None])))
    cdf = np.concatenate([cdf4, np.ones((len(x), 1))], axis=1)
    cdf = np.maximum.accumulate(cdf, axis=1)
    cdf = np.clip(cdf + rng.uniform(0, 1e-9, cdf.shape), 0, 1)
    cdf[:, -1] = 1.0
    return cdf


# ---------- rollup (Thm 4a packaging) ----------

def test_rollup_respects_floor_and_hierarchy():
    counties = np.array(["c1"] * 50 + ["c2"] * 2000 + ["c3"] * 30 + ["c4"] * 900)
    hierarchy = [
        {"c1": "cbsaA", "c2": "cbsaA", "c3": "cbsaB", "c4": "cbsaB"},
        {"cbsaA": "state", "cbsaB": "state"},
    ]
    roll = rollup_map(counties, hierarchy, n_min=1000)
    assert roll["c2"] == (0, "c2")            # big county stays a leaf
    assert roll["c1"] == (1, "cbsaA")         # small county rolls to CBSA (50+2000)
    assert roll["c3"] == (2, "state")         # cbsaB has 930 < 1000 -> state
    assert roll["c4"] == (2, "state")
    cells = apply_rollup(np.array(["c1", "c3", "c2"]), roll, hierarchy)
    assert list(cells) == [(1, "cbsaA"), (2, "state"), (0, "c2")]


# ---------- Thm 4b: density ratio recovers coverage; TV LCB is valid ----------

def test_density_ratio_weighted_coverage():
    alpha = 0.1
    rng = np.random.default_rng(1)
    x_tr = rng.normal(0, 1, 4000)     # training-split covariates (fit w-hat)
    x_cal, y_cal = make_world(rng, 6000)
    x_new = rng.normal(1.0, 1.0, 4000)  # unlabeled target covariates
    x_te, _ = make_world(rng, 30000)
    x_te = rng.normal(1.0, 1.0, 30000)  # target covariates
    cuts = np.array([-1.5, 0.0, 1.0, 2.2])
    u = x_te[:, None] + rng.logistic(0, 1, (len(x_te), 1))
    y_te = (1 + (u > cuts).sum(axis=1)).astype(int)  # same conditional (covariate shift)

    dre = DensityRatioEstimator().fit(x_tr.reshape(-1, 1), x_new.reshape(-1, 1))
    w_cal = dre.weights(x_cal.reshape(-1, 1))
    w_te = dre.weights(x_te.reshape(-1, 1))

    s_cal = cumulative_score(model_cdf(x_cal, rng), y_cal)
    lam = weighted_thresholds(s_cal, w_cal, w_te, alpha)
    lo, hi = interval_sets(model_cdf(x_te, rng), lam)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    # realizable logistic ratio (Gaussian mean shift) => slack O_p(n^-1/2); allow 2%
    assert cov >= 1 - alpha - 0.02, cov


def test_tv_slack_lcb_valid_and_detects_bad_weights():
    rng = np.random.default_rng(2)
    x_cal = rng.normal(0, 1, 8000).reshape(-1, 1)
    x_tgt = rng.normal(1.0, 1.0, 4000).reshape(-1, 1)

    # good weights: true ratio => Q_w ~ target => TV ~ 0 => LCB near 0
    w_good = np.exp(x_cal[:, 0] - 0.5)
    tilt_good = tilted_resample(x_cal, w_good, 4000, rng)
    d_good = tv_slack_lcb(x_tgt, tilt_good, rng=rng)
    assert d_good["tv_lcb"] <= 0.05, d_good

    # bad weights (uniform): Q_w = calibration law, true TV(N(0,1),N(1,1)) ~ 0.38
    tilt_bad = tilted_resample(x_cal, np.ones(len(x_cal)), 4000, rng)
    d_bad = tv_slack_lcb(x_tgt, tilt_bad, rng=rng)
    assert 0.05 < d_bad["tv_lcb"] <= 0.383, d_bad  # valid LCB: below true TV, but large
    assert d_bad["tv_lcb"] > d_good["tv_lcb"]


# ---------- DriftMonitor ----------

def test_drift_monitor_triggers_only_under_drift():
    rng = np.random.default_rng(3)
    cal = rng.normal(0, 1, 2000)
    m_null = DriftMonitor(threshold=100.0)
    for _ in range(1500):  # exchangeable stream: wealth should stay low
        m_null.update(conformal_pvalue(rng.normal(0, 1), cal, rng))
    assert not m_null.triggered

    m_drift = DriftMonitor(threshold=100.0)
    steps = 0
    for _ in range(1500):  # shifted stream: scores larger, p-values small
        steps += 1
        if m_drift.update(conformal_pvalue(rng.normal(1.5, 1), cal, rng)):
            break
    assert m_drift.triggered and steps < 1500


# ---------- Thm 6: composition, additive slack on product cells ----------

def test_composition_product_cells_with_noise():
    alpha, delta = 0.1, 0.02
    rng = np.random.default_rng(4)

    def draw(n):
        cls = rng.integers(0, 2, n)                       # heterogeneity class
        stratum = np.where(rng.uniform(size=n) < 0.5, "g0", "g1")
        x = rng.normal(0, 1, n) * np.where(cls == 0, 0.6, 2.0) + np.where(stratum == "g1", 0.8, 0)
        cuts = np.array([-1.5, 0.0, 1.0, 2.2])
        u = x[:, None] + rng.logistic(0, 1, (n, 1))
        y = (1 + (u > cuts).sum(axis=1)).astype(int)
        return x.reshape(-1, 1), cls, stratum, y

    def noisy(y):
        yt = y.copy()
        up = (rng.uniform(size=len(y)) < 0.25) & (y < K)   # within-band (paid by expansion)
        yt[up] = y[up] + 1
        far = (rng.uniform(size=len(y)) < delta) & (y <= K - 2)
        yt[far] = y[far] + 2                                # beyond-band (paid by delta)
        return yt

    x_cal, c_cal, g_cal, y_cal = draw(30000)
    x_te, c_te, g_te, y_te = draw(60000)
    yt_cal = noisy(y_cal)

    class Base:  # fixed misspecified model, features = x only
        def predict_proba(self, X):
            x = np.asarray(X)[:, 0]
            mcuts = np.array([-1.2, 0.3, 1.4, 2.0])
            cdf4 = 1 / (1 + np.exp(-(mcuts[None, :] - 0.8 * x[:, None])))
            cdf = np.concatenate([cdf4, np.ones((len(x), 1))], axis=1)
            cdf = np.maximum.accumulate(cdf, axis=1)
            cdf = np.clip(cdf + np.random.default_rng(0).uniform(0, 1e-9, cdf.shape), 0, 1)
            proba = np.diff(np.concatenate([np.zeros((len(x), 1)), cdf], axis=1), axis=1)
            return proba

    cls_cal, cls_te = c_cal.copy(), c_te.copy()
    cert = CertifiedOrdinal(
        base=Base(), K=K,
        partition=lambda X: cls_current[0],
        noise=NoiseModel(K=K, b_plus=1, b_minus=0, delta=delta),
        n_min=500,
    )
    cls_current = [cls_cal]
    cert.calibrate(x_cal, yt_cal, strata=g_cal)
    cls_current[0] = cls_te
    lo, hi = cert.predict_set(x_te, alpha=alpha, strata=g_te)

    floor = 1 - alpha - delta
    for c in (0, 1):
        for g in ("g0", "g1"):
            m = (c_te == c) & (g_te == g)
            cov = np.mean((y_te[m] >= lo[m]) & (y_te[m] <= hi[m]))
            tol = 4 * np.sqrt((alpha + delta) * (1 - alpha) / m.sum())
            assert cov >= floor - tol, (c, g, cov, floor)

    # certificates expose the additive budget
    certs = cert.certificate(alpha=alpha)
    assert all(abs(ct.floor - floor) < 1e-12 for ct in certs)
    assert len(certs) == 4
