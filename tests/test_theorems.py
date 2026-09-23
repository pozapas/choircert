"""Theorem-level tests: each test simulates the exact regime of one guarantee and
asserts the finite-sample bound within binomial tolerance. These tests ARE the
package's reproducibility claim (CHOIR_framework.md 5.3).

All tests seeded; tolerance = 4 binomial SEs below the bound (one-sided: lower
coverage bounds may be exceeded freely).
"""

import numpy as np
import pytest

from choir.core.calibrate import mondrian_calibrate, split_calibrate, weighted_quantile
from choir.core.intervals import expand_intervals, interval_sets, raw_interval_sets
from choir.core.scores import cumulative_score
from choir.risk import crc_threshold, inflated_costs

K = 5


def RNG(seed):
    return np.random.default_rng(seed)


def simulate_ordered_logit(rng, n, shift=0.0, scale=1.0):
    """Heterogeneous ordered-logit world: latent u = x + noise against thresholds.

    Returns x, y (1..K), and a *misspecified* model CDF (guarantees must hold anyway).
    """
    x = rng.normal(0, 1, n) * scale + shift
    cuts = np.array([-1.5, 0.0, 1.0, 2.2])
    u = x[:, None] + rng.logistic(0, 1, (n, 1))
    y = 1 + (u > cuts).sum(axis=1)
    # model: ordered logit with WRONG thresholds and damped slope
    mcuts = np.array([-1.2, 0.3, 1.4, 2.0])
    logits = mcuts[None, :] - 0.8 * x[:, None]
    cdf4 = 1.0 / (1.0 + np.exp(-logits))
    cdf = np.concatenate([cdf4, np.ones((n, 1))], axis=1)
    cdf = np.maximum.accumulate(cdf, axis=1)  # enforce monotone rows
    # jitter to break ties (Remark 1.1); tiny, preserves monotonicity ordering a.s.
    cdf = np.clip(cdf + rng.uniform(0, 1e-9, cdf.shape), 0, 1)
    cdf[:, -1] = 1.0
    return x, y.astype(int), cdf


def coverage_tol(alpha, n):
    return 4 * np.sqrt(alpha * (1 - alpha) / n)


# ---------- Proposition 1 + Lemma 1 (E1 regime) ----------


def test_raw_empty_set_is_distinct_from_deployed_fallback():
    cdf = np.array([[0.45, 0.55, 1.0]])
    raw_lo, raw_hi = raw_interval_sets(cdf, lam=0.1)
    dep_lo, dep_hi = interval_sets(cdf, lam=0.1)

    assert (raw_lo[0], raw_hi[0]) == (1, 0)
    assert (dep_lo[0], dep_hi[0]) == (2, 2)
    y = 2
    assert not (raw_lo[0] <= y <= raw_hi[0])
    assert dep_lo[0] <= y <= dep_hi[0]

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_and_contiguity(alpha):
    """Prop 1 is a guarantee in expectation over the calibration draw, so the test
    averages coverage over independent calibration/test replicates."""
    rng = RNG(1)
    n_cal, n_te, reps = 2000, 8000, 25
    covs = []
    for _ in range(reps):
        _, y_cal, cdf_cal = simulate_ordered_logit(rng, n_cal)
        _, y_te, cdf_te = simulate_ordered_logit(rng, n_te)
        qhat = split_calibrate(cumulative_score(cdf_cal, y_cal), alpha)
        lo, hi = interval_sets(cdf_te, qhat)
        covs.append(np.mean((y_te >= lo) & (y_te <= hi)))
        assert np.all(lo >= 1) and np.all(hi <= K) and np.all(lo <= hi)
    mean_cov = np.mean(covs)
    se = np.std(covs, ddof=1) / np.sqrt(reps)
    assert mean_cov >= 1 - alpha - 4 * se, (mean_cov, se)
    # upper bound (scores continuous after jitter): 1 - alpha + 1/(n_cal+1)
    assert mean_cov <= 1 - alpha + 1 / (n_cal + 1) + 4 * se, (mean_cov, se)


# ---------- Theorem 2 (Mondrian class-conditional) ----------

def test_class_conditional_coverage():
    alpha = 0.1
    rng = RNG(2)
    # two latent classes with very different difficulty (scale 0.5 vs 2.5)
    def draw(n):
        c = rng.integers(0, 2, n)
        parts = [simulate_ordered_logit(rng, n, shift=0.0, scale=0.5),
                 simulate_ordered_logit(rng, n, shift=1.0, scale=2.5)]
        y = np.where(c == 0, parts[0][1], parts[1][1])
        cdf = np.where((c == 0)[:, None], parts[0][2], parts[1][2])
        return c, y, cdf
    c_cal, y_cal, cdf_cal = draw(6000)
    c_te, y_te, cdf_te = draw(60000)
    q = mondrian_calibrate(cumulative_score(cdf_cal, y_cal), c_cal, alpha)
    lam = np.array([q[c] for c in c_te])
    lo, hi = interval_sets(cdf_te, lam)
    for c in (0, 1):
        m = c_te == c
        cov = np.mean((y_te[m] >= lo[m]) & (y_te[m] <= hi[m]))
        assert cov >= 1 - alpha - coverage_tol(alpha, m.sum()), f"class {c}: {cov}"


# ---------- Theorem 3 (banded noise transfer) ----------

def test_banded_noise_transfer():
    alpha, delta, b = 0.1, 0.03, 1
    rng = RNG(3)
    def noisy(y, n):
        yt = y.copy()
        u = rng.uniform(size=n)
        # adjacent confusion, heavy (within band, unpaid): 25% up where possible
        adj = (u < 0.25) & (y < K)
        yt[adj] = y[adj] + 1
        # beyond-band mass delta: jump +2 (choose rows where feasible)
        v = rng.uniform(size=n)
        far = (v < delta) & (y <= K - 2)
        yt[far] = y[far] + 2
        return yt
    _, y_cal, cdf_cal = simulate_ordered_logit(rng, 6000)
    _, y_te, cdf_te = simulate_ordered_logit(rng, 60000)
    yt_cal = noisy(y_cal, len(y_cal))
    # calibrate on NOISY labels; guarantee is on TRUE labels after expansion
    qhat = split_calibrate(cumulative_score(cdf_cal, yt_cal), alpha)
    lo, hi = interval_sets(cdf_te, qhat)
    # band: report can exceed truth by up to b (over-reporting) => b_plus = b... note
    # here noise moves reports UP, i.e. report overstates: truth = report - shift,
    # so expansion must reach DOWNWARD by b_plus = 1 (and delta pays for jumps of 2).
    lo_e, hi_e = expand_intervals(lo, hi, b_plus=b, b_minus=0, K=K)
    cov_true = np.mean((y_te >= lo_e) & (y_te <= hi_e))
    bound = 1 - alpha - delta
    assert cov_true >= bound - coverage_tol(alpha + delta, len(y_te)), cov_true


# ---------- Theorem 4a (observed strata, arbitrary mixture) ----------

def test_group_weighted_shift_exact():
    alpha = 0.1
    rng = RNG(4)
    def draw(n, mix):
        g = (rng.uniform(size=n) < mix).astype(int)  # stratum indicator
        parts = [simulate_ordered_logit(rng, n, shift=-0.5, scale=1.0),
                 simulate_ordered_logit(rng, n, shift=1.5, scale=1.8)]
        y = np.where(g == 0, parts[0][1], parts[1][1])
        cdf = np.where((g == 0)[:, None], parts[0][2], parts[1][2])
        return g, y, cdf
    g_cal, y_cal, cdf_cal = draw(6000, mix=0.5)
    # deployment redistributes mass 0.5 -> 0.9 toward the hard stratum
    g_te, y_te, cdf_te = draw(60000, mix=0.9)
    q = mondrian_calibrate(cumulative_score(cdf_cal, y_cal), g_cal, alpha)
    lam = np.array([q[g] for g in g_te])
    lo, hi = interval_sets(cdf_te, lam)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert cov >= 1 - alpha - coverage_tol(alpha, len(y_te)), cov


# ---------- Theorem 4b sanity (known weights, covariate shift) ----------

def test_weighted_conformal_known_w():
    alpha = 0.1
    rng = RNG(5)
    # calibration x ~ N(0,1); target x ~ N(1,1) => w(x) = exp(x - 0.5)
    x_cal = rng.normal(0, 1, 8000)
    x_te = rng.normal(1, 1, 40000)
    cuts = np.array([-1.5, 0.0, 1.0, 2.2])
    def labels(x):
        u = x[:, None] + rng.logistic(0, 1, (len(x), 1))
        return (1 + (u > cuts).sum(axis=1)).astype(int)
    def model_cdf(x):
        mcuts = np.array([-1.2, 0.3, 1.4, 2.0])
        cdf4 = 1 / (1 + np.exp(-(mcuts[None, :] - 0.8 * x[:, None])))
        cdf = np.concatenate([cdf4, np.ones((len(x), 1))], axis=1)
        cdf = np.maximum.accumulate(cdf, axis=1)
        cdf = np.clip(cdf + rng.uniform(0, 1e-9, cdf.shape), 0, 1)
        cdf[:, -1] = 1.0
        return cdf
    y_cal, y_te = labels(x_cal), labels(x_te)
    cdf_cal, cdf_te = model_cdf(x_cal), model_cdf(x_te)
    s_cal = cumulative_score(cdf_cal, y_cal)
    w_cal = np.exp(x_cal - 0.5)
    # conservative single threshold: test weight at the essential sup over target draws
    qhat = weighted_quantile(s_cal, w_cal, test_weight=float(np.quantile(np.exp(x_te - 0.5), 0.999)), alpha=alpha)
    lo, hi = interval_sets(cdf_te, qhat)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert cov >= 1 - alpha - 0.02 - coverage_tol(alpha, len(y_te)), cov


# ---------- Theorem 5a (CRC) and 5b (noisy CRC) ----------

def test_crc_cost_risk():
    """Thm 5a bounds E[loss] over calibration AND test draws; average over replicates."""
    beta = 0.05
    rng = RNG(6)
    kappa = np.array([1.0, 20.0, 30.0, 150.0, 1500.0])
    reps, risks = 25, []
    for _ in range(reps):
        _, y_cal, cdf_cal = simulate_ordered_logit(rng, 4000)
        _, y_te, cdf_te = simulate_ordered_logit(rng, 20000)
        s_cal = cumulative_score(cdf_cal, y_cal)
        lam = crc_threshold(s_cal, kappa[y_cal - 1], kappa_max=kappa[-1], beta=beta)
        lo, hi = interval_sets(cdf_te, lam)
        miss = ~((y_te >= lo) & (y_te <= hi))
        risks.append(np.mean(kappa[y_te - 1] * miss))
    mean_risk = np.mean(risks)
    se = np.std(risks, ddof=1) / np.sqrt(reps)
    assert mean_risk <= beta * kappa[-1] + 4 * se, (mean_risk, beta * kappa[-1], se)


def test_crc_noisy_true_label_risk():
    beta, delta, b_minus = 0.05, 0.02, 1
    rng = RNG(7)
    kappa = np.array([1.0, 20.0, 30.0, 150.0, 1500.0])
    def noisy(y, n):
        # under-reporting: report below truth by 1 within band; beyond-band mass delta
        yt = y.copy()
        u = rng.uniform(size=n)
        adj = (u < 0.3) & (y > 1)
        yt[adj] = y[adj] - 1
        v = rng.uniform(size=n)
        far = (v < delta) & (y > 2)
        yt[far] = y[far] - 2
        return yt
    _, y_cal, cdf_cal = simulate_ordered_logit(rng, 8000)
    _, y_te, cdf_te = simulate_ordered_logit(rng, 80000)
    yt_cal = noisy(y_cal, len(y_cal))
    s_cal = cumulative_score(cdf_cal, yt_cal)
    kplus_cal = inflated_costs(yt_cal, kappa, b_minus)
    lam = crc_threshold(s_cal, kplus_cal, kappa_max=kappa[-1], beta=beta)
    lo, hi = interval_sets(cdf_te, lam)
    lo_e, hi_e = expand_intervals(lo, hi, b_plus=0, b_minus=b_minus, K=K)
    miss = ~((y_te >= lo_e) & (y_te <= hi_e))
    risk = np.mean(kappa[y_te - 1] * miss)
    bound = (beta + delta) * kappa[-1]
    se = np.std(kappa[y_te - 1] * miss) / np.sqrt(len(y_te))
    assert risk <= bound + 4 * se, (risk, bound)
