"""Deployment-shift machinery (methods.tex Thm 4a/4b, honesty box).

Observed strata: per-stratum Mondrian calibration with hierarchical rollup at a
sample-size floor (Thm 4a). New strata: density-ratio weighted calibration with the
three-number certificate (nominal level, TV-slack lower confidence bound, parametric
slack estimate) of Thm 4b. Concept drift is monitored, never corrected (DriftMonitor).

Split discipline: the density-ratio model is fit on calibration covariates plus
unlabeled target covariates. The ratio is target-to-calibration because it weights
calibration scores. Target rows used to fit the ratio must be disjoint from target
rows used for prediction and evaluation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Theorem 4a: hierarchical rollup
# ---------------------------------------------------------------------------

def rollup_map(
    strata: np.ndarray,
    hierarchy: list[dict],
    n_min: int,
) -> dict:
    """Map each leaf stratum to the coarsest-needed calibration cell.

    strata: training-split leaf-stratum labels. The returned map must be frozen before
    calibration and must never use calibration counts, scores, or labels.
    hierarchy: list of dicts, hierarchy[j][leaf] = parent label at level j+1
    (level 0 = leaf itself). The last level must map everything to one root.
    n_min: per-cell floor (paper default ceil(2/alpha)*50).

    Returns {leaf: cell_label} for one disjoint tree cut. Every returned cell has at
    least n_min training-reference rows unless the root itself is smaller. If one child
    must use a parent, all descendants of that parent use it. Leaf and parent cells
    therefore never overlap.
    """
    strata = np.asarray(strata)
    leaves, counts = np.unique(strata, return_counts=True)
    leaf_count = dict(zip(leaves.tolist(), counts.tolist()))

    # level labels per leaf: level 0 = leaf, then parents
    def level_label(leaf, j):
        lab = leaf
        for lvl in range(j):
            lab = hierarchy[lvl][lab]
        return lab

    n_levels = len(hierarchy) + 1
    # counts per label at each level
    level_counts: list[dict] = []
    for j in range(n_levels):
        cnt: dict = {}
        for leaf, c in leaf_count.items():
            lab = level_label(leaf, j)
            cnt[lab] = cnt.get(lab, 0) + c
        level_counts.append(cnt)

    children: dict[tuple[int, object], list[tuple[int, object]]] = {}
    for j in range(1, n_levels):
        for leaf in leaf_count:
            parent = (j, level_label(leaf, j))
            child = (j - 1, level_label(leaf, j - 1))
            children.setdefault(parent, [])
            if child not in children[parent]:
                children[parent].append(child)

    def cut(node):
        level, label = node
        if level == 0:
            return {node} if level_counts[0][label] >= n_min else None
        child_cuts = [cut(child) for child in children.get(node, [])]
        if child_cuts and all(result is not None for result in child_cuts):
            return set().union(*child_cuts)
        if level_counts[level][label] >= n_min or level == n_levels - 1:
            return {node}
        return None

    roots = {(n_levels - 1, level_label(leaf, n_levels - 1)) for leaf in leaf_count}
    selected = set()
    for root in roots:
        selected.update(cut(root) or {root})

    out = {}
    for leaf in leaf_count:
        path = [(j, level_label(leaf, j)) for j in range(n_levels)]
        matches = [node for node in path if node in selected]
        if len(matches) != 1:
            raise AssertionError("rollup cells do not form a disjoint partition")
        out[leaf] = matches[0]
    return out


def apply_rollup(strata: np.ndarray, roll: dict, hierarchy: list[dict]) -> np.ndarray:
    """Translate leaf strata into rollup cell labels; unseen leaves climb until they
    hit a cell present in the rollup, else the root."""
    strata = np.asarray(strata)
    n_levels = len(hierarchy) + 1
    known_cells = set(roll.values())

    def cell_for(leaf):
        if leaf in roll:
            return roll[leaf]
        lab = leaf
        for j in range(1, n_levels):
            lab = hierarchy[j - 1][lab] if lab in hierarchy[j - 1] else lab
            if (j, lab) in known_cells:
                return (j, lab)
        return (n_levels - 1, lab)

    out = np.empty(len(strata), dtype=object)
    for i, s in enumerate(strata):
        out[i] = cell_for(s)
    return out


# ---------------------------------------------------------------------------
# Theorem 4b: density ratio + slack diagnostics
# ---------------------------------------------------------------------------

class LogisticRegressionNP:
    """Minimal L2-regularized logistic regression via IRLS (numpy-only core dep)."""

    def __init__(self, l2: float = 1e-4, max_iter: int = 100, tol: float = 1e-8):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _design(X):
        X = np.asarray(X, dtype=float)
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y):
        Z = self._design(X)
        y = np.asarray(y, dtype=float)
        beta = np.zeros(Z.shape[1])
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-Z @ beta))
            W = np.clip(p * (1 - p), 1e-10, None)
            grad = Z.T @ (y - p) - self.l2 * beta
            H = (Z * W[:, None]).T @ Z + self.l2 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, grad)
            beta += step
            if np.abs(step).max() < self.tol:
                break
        self.coef_ = beta
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self._design(X) @ self.coef_))
        return np.column_stack([1 - p, p])


class DensityRatioEstimator:
    """w_hat(x) ~ dP*_X / dP_cal_X via a probabilistic classifier.

    Fit on source covariates (label 0) vs unlabeled target covariates (label 1);
    w_hat(x) = p(x)/(1-p(x)) * n0/n1. `clf` is any object with fit/predict_proba;
    defaults to the numpy IRLS logistic (realizability case of Thm 4b(iii)). When
    the weights enter `weighted_thresholds`, X_source must be the calibration
    covariates, not an earlier training split.
    """

    def __init__(self, clf=None, clip: float = 1e3):
        self.clf = clf if clf is not None else LogisticRegressionNP()
        self.clip = clip
        self._ratio0 = 1.0

    def fit(self, X_source, X_target):
        X_source, X_target = np.asarray(X_source, float), np.asarray(X_target, float)
        X = np.concatenate([X_source, X_target])
        y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
        self.clf.fit(X, y)
        self._ratio0 = len(X_source) / max(len(X_target), 1)
        return self

    def weights(self, X):
        p = self.clf.predict_proba(np.asarray(X, float))[:, 1]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.clip(p / (1 - p) * self._ratio0, 0.0, self.clip)


def tilted_resample(X_cal, w, size, rng):
    """Sample from Q_w: w-weighted resampling of calibration covariates."""
    w = np.asarray(w, float)
    p = w / w.sum()
    idx = rng.choice(len(w), size=size, replace=True, p=p)
    return np.asarray(X_cal, float)[idx]


def _binom_lcb(x: int, n: int, a: float) -> float:
    """Exact one-sided (Clopper-Pearson) lower confidence bound for a binomial
    proportion at level 1-a: the p solving P_p(Bin(n, p) >= x) = a; 0 when x = 0.

    Numpy-only: the upper tail is evaluated in log space (log-binomial coefficients
    from a cumulative log-factorial table) and inverted by bisection.
    """
    if n <= 0 or x <= 0:
        return 0.0
    if x >= n:
        return float(a ** (1.0 / n))
    ks = np.arange(x, n + 1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=float)))])
    logC = logfact[n] - logfact[ks] - logfact[n - ks]
    kf = ks.astype(float)

    def upper_tail(p: float) -> float:
        logs = logC + kf * np.log(p) + (n - kf) * np.log1p(-p)
        mx = logs.max()
        return float(np.exp(mx) * np.exp(logs - mx).sum())

    lo, hi = 0.0, 1.0  # upper_tail is increasing in p; tail(0)=0, tail(1)=1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if upper_tail(mid) < a:
            lo = mid
        else:
            hi = mid
    return float(lo)


def tv_slack_lcb(
    X_target,
    X_tilted,
    clf=None,
    test_frac: float = 0.5,
    conf: float = 0.95,
    rng=None,
) -> dict:
    """Assumption-free LOWER confidence bound on d_TV(P*_X, Q_w_hat) (Thm 4b(ii)).

    Trains a discriminator on half of each sample and evaluates the two class
    recalls on the held-out halves. Each recall gets an exact one-sided
    Clopper-Pearson lower bound at level 1 - (1-conf)/2 (Bonferroni), giving a
    finite-sample-valid level-conf LCB of ba and hence of 2*ba - 1 (floored at 0)
    -- the binomial bound stated in Thm 4b(ii). A large value certifies the
    transfer certificate is weak; a small value is necessary but NOT sufficient
    for a small slack (honesty box).
    """
    rng = rng or np.random.default_rng(0)
    X_target, X_tilted = np.asarray(X_target, float), np.asarray(X_tilted, float)

    def split(X):
        idx = rng.permutation(len(X))
        k = int(len(X) * (1 - test_frac))
        return X[idx[:k]], X[idx[k:]]

    tr1, te1 = split(X_target)
    tr0, te0 = split(X_tilted)
    clf = clf if clf is not None else LogisticRegressionNP()
    clf.fit(np.concatenate([tr0, tr1]),
            np.concatenate([np.zeros(len(tr0)), np.ones(len(tr1))]))

    x1 = int((clf.predict_proba(te1)[:, 1] > 0.5).sum())   # target recall count
    x0 = int((clf.predict_proba(te0)[:, 1] <= 0.5).sum())  # tilted recall count
    n1, n0 = len(te1), len(te0)
    acc1, acc0 = x1 / max(n1, 1), x0 / max(n0, 1)
    ba = 0.5 * (acc0 + acc1)
    a = 1.0 - conf
    ba_lcb = 0.5 * (_binom_lcb(x0, n0, a / 2) + _binom_lcb(x1, n1, a / 2))
    return {"ba": float(ba), "ba_lcb": float(ba_lcb),
            "tv_lcb": float(max(0.0, 2 * ba_lcb - 1)),
            "n_eval": int(n0 + n1)}


def weighted_thresholds(
    scores: np.ndarray,
    w_cal: np.ndarray,
    w_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Vectorized weighted conformal thresholds, one per test point (Thm 4b display).

    For each test weight w_t: q = inf{t : sum_i w_i 1{S_i<=t} >= (1-alpha)(sum w + w_t)}
    with the test point's mass placed at +infinity (conservative, Tibshirani et al. 2019).
    """
    scores = np.asarray(scores, float)
    w_cal = np.asarray(w_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    csum = np.cumsum(w_cal[order])
    total = w_cal.sum() + w_test
    target = (1.0 - alpha) * total
    idx = np.searchsorted(csum, target, side="left")
    out = np.where(idx < len(s_sorted), s_sorted[np.minimum(idx, len(s_sorted) - 1)], np.inf)
    return out


# ---------------------------------------------------------------------------
# Monitoring (detection, never correction)
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Conformal test martingale against exchangeability (power martingale).

    Consumes conformal p-values p_t of new observations; wealth
    M_t = prod_s integral or fixed-epsilon betting: M_t = prod_s eps * p_s^(eps-1).
    Rejection at threshold 1/alpha_mon is a Ville-valid sequential test.
    Detection only: a triggered monitor says re-calibrate, not how to correct.
    """

    def __init__(self, epsilon: float = 0.5, threshold: float = 100.0):
        if not 0 < epsilon < 1:
            raise ValueError("epsilon in (0,1)")
        self.epsilon = epsilon
        self.threshold = threshold
        self.log_wealth = 0.0
        self.history: list[float] = []

    def update(self, p_value: float) -> bool:
        p = min(max(p_value, 1e-12), 1.0)
        self.log_wealth += np.log(self.epsilon) + (self.epsilon - 1.0) * np.log(p)
        self.history.append(self.log_wealth)
        return self.triggered

    @property
    def wealth(self) -> float:
        return float(np.exp(self.log_wealth))

    @property
    def triggered(self) -> bool:
        return self.log_wealth >= np.log(self.threshold)


def conformal_pvalue(score: float, cal_scores: np.ndarray, rng=None) -> float:
    """Smoothed conformal p-value of a new score against calibration scores."""
    cal_scores = np.asarray(cal_scores, float)
    rng = rng or np.random.default_rng(0)
    n = len(cal_scores)
    gt = (cal_scores > score).sum()
    eq = (cal_scores == score).sum()
    return (gt + rng.uniform() * (eq + 1)) / (n + 1)
