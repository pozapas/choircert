"""Split, Mondrian, and weighted conformal calibration (methods.tex Prop 1, Thm 2, Thm 4).

All calibration consumes scores only; enforcing that partitions/weights were fit
without calibration labels is the caller's contract (documented, and enforced by the
high-level CertifiedOrdinal API which fits partitions on the training split only).
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((1-alpha)(n+1))-th smallest score; +inf if index exceeds n (Prop 1)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def split_calibrate(scores: np.ndarray, alpha: float) -> float:
    """Marginal split conformal threshold (Proposition 1)."""
    return conformal_quantile(scores, alpha)


def mondrian_calibrate(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> dict:
    """Per-group conformal thresholds (Theorem 2 / Theorem 4a).

    groups: array of hashable group labels, same length as scores, produced by a
    function fit independently of the calibration labels (split discipline).
    Returns {group: threshold}. Groups absent at prediction time must be handled
    by the caller's rollup rule (see choir.shift.rollup).
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    return {
        g: conformal_quantile(scores[groups == g], alpha)
        for g in np.unique(groups)
    }


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    test_weight: float,
    alpha: float,
) -> float:
    """Weighted conformal threshold (Theorem 4b display equation).

    q = inf{ t : sum_i w_i 1{S_i <= t} >= (1-alpha) * (sum_i w_i + w_test) },
    with the test point's mass placed at +inf (conservative placement per
    Tibshirani et al. 2019); +inf when the calibration mass cannot reach the target.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or test_weight < 0:
        raise ValueError("weights must be non-negative")
    total = weights.sum() + test_weight
    if total <= 0:
        raise ValueError("all weights are zero")
    order = np.argsort(scores, kind="stable")
    csum = np.cumsum(weights[order])
    target = (1.0 - alpha) * total
    idx = np.searchsorted(csum, target, side="left")
    if idx >= len(scores):
        return np.inf
    return float(scores[order][idx])
