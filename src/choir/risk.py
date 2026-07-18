"""Conformal risk control for severity-weighted false-omission loss (methods.tex Thm 5a/5b).

Loss: l_lambda(x, y) = kappa(y)/kappa_max * 1{y not in C_lambda(x)}, non-increasing and
right-continuous in lambda with l_1 = 0 (C_1 = Y since s <= 1). Threshold per
Angelopoulos et al.: lambda_hat = inf{lambda : n/(n+1) * R_n(lambda) + 1/(n+1) <= beta},
lambda_hat = 1 (i.e. C = Y) if the set is empty.
"""

from __future__ import annotations

import numpy as np


def crc_threshold(
    scores: np.ndarray, costs: np.ndarray, kappa_max: float, beta: float
) -> float:
    """Select lambda_hat for the cost-weighted omission loss.

    scores: s(X_i, y_i) at the observed calibration labels; by the nested-interval
    structure the loss depends on lambda only through 1{score > lambda}.
    costs: kappa(y_i); kappa_max: the maximum of the full cost vector (pass it
    explicitly — calibration labels need not attain it).

    R_n(lambda) = (1/n) sum_i costs_i/kappa_max * 1{scores_i > lambda} is a
    right-continuous non-increasing step function with breakpoints at score values,
    so the infimum is attained at a calibration score value or at 0.
    Ties: at a candidate equal to a tied score value, R_n excludes the whole tie
    group only past its last occurrence; evaluating per-position overestimates R_n
    at earlier tie positions, which can only delay acceptance within the same
    value — conservative, never anti-conservative.
    """
    scores = np.asarray(scores, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if np.any(costs < 0) or kappa_max <= 0 or np.any(costs > kappa_max + 1e-12):
        raise ValueError("need 0 <= costs <= kappa_max, kappa_max > 0")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    n = len(scores)

    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    c_sorted = costs[order] / kappa_max

    tail = np.concatenate([np.cumsum(c_sorted[::-1])[::-1], [0.0]])  # tail[j] = sum_{i>=j}
    cand = np.concatenate([[0.0], s_sorted])          # candidate lambda values
    R = np.concatenate([[tail[0]], tail[1:]]) / n     # R_n at each candidate (per-position)

    ok = (n / (n + 1)) * R + 1.0 / (n + 1) <= beta
    if not ok.any():
        return 1.0  # lambda_max: C = Y, zero loss
    return float(cand[np.argmax(ok)])


def inflated_costs(y: np.ndarray, kappa: np.ndarray, b_minus: int) -> np.ndarray:
    """kappa_plus(y) = kappa(min(y + b_minus, K)) — the band-inflated cost of Thm 5b.

    kappa: length-K vector, kappa[k-1] = cost of category k, non-decreasing.
    """
    kappa = np.asarray(kappa, dtype=float)
    if np.any(np.diff(kappa) < 0):
        raise ValueError("kappa must be non-decreasing in severity")
    K = len(kappa)
    y = np.asarray(y)
    return kappa[np.minimum(y - 1 + b_minus, K - 1)]
