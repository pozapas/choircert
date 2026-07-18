"""Ordinal cumulative score (methods.tex Definition 1).

s(x, y) = max{ F(y-1 | x), 1 - F(y | x) }, with F(0|x) = 0, F(K|x) = 1.
"""

from __future__ import annotations

import numpy as np


def _validate_cdf(cdf: np.ndarray) -> np.ndarray:
    """Validate an (n, K) conditional-CDF matrix: rows non-decreasing, last col 1."""
    cdf = np.asarray(cdf, dtype=float)
    if cdf.ndim != 2:
        raise ValueError(f"cdf must be 2-D (n, K); got shape {cdf.shape}")
    if np.any(np.diff(cdf, axis=1) < -1e-9):
        raise ValueError("cdf rows must be non-decreasing in the category index")
    if not np.allclose(cdf[:, -1], 1.0, atol=1e-6):
        raise ValueError("cdf last column must equal 1")
    return np.clip(cdf, 0.0, 1.0)


def cdf_from_proba(proba: np.ndarray) -> np.ndarray:
    """Cumulate an (n, K) class-probability matrix into a conditional CDF."""
    proba = np.asarray(proba, dtype=float)
    cdf = np.cumsum(proba, axis=1)
    cdf /= cdf[:, -1:]  # renormalize against float drift
    return cdf


def score_matrix(cdf: np.ndarray) -> np.ndarray:
    """All candidate-label scores: out[i, k-1] = s(x_i, k), k = 1..K.

    Vectorized Definition 1: F(y-1) is the CDF shifted right with 0 prepended.
    """
    cdf = _validate_cdf(cdf)
    below = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf[:, :-1]], axis=1)
    return np.maximum(below, 1.0 - cdf)


def cumulative_score(cdf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s(x_i, y_i) for observed labels y in {1..K} (1-indexed)."""
    y = np.asarray(y)
    if y.min() < 1 or y.max() > cdf.shape[1]:
        raise ValueError("labels must be 1-indexed in {1..K}")
    sm = score_matrix(cdf)
    return sm[np.arange(len(y)), y - 1]
