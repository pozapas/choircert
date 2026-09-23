"""Raw and deployed interval prediction sets.

The mathematical set C_lambda(x) = {k : s(x,k) <= lambda} can be empty. The deployed
Convention 1 set replaces an empty raw set by the singleton argmin_k s(x,k). The two
classes remain explicit because the fallback preserves lower coverage bounds but can
invalidate coverage upper bounds and raw-set efficiency equalities.
"""

from __future__ import annotations

import numpy as np

from choir.core.scores import score_matrix


def raw_interval_sets(
    cdf: np.ndarray, lam: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints for the raw mathematical threshold set.

    An empty set has the endpoint sentinel (lo, hi) = (1, 0). Thus, the usual
    membership test ``(y >= lo) & (y <= hi)`` is false for every label.
    """
    sm = score_matrix(cdf)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (sm.shape[0],))
    member = sm <= lam[:, None]

    K = sm.shape[1]
    idx = np.arange(1, K + 1)
    nonempty = member.any(axis=1)
    lo = np.where(nonempty, np.where(member, idx, K + 1).min(axis=1), 1)
    hi = np.where(nonempty, np.where(member, idx, 0).max(axis=1), 0)

    width = hi - lo + 1
    if not np.array_equal(member.sum(axis=1)[nonempty], width[nonempty]):
        raise AssertionError("non-contiguous set: cdf violates monotonicity")
    return lo, hi


def interval_sets(cdf: np.ndarray, lam: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return deployed non-empty interval endpoints per row.

    lam may be a scalar or an (n,) vector of per-row thresholds (Mondrian use).
    This function applies the Convention 1 argmin fallback. Use raw_interval_sets
    for theorem checks that require the raw threshold family.
    """
    lo, hi = raw_interval_sets(cdf, lam)
    empty = lo > hi
    if empty.any():  # Convention 1
        sm = score_matrix(cdf)
        arg = sm[empty].argmin(axis=1) + 1
        lo = lo.copy()
        hi = hi.copy()
        lo[empty] = arg
        hi[empty] = arg
    return lo, hi


def expand_intervals(
    lo: np.ndarray, hi: np.ndarray, b_plus: int, b_minus: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Banded compatibility expansion (Definition 3): [lo - b_plus, hi + b_minus] ∩ Y.

    b_plus guards over-reporting (reach downward); b_minus guards under-reporting
    (reach upward). See methods.tex Assumption N.
    """
    if b_plus < 0 or b_minus < 0:
        raise ValueError("band widths must be non-negative")
    return np.maximum(lo - b_plus, 1), np.minimum(hi + b_minus, K)


def expand_intervals_map(
    lo: np.ndarray, hi: np.ndarray, tmap: dict[int, tuple[int, int]], K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Category-dependent expansion (Remark 3.2).

    tmap[k] = (down_k, up_k): a report k is compatible with truths [k - down_k, k + up_k].
    The expanded interval is the union of T(k) over k in [lo, hi]; with interval T(k)
    this is [min_k (k - down_k), max_k (k + up_k)] over k in [lo, hi].
    """
    lo_out = np.empty_like(lo)
    hi_out = np.empty_like(hi)
    for i, (a, b) in enumerate(zip(lo, hi)):
        ks = range(int(a), int(b) + 1)
        lo_out[i] = max(1, min(k - tmap[k][0] for k in ks))
        hi_out[i] = min(K, max(k + tmap[k][1] for k in ks))
    return lo_out, hi_out
