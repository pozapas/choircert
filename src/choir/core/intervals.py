"""Interval prediction sets (methods.tex Definition 2, Lemma 1, Convention 1).

C_lambda(x) = {k : s(x,k) <= lambda} is a contiguous interval by Lemma 1; if empty,
Convention 1 returns the singleton argmin_k s(x,k) (this only enlarges sets, so every
lower coverage bound is preserved).
"""

from __future__ import annotations

import numpy as np

from choir.core.scores import score_matrix


def interval_sets(cdf: np.ndarray, lam: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) 1-indexed inclusive interval endpoints per row.

    lam may be a scalar or an (n,) vector of per-row thresholds (Mondrian use).
    Never returns an empty set (Convention 1).
    """
    sm = score_matrix(cdf)  # (n, K)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (sm.shape[0],))
    member = sm <= lam[:, None]  # (n, K)

    K = sm.shape[1]
    idx = np.arange(1, K + 1)
    lo = np.where(member.any(axis=1), np.where(member, idx, K + 1).min(axis=1), 0)
    hi = np.where(member.any(axis=1), np.where(member, idx, 0).max(axis=1), 0)

    empty = lo == 0
    if empty.any():  # Convention 1
        arg = sm[empty].argmin(axis=1) + 1
        lo = lo.copy()
        hi = hi.copy()
        lo[empty] = arg
        hi[empty] = arg

    # Lemma 1 invariant: membership must be contiguous between lo and hi.
    # (Cheap runtime check; guards against a non-monotone cdf slipping through.)
    width = hi - lo + 1
    if not np.array_equal(member.sum(axis=1)[~empty], width[~empty]):
        raise AssertionError("non-contiguous set: cdf violates monotonicity")
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
