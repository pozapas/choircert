"""Partition sources for heterogeneity-conditional calibration (methods.tex Thm 2).

A partition is any function X -> labels fit WITHOUT calibration labels (split
discipline). Under exchangeability within the resulting observed final cells, Theorem 2
gives per-cell validity for arbitrary partitions; partition quality affects efficiency
only (Thm 2b). Sources supported:

- a callable (e.g. the MAP class of a latent-class model fit on the training split);
- a column of the feature matrix (declared covariate bins);
- any fitted object with .predict (sklearn-like classifier or clusterer).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class Partition:
    """Wraps a partition source into a uniform labels(X) interface."""

    def __init__(self, source: Callable | str | int | object, columns: list[str] | None = None):
        self._source = source
        self._columns = columns

    def labels(self, X) -> np.ndarray:
        src = self._source
        if callable(src):
            out = src(X)
        elif isinstance(src, (str, int)):
            if isinstance(src, str):
                if self._columns is None:
                    # pandas/polars-style access
                    out = np.asarray(X[src])
                else:
                    out = np.asarray(X)[:, self._columns.index(src)]
            else:
                out = np.asarray(X)[:, src]
        elif hasattr(src, "predict"):
            out = src.predict(X)
        else:
            raise TypeError(f"unsupported partition source: {type(src)}")
        out = np.asarray(out)
        if out.ndim != 1:
            raise ValueError("partition labels must be 1-D")
        return out
