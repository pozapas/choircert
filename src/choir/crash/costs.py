"""Severity cost vectors for risk control (methods.tex §risk).

usdot_relative: order-of-magnitude comprehensive-crash-cost scale relative to O=1,
per the USDOT/FHWA VSL-based guidance. Exact dollar figures and citation year are
pinned during E7 (verification memo §4 item 4); the guarantees only require kappa
non-decreasing, and all E7 results are reported for the pinned vector.
"""

from __future__ import annotations

import numpy as np


def usdot_relative() -> np.ndarray:
    """Relative comprehensive-cost scale (O, C, B, A, K), kappa(O)=1."""
    return np.array([1.0, 20.0, 30.0, 150.0, 1500.0])


def fatal_omission() -> np.ndarray:
    """Indicator cost for the fatal-omission guarantee (Corollary 5c)."""
    return np.array([0.0, 0.0, 0.0, 0.0, 1.0])
