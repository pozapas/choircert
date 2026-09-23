"""Declared reporting-noise models and the compatibility expansion (methods.tex §noise).

The misclassification structure (band or category-dependent map, plus delta) is a
DECLARED SENSITIVITY INPUT, never an estimated parameter. Results are guarantee
curves over delta (and over candidate maps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from choir.core.intervals import expand_intervals, expand_intervals_map


@dataclass(frozen=True)
class NoiseModel:
    """Compatibility assumption N(T, delta) (methods.tex Assumption N).

    Either a constant band (b_plus, b_minus) or a category-dependent map
    tmap[k] = (down_k, up_k): report k is compatible with truths [k-down_k, k+up_k].
    b_plus / down guards over-reporting (expansion reaches downward);
    b_minus / up guards under-reporting (expansion reaches upward; safety-critical).
    delta = declared beyond-compatibility mass, swept in sensitivity curves.
    """

    K: int = 5
    b_plus: int = 1
    b_minus: int = 1
    delta: float = 0.02
    tmap: dict = field(default_factory=dict)  # optional category-dependent override

    @staticmethod
    def kabco(delta: float = 0.02, exact_fatal: bool = True, a_reaches_down: int = 2):
        """KABCO default informed by KABCO-MAIS linkage studies (verification memo §1c).

        Under-reporting: one adjacent category (b_minus=1), beyond-mass in delta.
        Over-reporting from A (k=4) empirically reaches two categories down.
        Fatal (k=5) exact when death-record verified: T(5)={5}, and no report k<=4
        is compatible with truth 5 in the upward direction only if allowed explicitly
        (we keep up_4=1 so a report A remains compatible with truth K unless the user
        turns exact_fatal into full K-isolation; the conservative default keeps it).
        """
        tmap = {
            1: (0, 1),                  # O: truth in [O, C]
            2: (1, 1),                  # C: truth in [O, B]
            3: (1, 1),                  # B: truth in [C, A]
            4: (a_reaches_down, 1),     # A: truth in [A-2, K] (over-reporting heavy)
            5: (0, 0) if exact_fatal else (1, 0),  # K: exact (death records)
        }
        return NoiseModel(K=5, delta=delta, tmap=tmap)

    def expand(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Definition 3 applied to interval endpoints."""
        if self.tmap:
            return expand_intervals_map(lo, hi, self.tmap, self.K)
        return expand_intervals(lo, hi, self.b_plus, self.b_minus, self.K)

    def coverage_floor(self, alpha: float) -> float:
        """Conditional Theorem 3 floor for true-label coverage after expansion."""
        return 1.0 - alpha - self.delta

    def guarantee_curve(self, alpha: float, deltas=None) -> list[tuple[float, float]]:
        """(delta, floor) pairs for the sensitivity figure (E4)."""
        deltas = deltas if deltas is not None else [0.0, 0.01, 0.02, 0.05, 0.1]
        return [(d, 1.0 - alpha - d) for d in deltas]
