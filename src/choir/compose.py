"""Certificate objects, slack-budget algebra, and the CertifiedOrdinal API
(methods.tex Thm 6; CHOIR_framework.md 5.1).

Canonical composition order, enforced by construction:
condition (partition) -> weight (within cell) -> calibrate -> expand (noise) -> risk-adjust.
Slacks are additive and each is attributed to one declared assumption (Thm 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from choir.core.scores import cdf_from_proba, cumulative_score
from choir.core.intervals import interval_sets
from choir.core.calibrate import conformal_quantile
from choir.noise import NoiseModel
from choir.partitions import Partition
from choir.risk import crc_threshold, inflated_costs


@dataclass(frozen=True)
class Certificate:
    """A coverage certificate: nominal level minus named, attributed slacks."""

    nominal: float                      # 1 - alpha
    slacks: dict = field(default_factory=dict)  # name -> (value, assumption)
    cell: object = None
    n_cal: int = 0

    @property
    def floor(self) -> float:
        return self.nominal - sum(v for v, _ in self.slacks.values())

    def as_row(self) -> dict:
        row = {"cell": self.cell, "n_cal": self.n_cal, "nominal": self.nominal,
               "floor": self.floor}
        for name, (v, assumption) in self.slacks.items():
            row[f"slack_{name}"] = v
            row[f"assumption_{name}"] = assumption
        return row


class CertifiedOrdinal:
    """Wrap any ordinal severity model; export certified interval predictions.

    base: object with predict_proba(X) -> (n, K), or a callable X -> conditional CDF
    (n, K). The base model must be fit on the training split only.
    partition: None | Partition | anything Partition accepts (fit on training split).
    noise: NoiseModel or None.
    n_min: per-cell floor; cells below it roll up (product cell -> class -> global).
    """

    def __init__(self, base, K: int = 5, partition=None, noise: NoiseModel | None = None,
                 n_min: int = 1000):
        self.base = base
        self.K = K
        self.partition = (partition if isinstance(partition, Partition) or partition is None
                          else Partition(partition))
        self.noise = noise
        self.n_min = n_min
        self._cal: dict | None = None

    # -- base-model plumbing --

    def _cdf(self, X) -> np.ndarray:
        if callable(self.base) and not hasattr(self.base, "predict_proba"):
            cdf = np.asarray(self.base(X), dtype=float)
        else:
            cdf = cdf_from_proba(self.base.predict_proba(X))
        if cdf.shape[1] != self.K:
            raise ValueError(f"base model emits {cdf.shape[1]} categories, expected {self.K}")
        return cdf

    def fit(self, X_train, y_train):
        if hasattr(self.base, "fit"):
            self.base.fit(X_train, np.asarray(y_train))
        return self

    # -- calibration (condition -> calibrate) --

    def _keys(self, X, strata=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (cell_keys, class_keys) as string arrays 'class|stratum'."""
        cls = self.partition.labels(X) if self.partition is not None else np.zeros(len(X), int)
        cls_keys = np.array([str(c) for c in cls])
        if strata is None:
            return cls_keys.copy(), cls_keys
        strata = np.asarray(strata)
        cell_keys = np.array([f"{c}|{g}" for c, g in zip(cls_keys, strata)])
        return cell_keys, cls_keys

    def calibrate(self, X_cal, y_cal, strata=None):
        y_cal = np.asarray(y_cal)
        scores = cumulative_score(self._cdf(X_cal), y_cal)
        cell_keys, cls_keys = self._keys(X_cal, strata)
        self._cal = {
            "scores": scores, "y": y_cal,
            "cell_keys": cell_keys, "class_keys": cls_keys,
        }
        return self

    def _threshold_for(self, key: str, cls_key: str, alpha: float) -> tuple[float, int, str]:
        """Rollup: product cell -> class -> global, first level with n >= n_min."""
        cal = self._cal
        for level, mask in (
            ("cell", cal["cell_keys"] == key),
            ("class", cal["class_keys"] == cls_key),
            ("global", np.ones(len(cal["scores"]), bool)),
        ):
            n = int(mask.sum())
            if n >= self.n_min or level == "global":
                return conformal_quantile(cal["scores"][mask], alpha), n, level
        raise AssertionError("unreachable")

    # -- prediction (calibrate -> expand) --

    def predict_set(self, X, alpha: float = 0.1, strata=None):
        """Contiguous KABCO intervals with per-cell thresholds and noise expansion.

        Returns (lo, hi), 1-indexed inclusive endpoints on the TRUE-label scale if a
        noise model is set (expanded), else on the reported-label scale.
        """
        if self._cal is None:
            raise RuntimeError("call calibrate() first")
        cdf = self._cdf(X)
        keys, cls_keys = self._keys(X, strata)
        uniq, inverse = np.unique(keys, return_inverse=True)
        thr = np.empty(len(uniq))
        for j, key in enumerate(uniq):
            cls_key = cls_keys[np.argmax(inverse == j)]
            thr[j], _, _ = self._threshold_for(key, cls_key, alpha)
        lam = thr[inverse]
        lo, hi = interval_sets(cdf, lam)
        if self.noise is not None:
            lo, hi = self.noise.expand(lo, hi)
        return lo, hi

    def predict_set_risk(self, X, beta: float = 0.05, kappa=None, strata=None):
        """Severity-cost risk-controlled sets (Thm 5a/5b), cell-wise CRC thresholds.

        With a noise model set, CRC runs on band-inflated costs kappa_plus and the
        output is expanded (Thm 5b guarantee: risk <= beta*kmax + delta*kmax).
        """
        if self._cal is None:
            raise RuntimeError("call calibrate() first")
        if kappa is None:
            from choir.crash.costs import usdot_relative
            kappa = usdot_relative()
        kappa = np.asarray(kappa, float)
        kmax = float(kappa.max())
        b_minus = 0
        if self.noise is not None:
            b_minus = (max(up for _, up in self.noise.tmap.values())
                       if self.noise.tmap else self.noise.b_minus)
        cal = self._cal
        costs = (inflated_costs(cal["y"], kappa, b_minus) if b_minus > 0
                 else kappa[cal["y"] - 1])

        cdf = self._cdf(X)
        keys, cls_keys = self._keys(X, strata)
        uniq, inverse = np.unique(keys, return_inverse=True)
        thr = np.empty(len(uniq))
        for j, key in enumerate(uniq):
            cls_key = cls_keys[np.argmax(inverse == j)]
            for mask_level in (cal["cell_keys"] == key, cal["class_keys"] == cls_key,
                               np.ones(len(cal["scores"]), bool)):
                if mask_level.sum() >= self.n_min or mask_level.all():
                    thr[j] = crc_threshold(cal["scores"][mask_level],
                                           costs[mask_level], kmax, beta)
                    break
        lam = thr[inverse]
        lo, hi = interval_sets(cdf, lam)
        if self.noise is not None:
            lo, hi = self.noise.expand(lo, hi)
        return lo, hi

    # -- certificates (Thm 6 slack budget) --

    def certificate(self, alpha: float = 0.1) -> list[Certificate]:
        """Per-cell coverage certificates for all calibrated cells (observed strata).

        New-stratum certificates additionally need the TV-slack diagnostics of
        choir.shift (tv_slack_lcb); attach via shift tools in the experiments layer.
        """
        if self._cal is None:
            raise RuntimeError("call calibrate() first")
        out = []
        delta = self.noise.delta if self.noise is not None else 0.0
        for key in np.unique(self._cal["cell_keys"]):
            n = int((self._cal["cell_keys"] == key).sum())
            slacks = {}
            if self.noise is not None:
                slacks["noise"] = (delta, "N(T, delta) compatibility, declared")
            out.append(Certificate(nominal=1 - alpha, slacks=slacks, cell=key, n_cal=n))
        return out
