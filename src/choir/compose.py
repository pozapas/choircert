"""Certificate objects, slack-budget algebra, and the CertifiedOrdinal API
(methods.tex Thm 6; CHOIR_framework.md 5.1).

Canonical composition order, enforced by construction:
condition (partition) -> weight (within cell) -> calibrate -> expand (noise) -> risk-adjust.
Slacks are additive and each is attributed to one declared assumption (Thm 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from choir.core.calibrate import conformal_quantile
from choir.core.intervals import interval_sets
from choir.core.scores import cdf_from_proba, cumulative_score
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
    n_min: training-reference count floor used by freeze_rollup.
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
        self._rollup_spec: dict | None = None

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
        final_keys = self._apply_final_partition(cell_keys, cls_keys)
        self._cal = {
            "scores": scores, "y": y_cal,
            "cell_keys": cell_keys, "class_keys": cls_keys, "final_keys": final_keys,
        }
        return self

    def freeze_rollup(self, X_train, strata=None):
        """Freeze one disjoint rollup map from training-split covariates.

        Call this method before calibrate. The map uses no calibration rows, scores, or
        labels. If it is not called, calibrate keeps the declared leaf partition and does
        not select a rollup from calibration counts.
        """
        cell_keys, cls_keys = self._keys(X_train, strata)
        self._rollup_spec = self._fit_final_partition(cell_keys, cls_keys)
        return self

    def _fit_final_partition(self, cell_keys: np.ndarray, cls_keys: np.ndarray) -> dict:
        """Fit a disjoint product-cell, class, or global partition.

        If one product leaf in a class needs its class parent, every sibling leaf in that
        class uses the same parent. If that parent is too small, every class uses the global
        cell. These collapses prevent overlapping leaf and parent calibration events.
        """
        cell_keys = np.asarray(cell_keys).astype(str)
        cls_keys = np.asarray(cls_keys).astype(str)
        cell_counts = {key: int(np.sum(cell_keys == key)) for key in np.unique(cell_keys)}
        class_counts = {key: int(np.sum(cls_keys == key)) for key in np.unique(cls_keys)}

        if any(count < self.n_min for count in class_counts.values()):
            return {"global": True, "class_modes": {}, "known_leaves": set(cell_counts)}

        class_modes = {}
        for cls_key in np.unique(cls_keys):
            class_mask = cls_keys == cls_key
            leaves = np.unique(cell_keys[class_mask])
            collapse_class = any(cell_counts[leaf] < self.n_min for leaf in leaves)
            class_modes[str(cls_key)] = "class" if collapse_class else "leaf"
        return {
            "global": False,
            "class_modes": class_modes,
            "known_leaves": set(cell_counts),
        }

    def _apply_final_partition(
        self, cell_keys: np.ndarray, cls_keys: np.ndarray,
    ) -> np.ndarray:
        """Apply the pre-frozen map, or keep the declared leaf partition."""
        cell_keys = np.asarray(cell_keys).astype(str)
        cls_keys = np.asarray(cls_keys).astype(str)
        spec = self._rollup_spec
        if spec is None:
            return np.array([f"cell:{key}" for key in cell_keys], dtype=object)
        if spec["global"]:
            return np.full(len(cell_keys), "global:ALL", dtype=object)

        out = np.empty(len(cell_keys), dtype=object)
        for i, (key, cls_key) in enumerate(zip(cell_keys, cls_keys)):
            mode = spec["class_modes"].get(cls_key)
            if mode == "class":
                out[i] = f"class:{cls_key}"
            elif mode == "leaf" and key in spec["known_leaves"]:
                out[i] = f"cell:{key}"
            else:
                raise ValueError(
                    f"leaf {key!r} was not in the frozen rollup map and needs "
                    "new-stratum transfer"
                )
        return out

    def _final_cell_for(self, key: str, cls_key: str) -> str:
        """Resolve one requested leaf to its disjoint final partition cell."""
        return str(self._apply_final_partition(np.array([key]), np.array([cls_key]))[0])

    def _threshold_for(
        self, key: str, cls_key: str, alpha: float,
    ) -> tuple[float, int, str, str]:
        """Resolve a product leaf to its final calibrated cell.

        The returned cell identity is the conditioning event for the threshold. A
        rolled-up leaf therefore never receives a leaf-cell certificate.
        """
        cal = self._cal
        final_cell = self._final_cell_for(key, cls_key)
        mask = cal["final_keys"] == final_cell
        n = int(mask.sum())
        level = final_cell.split(":", 1)[0]
        return conformal_quantile(cal["scores"][mask], alpha), n, level, final_cell

    def resolved_cells(self, X, alpha: float = 0.1, strata=None) -> dict:
        """Return leaf-to-final-cell assignments for a prediction batch.

        Use this audit record when reporting conditional coverage. `leaf_cell` is
        descriptive. `final_cell` is the actual calibration conditioning event.
        """
        if self._cal is None:
            raise RuntimeError("call calibrate() first")
        keys, cls_keys = self._keys(X, strata)
        final = np.empty(len(keys), dtype=object)
        levels = np.empty(len(keys), dtype=object)
        n_cal = np.empty(len(keys), dtype=int)
        for key in np.unique(keys):
            idx = np.flatnonzero(keys == key)
            _, n, level, final_cell = self._threshold_for(key, cls_keys[idx[0]], alpha)
            final[idx] = final_cell
            levels[idx] = level
            n_cal[idx] = n
        return {
            "leaf_cell": keys.astype(str),
            "class_cell": cls_keys.astype(str),
            "final_cell": final.astype(str),
            "rollup_level": levels.astype(str),
            "final_n_cal": n_cal,
        }

    def partition_audit(self, alpha: float = 0.1) -> list[dict]:
        """Emit leaf definitions, final rollups, and calibration counts.

        This is an audit ledger, not a claim of leaf-cell validity after rollup.
        """
        if self._cal is None:
            raise RuntimeError("call calibrate() first")
        cal = self._cal
        rows = []
        for key in np.unique(cal["cell_keys"]):
            idx = np.flatnonzero(cal["cell_keys"] == key)
            cls_key = cal["class_keys"][idx[0]]
            _, final_n, level, final_cell = self._threshold_for(key, cls_key, alpha)
            rows.append({
                "leaf_cell": str(key), "class_cell": str(cls_key),
                "leaf_n_cal": len(idx), "final_cell": final_cell,
                "rollup_level": level, "final_n_cal": int(final_n),
            })
        return rows

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
            thr[j], _, _, _ = self._threshold_for(key, cls_key, alpha)
        lam = thr[inverse]
        lo, hi = interval_sets(cdf, lam)
        if self.noise is not None:
            lo, hi = self.noise.expand(lo, hi)
        return lo, hi

    def predict_set_risk(self, X, beta: float = 0.05, kappa=None, strata=None):
        """Severity-cost risk-controlled sets (Thm 5a/5b), cell-wise CRC thresholds.

        With a noise model set, CRC runs on band-inflated costs kappa_plus and the
        output is expanded (Thm 5b conditional bound: risk <= beta*kmax + delta*kmax).
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
            final_cell = self._final_cell_for(key, cls_key)
            final_mask = cal["final_keys"] == final_cell
            thr[j] = crc_threshold(cal["scores"][final_mask],
                                   costs[final_mask], kmax, beta)
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
        emitted = set()
        for record in self.partition_audit(alpha):
            if record["final_cell"] in emitted:
                continue
            emitted.add(record["final_cell"])
            slacks = {}
            if self.noise is not None:
                slacks["noise"] = (delta, "N(T, delta) compatibility, declared")
            out.append(Certificate(nominal=1 - alpha, slacks=slacks,
                                   cell=record["final_cell"], n_cal=record["final_n_cal"]))
        return out
