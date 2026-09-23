"""choir: a certification layer for ordinal, safety-critical prediction.

Guarantees are statements about prediction-set coverage and expected risk under
declared sampling assumptions. No causal quantities are estimated or reported.
"""

from choir.compose import Certificate, CertifiedOrdinal
from choir.core.calibrate import (
    conformal_quantile,
    mondrian_calibrate,
    split_calibrate,
    weighted_quantile,
)
from choir.core.intervals import expand_intervals, interval_sets
from choir.core.scores import cdf_from_proba, cumulative_score, score_matrix
from choir.noise import NoiseModel
from choir.partitions import Partition
from choir.risk import crc_threshold, inflated_costs

__all__ = [
    "Certificate",
    "CertifiedOrdinal",
    "NoiseModel",
    "Partition",
    "cdf_from_proba",
    "conformal_quantile",
    "crc_threshold",
    "cumulative_score",
    "expand_intervals",
    "inflated_costs",
    "interval_sets",
    "mondrian_calibrate",
    "score_matrix",
    "split_calibrate",
    "weighted_quantile",
]

__version__ = "0.2.0"
