"""choir: a certification layer for ordinal, safety-critical prediction.

Guarantees are statements about prediction-set coverage and expected risk under
declared sampling assumptions. No causal quantities are estimated or reported.
"""

from choir.core.scores import cumulative_score, score_matrix, cdf_from_proba
from choir.core.intervals import interval_sets, expand_intervals
from choir.core.calibrate import (
    conformal_quantile,
    split_calibrate,
    mondrian_calibrate,
    weighted_quantile,
)
from choir.noise import NoiseModel
from choir.partitions import Partition
from choir.compose import Certificate, CertifiedOrdinal
from choir.risk import crc_threshold, inflated_costs

__all__ = [
    "cumulative_score",
    "score_matrix",
    "cdf_from_proba",
    "interval_sets",
    "expand_intervals",
    "conformal_quantile",
    "split_calibrate",
    "mondrian_calibrate",
    "weighted_quantile",
    "NoiseModel",
    "Partition",
    "Certificate",
    "CertifiedOrdinal",
    "crc_threshold",
    "inflated_costs",
]

__version__ = "0.1.0"
