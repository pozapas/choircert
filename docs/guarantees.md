# Guarantees

All results are finite-sample and distribution-free only under the assumptions stated in
the companion paper. The package's `tests/` exercise the implementations on simulated
data. Passing a test does not validate an assumption for an application dataset.

## Class-conditional coverage

For a partition frozen from the training split, and exchangeability within each observed
final calibration cell, coverage within each such cell is at least $1-\alpha$. A rollup
cell is the conditioning unit: the result does not claim coverage inside a raw child
cell after rollup. The efficiency comparison is a theorem about the specified predictor
class, not a claim that every fitted partition is substantively meaningful.

## True-label coverage under banded noise

Calibration uses reported labels. Under a declared banded compatibility assumption
$\mathbb{P}(Y \in \mathcal{T}(\tilde{Y})) \ge 1 - \delta$, the compatibility-expanded set
covers the true label with probability at least $1 - \alpha - \delta$. The band is a
declared sensitivity input, reported as curves over $\delta$, never estimated from the
bundled data. This is not a statement about an unidentified reporting channel.

## Deployment transfer certificates

For observed final cells, calibration is conditional on the cell and its exchangeability
assumption. For a new stratum, weighted conformal is valid under covariate shift when the
reported-label conditional law is stable and the density-ratio fit uses independent
reference and scored calibration samples. The exported tuple separates the nominal level,
an empirical discrepancy lower confidence bound, and a parametric slack estimate. The
empirical discrepancy is a diagnostic and must not be subtracted as if it were a coverage
penalty bound.

## Severity-cost risk control

Conformal risk control bounds the expected severity-weighted cost of screening misses by
$\beta \kappa_{\max}$ on the exchangeable observed-cell branch. With the declared
no-under-reporting or exact-fatality premise, the joint probability of excluding a true
fatality from the flagged set is at most $\beta$. This is a population-joint statement;
the conditional omission rate among fatal records has a different denominator and is not
bounded by $\beta$ without an additional argument.

## Composition

Compatible branches compose in a fixed order with an additive slack budget. Each slack
term is attributable to one declared assumption. A shift diagnostic alone cannot be
composed into a coverage guarantee.
