# Guarantees

All guarantees are finite-sample and distribution-free in the wrapped model. The proofs
are in the companion paper; the package's `tests/` reproduce each on simulated data.

## Class-conditional coverage

For any partition fit on the training split, coverage within every cell is at least
$1-\alpha$ (Theorem: exact latent-class-conditional coverage). Among predictors that
apply a fixed threshold per class, the class-conditional predictor is the smallest one
that stays valid in every class; marginal conformal prediction undercovers exactly the
harder classes.

## True-label coverage under banded noise

Calibration uses reported labels. Under a declared banded compatibility assumption
$\mathbb{P}(Y \in \mathcal{T}(\tilde{Y})) \ge 1 - \delta$, the compatibility-expanded set
covers the true label with probability at least $1 - \alpha - \delta$. The band is a
declared sensitivity input, reported as guarantee curves over $\delta$, never estimated.

## Deployment transfer certificates

For observed jurisdiction-year strata, per-stratum calibration is exact under any mixture
reweighting. For a new stratum under covariate shift, weighted conformal is valid up to a
label-free total-variation slack; the exported certificate is the triple (nominal level,
slack lower confidence bound, parametric slack estimate).

## Severity-cost risk control

Conformal risk control bounds the expected severity-weighted cost of screening misses by
$\beta \kappa_{\max}$. With exact fatality recording, the probability of excluding a true
fatality from the flagged set is at most $\beta$, regardless of the model, the covariate
distribution, or reporting noise between the non-fatal categories.

## Composition

The guarantees compose in a fixed order with an additive slack budget, and each slack
term is attributable to exactly one declared assumption.
