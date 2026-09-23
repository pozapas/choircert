# CHOIR

[![PyPI version](https://img.shields.io/pypi/v/choircert.svg)](https://pypi.org/project/choircert/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21434172.svg)](https://doi.org/10.5281/zenodo.21434172)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A certification layer for ordinal, safety-critical prediction.** Distribution name
on PyPI: `choircert`; import name: `choir`.

Wrap any ordinal severity model and obtain finite-sample, distribution-free certificates:
contiguous ordinal prediction sets with marginal and observed-cell coverage
(training-frozen heterogeneity and jurisdiction-year strata), coverage on the *true*
label under a declared banded reporting-noise assumption, deployment-shift diagnostics,
and severity-weighted risk control including a fatal-omission bound. These statements are
conditional on the assumptions declared for the selected branch and are composable with
an explicit slack budget.

Every guarantee is a statement about prediction-set coverage or expected risk under a
declared sampling assumption. The package estimates no causal quantities.

## Install

```
pip install choircert            # core (numpy only)
pip install "choircert[torch]"   # + the DLCON deep base model
pip install "choircert[econ,maps]"  # + scipy models, county maps
```

## Quickstart

```python
import numpy as np
from choir import CertifiedOrdinal, NoiseModel

cert = CertifiedOrdinal(
    base=any_model_with_predict_proba,     # ordered logit, XGBoost, deep net, ...
    partition=latent_class_assigner,       # fit on the training split (or None)
    noise=NoiseModel.kabco(delta=0.02),    # declared band; swept in sensitivity curves
    n_min=1000,                            # per-cell floor with automatic rollup
)
cert.fit(X_train, y_train).calibrate(X_cal, y_reported)

lo, hi = cert.predict_set(X_new, alpha=0.10)      # contiguous KABCO intervals
lo, hi = cert.predict_set_risk(X_new, beta=0.05)  # severity-cost risk control
for c in cert.certificate(alpha=0.10):
    print(c.cell, c.n_cal, c.floor)               # per-cell slack budget
```

Runnable end to end on bundled synthetic data:

```python
from choir.datasets import load_demo
rows, y, cols = load_demo()   # FARS-schema synthetic sample, no PII, no download
```

`python examples/demo.py` runs in seconds. `pytest tests/` exercises the certificate
implementations on simulated data: marginal validity, observed-cell coverage,
banded-noise expansion, weighted-shift calculations, cost risk control, and composition.
These tests are software checks; they do not validate the assumptions for a new dataset.

## How it compares

Generic conformal toolkits (MAPIE, crepes, puncc) provide split and Mondrian machinery.
They are correct and attain marginal coverage under their stated conditions. What they do
not provide as a common interface for an ordinal, safety-critical target is contiguous
sets, a declared reporting-noise expansion, and a severity-cost risk certificate. The
table below is a bundled-demo benchmark at a nominal 0.90 level; it is empirical and is
not a cross-dataset guarantee.

| method | coverage | avg set size | contiguous sets | true-label guarantee | fatal-omission guarantee |
|--------|:--------:|:------------:|:---------------:|:--------------------:|:------------------------:|
| CHOIR  | 0.895    | 2.19         | yes (by construction) | yes            | yes                      |
| MAPIE  | 0.899    | 2.22         | 99%             | no                   | no                       |
| crepes | 0.899    | 2.22         | 99%             | no                   | no                       |

The coverage values are empirical results from this demo. CHOIR's sets are contiguous
intervals on the KABCO scale. Its true-label and fatal-omission statements apply only
when the corresponding compatibility and exact-fatality recording assumptions are
declared and plausible; the deployment-shift output is a diagnostic unless its stronger
covariate-shift conditions are established.

## What is guaranteed

The results and their proofs are in the companion paper prepared for submission to
Analytic Methods in Accident Research. Each is finite-sample and distribution-free only under its stated
conditions: coverage conditional on training-frozen observed final cells; true-label
coverage `1 - alpha - delta` under a declared banded compatibility map; weighted-shift
coverage under covariate shift with an independent, correctly specified ratio fit; and
severity-cost risk control, including the fatal-omission bound, under the declared noise
and exact-fatality premise. The shift discrepancy reported by CHOIR is not itself a
coverage guarantee. The composition theorem combines only compatible branches with an
additive, assumption-attributable slack budget.

## Citing

See `CITATION.cff`. Cite both the software (Zenodo DOI, on release) and the paper.

## License

MIT.
