# CHOIR

[![PyPI version](https://img.shields.io/pypi/v/choircert.svg)](https://pypi.org/project/choircert/)
[![CI](https://github.com/pozapas/choircert/actions/workflows/ci.yml/badge.svg)](https://github.com/pozapas/choircert/actions/workflows/ci.yml)
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

MAPIE and crepes provide general classification conformal workflows. The comparison below
uses one fixed ordinal task. It is not a claim of general superiority. All methods use the
same 6,000-row synthetic dataset, 50/25/25 training-calibration-test split, 1,500-record
test set, histogram gradient boosting base model, and random seed `20260704`. The nominal
coverage is 0.90. The recorded environment uses MAPIE 1.5.0, crepes 0.9.1,
scikit-learn 1.9.1, and NumPy 2.5.1.

| method | observed-label coverage | mean set size | empirically contiguous | declared-map transfer in this workflow | fatal-omission control in this workflow |
|--------|:-----------------------:|:-------------:|:----------------------:|:--------------------------------------:|:---------------------------------------:|
| CHOIR  | 0.8953 | 2.1920 | 100% by construction | included under the compatibility map | included under the fatality premise |
| MAPIE  | 0.8993 | 2.2187 | 99.07% | not included | not included |
| crepes | 0.8993 | 2.2187 | 99.07% | not included | not included |

The coverage, size, and contiguity columns are empirical results from the demonstration.
The last two columns describe the branches included in this benchmark workflow. They do
not imply that another library cannot be extended with additional user code. CHOIR's
declared-map transfer and fatal-omission control apply only when the corresponding
compatibility and exact-fatality recording assumptions are declared and plausible. The
deployment-shift output is a diagnostic unless its stronger covariate-shift conditions
are established.

Reproduce the comparison with
`uv run --extra benchmarks python benchmarks/vs_mapie_crepes.py`. The
[benchmark script](benchmarks/vs_mapie_crepes.py) and
[raw results](benchmarks/results/generic_tool_benchmark.csv) are stored in the repository.

## What is guaranteed

The formal statements and operating conditions are summarized in
[`docs/guarantees.md`](docs/guarantees.md). Each result is finite-sample and
distribution-free only under its stated conditions: coverage conditional on training-frozen observed final cells; true-label
coverage `1 - alpha - delta` under a declared banded compatibility map; weighted-shift
coverage under covariate shift with an independent, correctly specified ratio fit; and
severity-cost risk control, including the fatal-omission bound, under the declared noise
and exact-fatality premise. The shift discrepancy reported by CHOIR is not itself a
coverage guarantee. The composition theorem combines only compatible branches with an
additive, assumption-attributable slack budget.

## Reproducibility materials

The [`reproducibility`](reproducibility) directory contains experiment scripts,
aggregate result artifacts, table generators, and illustration sources. It excludes
Texas CRIS record-level data, crash identifiers, person identifiers, and private
repository history. The bundled `demo_fars.csv` file is a synthetic public-schema
demonstration table with 6,000 rows.

See [`reproducibility/README.md`](reproducibility/README.md) for the directory
inventory, environment requirements, and data-access boundary.

## Citing

See `CITATION.cff` for the software citation and version-independent Zenodo DOI.

## License

MIT.
