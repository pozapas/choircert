# CHOIR

[![PyPI version](https://img.shields.io/pypi/v/choircert.svg)](https://pypi.org/project/choircert/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21434172.svg)](https://doi.org/10.5281/zenodo.21434172)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A certification layer for ordinal, safety-critical prediction.** Distribution name
on PyPI: `choircert`; import name: `choir`.

Wrap any ordinal severity model and obtain finite-sample, distribution-free guarantees:
contiguous ordinal prediction sets with marginal and group-conditional coverage
(heterogeneity classes, jurisdiction-year strata), coverage on the *true* label under a
declared banded reporting-noise assumption, deployment-shift transfer certificates, and
severity-weighted risk control including a fatal-omission guarantee, all composable with
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

`python examples/demo.py` runs in seconds. `pytest tests/` reproduces every guarantee
on simulated data: marginal validity, class-conditional coverage, banded-noise transfer,
group-weighted shift, weighted transfer, cost risk control, and composition.

## How it compares

Generic conformal toolkits (MAPIE, crepes, puncc) provide split and Mondrian machinery.
They are correct and attain marginal coverage. What they do not provide for an ordinal,
safety-critical target is contiguity, a guarantee on the true (noisy) label, and a
fatal-omission guarantee. The table below is produced by `benchmarks/vs_mapie_crepes.py`
on the bundled demo at a nominal 0.90 level.

| method | coverage | avg set size | contiguous sets | true-label guarantee | fatal-omission guarantee |
|--------|:--------:|:------------:|:---------------:|:--------------------:|:------------------------:|
| CHOIR  | 0.895    | 2.19         | yes (by construction) | yes            | yes                      |
| MAPIE  | 0.899    | 2.22         | 99%             | no                   | no                       |
| crepes | 0.899    | 2.22         | 99%             | no                   | no                       |

The coverage is deliberately the same; the guarantee is the same theorem. The difference
is that CHOIR's sets are always contiguous intervals on the KABCO scale, and that CHOIR
additionally transfers coverage to the true injury under a declared band and bounds the
probability of excluding a true fatality. On this demo, the generic toolkits return a
non-contiguous set about one percent of the time, which is not an operationally meaningful
"B or worse" statement.

## What is guaranteed

The four guarantees and their proofs are in the companion paper (Transportation Research
Part B, under review). Each is finite-sample and distribution-free in the wrapped model:
class-conditional coverage (validity for any partition, oracle efficiency for a good one),
true-label coverage `1 - alpha - delta` under a banded compatibility assumption,
deployment transfer certificates reported as (nominal level, slack lower confidence bound,
parametric slack estimate), and severity-cost risk control including the fatal-omission
bound. The composition theorem combines them with an additive, assumption-attributable
slack budget.

## Citing

See `CITATION.cff`. Cite both the software (Zenodo DOI, on release) and the paper.

## License

MIT.
