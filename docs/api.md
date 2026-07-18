# API

## `CertifiedOrdinal`

The high-level wrapper. Enforces the canonical composition order (condition, weight,
calibrate, expand, risk-adjust).

```python
CertifiedOrdinal(base, K=5, partition=None, noise=None, n_min=1000)
  .fit(X_train, y_train)
  .calibrate(X_cal, y_cal, strata=None)
  .predict_set(X, alpha=0.10, strata=None)        -> (lo, hi)
  .predict_set_risk(X, beta=0.05, kappa=None)      -> (lo, hi)
  .certificate(alpha=0.10)                         -> list[Certificate]
```

## Core functions

- `cumulative_score(cdf, y)` and `score_matrix(cdf)`: the ordinal score (Definition 1).
- `interval_sets(cdf, lam)`: contiguous interval endpoints (Lemma 1, Convention 1).
- `split_calibrate`, `mondrian_calibrate`, `weighted_quantile`: calibration.
- `NoiseModel` and `NoiseModel.kabco(...)`: the banded compatibility model.
- `crc_threshold`, `inflated_costs`: severity-cost risk control.

## Shift and monitoring

`choir.shift` provides `rollup_map`, `DensityRatioEstimator`, `tv_slack_lcb`, and
`DriftMonitor` (detection only, never correction).

## Crash batteries

`choir.crash` provides KABCO mappings and USDOT cost vectors; `choir.datasets.load_demo`
returns a small synthetic FARS-schema sample so examples run without external data.
