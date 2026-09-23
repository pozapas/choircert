# CHOIR

A certification layer for ordinal, safety-critical prediction. Wrap any model that
exports an ordinal conditional CDF and obtain finite-sample, distribution-free
certificates on contiguous prediction sets under declared sampling and reporting
assumptions.

Install with `pip install choircert` (import name `choir`). See the
[quickstart in the README](https://github.com/pozapas/choircert) and the
[Guarantees](guarantees.md) page for what is proved and under which assumptions.

Every guarantee is a statement about prediction-set coverage or expected risk under a
declared sampling assumption. The package estimates no causal quantities.
