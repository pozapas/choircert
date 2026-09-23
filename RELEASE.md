# Release notes

## 0.2.0 (2026-09-23)

This release updates the package metadata, documentation, and reproducibility materials.
Guarantee language is conditional on exchangeability, training-frozen observed
cells, declared reporting compatibility, and the relevant risk or shift assumptions. The
new-stratum discrepancy is documented as a diagnostic, not as a coverage guarantee. The
repository includes aggregate result artifacts and analysis scripts. No CRIS record-level
data are included.

The release passed 21 package tests, wheel and source-distribution builds, package metadata
checks, an isolated wheel installation, and the bundled demonstration. The public artifacts
contain the synthetic demonstration data only.

Previous 0.1.0 verification (2026-07-18) used a clean Python 3.12 environment.

The packaging guard remains important: `src/choir/datasets/demo_fars.csv`
is now git-tracked (root `.gitignore` `*.csv` PII guard was excluding it, so clones
and the sdist shipped without it), and `pyproject.toml` gained a `[tool.hatch.build]
artifacts` entry so both sdist and wheel carry the demo table.

Release services:

1. **GitHub.** Source and release artifacts are published at
   `https://github.com/pozapas/choircert/releases/tag/v0.2.0`. CI passed on Python
   3.10, 3.11, and 3.12.
2. **PyPI.** Version 0.2.0 is published at
   `https://pypi.org/project/choircert/0.2.0/`. The distribution name is
   `choircert`, while the import remains `choir`.
3. **Zenodo.** The archived release DOI is `10.5281/zenodo.22919393`.
   `CITATION.cff` retains concept DOI `10.5281/zenodo.21434172` as the
   latest-version pointer.
4. **Docs.** `mkdocs gh-deploy` publishes the site from `mkdocs.yml` and `docs/`.
