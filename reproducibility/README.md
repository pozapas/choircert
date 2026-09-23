# Reproducibility materials

This directory contains the analysis scripts, aggregate result artifacts, generated
tables, and illustration sources used to validate CHOIR on ordinal crash-injury data.
It is versioned with the software so that code, stored aggregate outputs, and generated
artifacts can be inspected together.

## Contents

- `experiments/` contains experiment drivers, aggregate result files, audit outputs,
  and figure-data exporters.
- `paper/figure_src/` contains exported figure data and source-specific builders.
- `paper/figures/` contains illustration generators and rendered outputs.
- `tables/` contains generated table files and the scripts that build them from the
  aggregate result files.
- `requirements.txt` lists the additional Python packages used by the analysis code.

The public package source, tests, examples, and generic-tool benchmark remain in the
repository root.

## Data boundary

Texas CRIS records are restricted and are not included. This directory contains no
record-level crash data, crash identifiers, person identifiers, or private repository
history. The stored Parquet, CSV, and JSON files contain aggregate experiment results or
figure-ready summaries. The `demo_fars.csv` file in the package source is synthetic and
contains 6,000 public-schema demonstration rows.

The CRIS-based experiment drivers require an authorized local analysis table. Without
that restricted input, users can inspect the scripts, audit the stored aggregate results,
regenerate tables and illustrations from those results, and run the package tests and
synthetic demonstration.

## Environment

Install the package and analysis dependencies from the repository root.

```text
python -m pip install -e ".[econ,maps,torch]"
python -m pip install -r reproducibility/requirements.txt
```

Run the package tests with `pytest tests/`. Individual experiment and artifact scripts
record their declared seeds and input paths in source.
