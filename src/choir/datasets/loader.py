"""Loader for the bundled synthetic FARS-schema demo table."""

from __future__ import annotations

import csv
from importlib import resources

# FARS-style public schema subset (no PII). y is KABCO 1..5.
DEMO_COLUMNS = ["age", "speed_limit", "restraint", "light", "vehicle_age",
                "rural", "body_class", "y"]

_CONT = {"age", "speed_limit", "vehicle_age"}


def load_demo():
    """Return (X, y, columns) for the bundled demo.

    X is a list of dict rows (dependency-light; convert to a frame as needed);
    y is a list of KABCO integers 1..5. For a numpy design matrix, one-hot the
    categorical columns downstream, or use the tutorial's encoder.
    """
    rows, y = [], []
    with resources.files("choir.datasets").joinpath("demo_fars.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            y.append(int(r.pop("y")))
            for c in _CONT:
                r[c] = float(r[c])
            rows.append(r)
    feat_cols = [c for c in DEMO_COLUMNS if c != "y"]
    return rows, y, feat_cols
