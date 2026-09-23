"""Small public demo data so examples run without CRIS in under a minute.

The bundled table is a deterministically generated SYNTHETIC sample that follows the
public FARS/KABCO schema (no real records, no PII). It exists only to make the README
and tutorials runnable anywhere. Swap in a real FARS extract by replacing demo_fars.csv
with the same columns.
"""

from choir.datasets.loader import DEMO_COLUMNS, load_demo

__all__ = ["DEMO_COLUMNS", "load_demo"]
