"""KABCO ordinal scale utilities.

y in {1..5}: 1=O (not injured), 2=C (possible), 3=B (non-incapacitating / suspected
minor), 4=A (incapacitating / suspected serious), 5=K (fatal).
"""

from __future__ import annotations

import numpy as np

KABCO_LEVELS = ("O", "C", "B", "A", "K")

# CRIS Prsn_Injry_Sev_ID strings (TxDOT), verified against the 2017-2025 extract.
CRIS_MAP = {
    "Not Injured": 1,
    "Possible Injury": 2,
    "Non-Incapacitating Injury": 3,
    "Incapacitating Injury": 4,
    "Killed": 5,
}


def kabco_to_int(labels, mapping: dict | None = None) -> np.ndarray:
    """Map string severity labels to 1..5; unmapped values (Unknown/None) become 0.

    Rows with 0 must be dropped by the caller and counted in the attrition table.
    """
    m = mapping or CRIS_MAP
    return np.array([m.get(v, 0) for v in labels], dtype=int)
