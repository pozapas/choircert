"""Release metadata checks for the local publication candidate."""

from importlib.metadata import version

import choir


def test_version_is_synced():
    assert choir.__version__ == "0.2.0"
    assert version("choircert") == "0.2.0"


def test_bundled_demo_has_no_record_level_cris_fields():
    from choir.datasets import DEMO_COLUMNS

    # The package ships only a synthetic FARS-schema demonstration table.
    assert "case_id" not in DEMO_COLUMNS
    assert "record_id" not in DEMO_COLUMNS
    assert "cris_id" not in DEMO_COLUMNS
