"""Shared test setup.

* Points MAGPIE_DATA_DIR at a throwaway directory BEFORE any magpie module is
  imported, so no test can touch a real data/ directory.
* Tests marked `network` hit MAST / Kepler archives and are skipped unless
  MAGPIE_NETWORK_TESTS=1.
"""
import atexit
import os
import shutil
import tempfile

_DATA = tempfile.mkdtemp(prefix="magpie-test-data-")
atexit.register(shutil.rmtree, _DATA, ignore_errors=True)
os.environ["MAGPIE_DATA_DIR"] = _DATA
os.environ.pop("MAGPIE_DB", None)

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: needs internet access (MAST/Kepler archives)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MAGPIE_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set MAGPIE_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
