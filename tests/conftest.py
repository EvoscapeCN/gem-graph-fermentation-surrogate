import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pytest  # noqa: E402

GEM = os.path.join(REPO, "data", "gem", "iyli21_modified.xml")


def pytest_collection_modifyitems(config, items):
    """Tests that read the GEM are skipped until scripts/build_gem.py has built it (see README)."""
    if os.path.exists(GEM):
        return
    skip = pytest.mark.skip(reason="data/gem/iyli21_modified.xml not built; run scripts/build_gem.py (see README)")
    for item in items:
        if os.path.basename(str(item.fspath)) in ("test_model.py", "test_smoke.py"):
            item.add_marker(skip)
