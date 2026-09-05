from __future__ import annotations

import pytest

from icilval.spec import load_spec


@pytest.fixture(scope="session")
def spec():
    return load_spec()
