"""The committed parity fixtures (what the browser's remap/crop code is tested against) must match the Python code."""
from __future__ import annotations

import json

from scripts.make_frontend_fixtures import OUT, build


def test_frontend_parity_fixtures_are_current():
    assert json.loads(OUT.read_text(encoding="utf-8")) == build(), \
        "run: python backend/scripts/make_frontend_fixtures.py"
