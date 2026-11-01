from __future__ import annotations

import os

os.environ.setdefault("ROUGHCUT_AGENTS", "0")  # the API tests cover the base decision set; test_agents.py the full one

import pytest
from fastapi.testclient import TestClient

from app.appctx import AppContext
from app.main import create_app
from tests.apihelpers import fake_ingest


@pytest.fixture
def ctx(tmp_path):
    c = AppContext(tmp_path / "data", ingest_fn=fake_ingest)
    yield c
    c.close()


@pytest.fixture
def client(ctx):
    with TestClient(create_app(ctx)) as c:
        yield c
