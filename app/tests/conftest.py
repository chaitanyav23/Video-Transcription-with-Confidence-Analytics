"""Pytest Configuration and Fixtures for App Test Suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pytest
from fastapi.testclient import TestClient

from backend.database.db import init_db
from backend.main import create_app


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database tables exist before test session starts."""
    init_db()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    app = create_app()
    return TestClient(app)
