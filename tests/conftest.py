"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import Base


@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create test database session."""
    Session = sessionmaker(bind=test_db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "float_wmo": "2902123",
        "cycle_number": 42,
        "timestamp": "2023-03-15T12:00:00",
        "latitude": 2.1,
        "longitude": 45.2,
        "depths": [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        "temperatures": [28.3, 28.1, 27.8, 26.5, 24.2, 20.1, 15.3, 10.2, 5.1],
        "salinities": [34.1, 34.2, 34.3, 34.5, 34.8, 35.0, 35.1, 35.0, 34.9],
        "summary": "Float 2902123: 2023-03-15, location (2.1N, 45.2E), max depth 2000m. Surface temp 28.3°C.",
    }


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
