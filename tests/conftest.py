"""
Pytest configuration and fixtures for UrbanAI tests.
"""
import pytest


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "preprocessing": {
            "start_year": 2000,
            "end_year": 2020,
            "interval": 2,
        },
        "training": {
            "epochs": 10,
            "batch_size": 4,
            "learning_rate": 0.001,
        },
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
