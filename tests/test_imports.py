"""
Basic import tests to verify package structure.
"""
import sys
import pytest


def test_urbanai_version():
    """Test that main urbanai package has version."""
    from urbanai import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_models_module_exists():
    """Test that models module exists."""
    from urbanai import models

    assert models is not None


def test_utils_module_exists():
    """Test that utils module exists."""
    from urbanai import utils

    assert utils is not None


def test_training_module_exists():
    """Test that training module exists."""
    from urbanai import training

    assert training is not None


def test_preprocessing_module_exists():
    """Test that preprocessing module exists."""
    from urbanai import preprocessing

    assert preprocessing is not None


def test_prediction_module_exists():
    """Test that prediction module exists."""
    from urbanai import prediction

    assert prediction is not None


def test_analysis_module_exists():
    """Test that analysis module exists."""
    from urbanai import analysis

    assert analysis is not None


def test_io_module_exists():
    """Test that io module exists."""
    from urbanai import io

    assert io is not None


def test_visualization_module_exists():
    """Test that visualization module exists."""
    from urbanai import visualization

    assert visualization is not None


def test_python_version():
    """Test that Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
