"""
Basic import tests to verify package structure.
"""
import sys


def test_urbanai_imports():
    """Test that main urbanai package can be imported."""
    import urbanai

    assert urbanai is not None
    assert hasattr(urbanai, "__version__")


def test_models_imports():
    """Test that models module can be imported."""
    from urbanai import models

    assert models is not None


def test_utils_imports():
    """Test that utils module can be imported."""
    from urbanai import utils

    assert utils is not None


def test_training_imports():
    """Test that training module can be imported."""
    from urbanai import training

    assert training is not None


def test_preprocessing_imports():
    """Test that preprocessing module can be imported."""
    from urbanai import preprocessing

    assert preprocessing is not None


def test_prediction_imports():
    """Test that prediction module can be imported."""
    from urbanai import prediction

    assert prediction is not None


def test_analysis_imports():
    """Test that analysis module can be imported."""
    from urbanai import analysis

    assert analysis is not None


def test_io_imports():
    """Test that io module can be imported."""
    from urbanai import io

    assert io is not None


def test_visualization_imports():
    """Test that visualization module can be imported."""
    from urbanai import visualization

    assert visualization is not None


def test_python_version():
    """Test that Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
