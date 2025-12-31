# src/urbanai/__init__.py
"""
UrbanAI - Deep Learning Framework for Spatiotemporal Urban Heat Prediction
"""

from urbanai.__version__ import __version__
from urbanai.pipeline import UrbanAIPipeline

# Make submodules available
from urbanai import preprocessing, models, training, prediction, analysis

__all__ = [
    "__version__",
    "UrbanAIPipeline",
    "preprocessing",
    "models",
    "training",
    "prediction",
    "analysis",
]
