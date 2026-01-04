"""
Main ARK package

Provides high-level interfaces for KG construction, training, and inference.
"""

from . import kg
from . import training
from . import inference
from . import config

__version__ = "2.0.0"

__all__ = [
    "kg",
    "training",
    "inference",
    "config",
]
