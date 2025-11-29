"""
Models module for MAFS package.

Provides Adaptive Lasso feature selection with neural networks.
"""

from .feature_selector import (
    FeatureSelection,
    Classifier,
    AdaptiveLassoSelector
)

from .multi_head_selector import MultiHeadSelector

__all__ = [
    'FeatureSelection',
    'Classifier',
    'AdaptiveLassoSelector',
    'MultiHeadSelector',
]
