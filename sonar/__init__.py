"""Reproducible reference implementation of the SONAR paper's core methods."""

from sonar.fusion import (
    COARSE_LABELS,
    FusionConfig,
    FusionResult,
    Prediction,
    fuse_emotions,
)

__all__ = [
    "COARSE_LABELS",
    "FusionConfig",
    "FusionResult",
    "Prediction",
    "fuse_emotions",
]
