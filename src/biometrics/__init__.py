"""Biometric authentication abstraction layer (Face & Palmprint)."""
from .base import BaseBiometricExtractor, BiometricMatchResult
from .face import (
    DEFAULT_FEATURE_DIMENSION,
    DEFAULT_MODEL_VERSION,
    FaceQualityAssessment,
    FaceRecognitionEngine,
)
from .palm import PalmprintRecognitionEngine

__all__ = [
    "BaseBiometricExtractor",
    "BiometricMatchResult",
    "FaceRecognitionEngine",
    "FaceQualityAssessment",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_FEATURE_DIMENSION",
    "PalmprintRecognitionEngine",
]
