"""Abstract Base Class and data models for biometric feature extractors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class BiometricMatchResult:
    """Authentication verification result."""
    is_matched: bool
    similarity_score: float
    user_id: Optional[str] = None
    confidence_level: str = "LOW"
    message: str = ""


class BaseBiometricExtractor(ABC):
    """
    Abstract interface for biometric authentication engines (Face & Palmprint).
    
    Strictly forbids fake returns. Any model not yet loaded or implemented
    must raise NotImplementedError rather than returning dummy results.
    """

    @abstractmethod
    def detect_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Locate and crop Region of Interest (ROI) from raw camera frame.
        
        Args:
            image (np.ndarray): Input BGR frame.
            
        Returns:
            Optional[np.ndarray]: Cropped, aligned ROI image or None if not detected.
        """
        pass

    @abstractmethod
    def extract_embedding(self, roi_image: np.ndarray) -> np.ndarray:
        """
        Extract numerical biometric feature vector (embedding).
        
        Args:
            roi_image (np.ndarray): Cropped ROI image.
            
        Returns:
            np.ndarray: Normalized 1D feature vector.
        """
        pass

    @abstractmethod
    def verify(self, probe_image: np.ndarray, enrolled_embedding: np.ndarray) -> BiometricMatchResult:
        """
        Compare captured image against registered user embedding.
        
        Args:
            probe_image (np.ndarray): Live camera frame.
            enrolled_embedding (np.ndarray): Registered feature vector.
            
        Returns:
            BiometricMatchResult: Verification outcome with similarity score.
        """
        pass

