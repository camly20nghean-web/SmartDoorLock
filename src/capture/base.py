"""Abstract Base Class and data models for video/image capture sources."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FrameInfo:
    """Metadata describing a captured video frame."""
    width: int
    height: int
    channels: int
    timestamp: datetime

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> "FrameInfo":
        """Construct FrameInfo from a numpy frame array."""
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1
        return cls(width=w, height=h, channels=c, timestamp=datetime.now())


class BaseCamera(ABC):
    """
    Abstract Base Class for all camera input sources (Webcam, ESP32-S3-CAM, etc.).
    
    Guarantees that downstream biometric processing modules receive standardized
    NumPy BGR frames, completely decoupled from the physical transport medium.
    Supports Python context manager for deterministic resource disposal.
    """

    def __enter__(self) -> "BaseCamera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the camera device or network stream.
        
        Returns:
            bool: True if connection succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture and return a single frame.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - status (bool): True if frame was successfully captured and valid.
                - frame (Optional[np.ndarray]): Standard BGR image array, or None if failed.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera is currently connected and ready.
        
        Returns:
            bool: True if active, False otherwise.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying hardware, video device, or network connections."""
        pass

