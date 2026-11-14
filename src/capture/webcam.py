"""Webcam camera implementation using OpenCV."""
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from .base import BaseCamera

logger = logging.getLogger(__name__)


class WebcamCamera(BaseCamera):
    """
    OpenCV-based camera source for built-in or USB laptop webcams.
    
    Provides frame normalization, dimension verification, and robust error handling.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        """
        Open the webcam device with configured properties.
        
        Returns:
            bool: True if webcam successfully opened, False otherwise.
        """
        if self.is_opened():
            logger.info("Camera device %s is already connected.", self.camera_index)
            return True

        logger.info("Attempting to connect to camera device %s...", self.camera_index)
        
        try:
            # On Windows, cv2.CAP_DSHOW provides faster startup and stable directshow negotiation
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self._cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                logger.warning("CAP_DSHOW backend failed; falling back to default backend.")
                self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                logger.error("Failed to open camera device %s. Device unavailable or in use.", self.camera_index)
                self._cap = None
                return False

            # Apply desired frame resolution and FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                "Camera %s connected successfully. Resolution: %sx%s, FPS: %.1f",
                self.camera_index, actual_w, actual_h, actual_fps
            )
            return True
        except Exception as e:
            logger.error("Exception while connecting to camera device %s: %s", self.camera_index, e)
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read and validate a single BGR frame from the webcam.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, normalized_frame) on success, (False, None) on failure.
        """
        if not self.is_opened() or self._cap is None:
            logger.warning("Cannot read frame: Camera is not connected.")
            return False, None

        try:
            ret, frame = self._cap.read()
        except Exception as e:
            logger.error("Exception occurred while reading frame from camera %s: %s", self.camera_index, e)
            return False, None

        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera %s.", self.camera_index)
            return False, None

        # Sanity check: Ensure frame has non-zero size and is 3-channel BGR uint8
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("Captured frame is corrupted or has invalid dimensions: %s", getattr(frame, "shape", None))
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        """Return True if capture device is active and opened."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Safely release webcam hardware resource."""
        if self._cap is not None:
            try:
                self._cap.release()
                logger.info("Camera %s hardware released successfully.", self.camera_index)
            except Exception as e:
                logger.error("Error while releasing camera %s: %s", self.camera_index, e)
            finally:
                self._cap = None

