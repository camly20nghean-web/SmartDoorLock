"""Image and video capture abstraction layer."""
from .base import BaseCamera, FrameInfo
from .webcam import WebcamCamera
from .esp32_cam import ESP32CamStream

__all__ = ["BaseCamera", "FrameInfo", "WebcamCamera", "ESP32CamStream"]

