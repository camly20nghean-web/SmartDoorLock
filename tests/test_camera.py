"""Unit tests for camera abstraction and webcam implementation without physical hardware."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.capture.base import BaseCamera, FrameInfo
from src.capture.webcam import WebcamCamera
from src.capture.esp32_cam import ESP32CamStream


class TestFrameInfo:
    """Test suite for FrameInfo metadata model."""

    def test_frame_info_from_valid_numpy_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = FrameInfo.from_frame(dummy_frame)

        assert info.width == 640
        assert info.height == 480
        assert info.channels == 3
        assert info.timestamp is not None

    def test_frame_info_grayscale_frame(self):
        dummy_gray = np.zeros((240, 320), dtype=np.uint8)
        info = FrameInfo.from_frame(dummy_gray)

        assert info.width == 320
        assert info.height == 240
        assert info.channels == 1


class TestWebcamCamera:
    """Unit tests for WebcamCamera lifecycle, frame validation, and error recovery."""

    @patch("cv2.VideoCapture")
    def test_connect_success(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        mock_cv_capture.return_value = mock_cap_instance

        cam = WebcamCamera(camera_index=0, width=640, height=480)
        connected = cam.connect()

        assert connected is True
        assert cam.is_opened() is True
        mock_cap_instance.set.assert_any_call(3, 640)
        mock_cap_instance.set.assert_any_call(4, 480)

    @patch("cv2.VideoCapture")
    def test_connect_failure_unavailable_device(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cv_capture.return_value = mock_cap_instance

        cam = WebcamCamera(camera_index=99)
        connected = cam.connect()

        assert connected is False
        assert cam.is_opened() is False

    @patch("cv2.VideoCapture")
    def test_get_frame_success(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_cap_instance.read.return_value = (True, dummy_frame)
        mock_cv_capture.return_value = mock_cap_instance

        cam = WebcamCamera(camera_index=0)
        cam.connect()
        ret, frame = cam.get_frame()

        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_get_frame_when_not_connected(self):
        cam = WebcamCamera(camera_index=0)
        ret, frame = cam.get_frame()

        assert ret is False
        assert frame is None

    @patch("cv2.VideoCapture")
    def test_get_frame_corrupted_data(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        # Frame with empty array
        mock_cap_instance.read.return_value = (True, np.array([]))
        mock_cv_capture.return_value = mock_cap_instance

        cam = WebcamCamera(camera_index=0)
        cam.connect()
        ret, frame = cam.get_frame()

        assert ret is False
        assert frame is None

    @patch("cv2.VideoCapture")
    def test_connect_handles_exception_gracefully(self, mock_cv_capture):
        mock_cv_capture.side_effect = RuntimeError("Driver fatal crash")

        cam = WebcamCamera(camera_index=0)
        connected = cam.connect()

        assert connected is False
        assert cam.is_opened() is False

    @patch("cv2.VideoCapture")
    def test_get_frame_handles_read_exception(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = RuntimeError("DirectShow buffer error")
        mock_cv_capture.return_value = mock_cap_instance

        cam = WebcamCamera(camera_index=0)
        cam.connect()
        ret, frame = cam.get_frame()

        assert ret is False
        assert frame is None

    def test_double_release_is_idempotent(self):
        cam = WebcamCamera(camera_index=0)
        cam.release()
        cam.release()  # Calling release multiple times should be safe and idempotent
        assert cam.is_opened() is False

    @patch("cv2.VideoCapture")
    def test_context_manager_auto_release(self, mock_cv_capture):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cv_capture.return_value = mock_cap_instance

        with WebcamCamera(camera_index=0) as cam:
            assert cam.is_opened() is True

        mock_cap_instance.release.assert_called_once()
        assert cam.is_opened() is False


class TestESP32CamStream:
    def test_local_url_required(self):
        for url in ("https://192.168.1.5/stream", "http://8.8.8.8/stream", "http://camera.local/stream"):
            with pytest.raises(ValueError):
                ESP32CamStream(url)

    def test_mjpeg_decodes_split_jpeg_and_releases_on_disconnect(self, monkeypatch):
        import cv2
        import httpx

        _, encoded = cv2.imencode(".jpg", np.zeros((16, 24, 3), dtype=np.uint8))
        jpeg = encoded.tobytes()
        closed = []

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg[:1]
                yield jpeg[1:-1]
                yield jpeg[-1:]

            def close(self):
                closed.append(True)

        real_client = httpx.Client
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"}, stream=Stream(),
        ))
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
        camera = ESP32CamStream("http://192.168.1.5:81/stream")
        assert camera.connect()
        ok, frame = camera.get_frame()
        assert ok and frame.shape == (16, 24, 3)
        assert camera.get_frame() == (False, None)
        assert not camera.is_opened()
        assert closed == [True]
