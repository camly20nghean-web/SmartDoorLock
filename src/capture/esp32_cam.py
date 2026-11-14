"""HTTP MJPEG camera for the local demo."""
from contextlib import ExitStack
from ipaddress import IPv4Address, IPv4Network
from time import monotonic
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from .base import BaseCamera

LAN = tuple(IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"))


class ESP32CamStream(BaseCamera):
    def __init__(self, url=""):
        parts = urlsplit(url.strip())
        try:
            ip = IPv4Address(parts.hostname or "")
            port = parts.port
        except ValueError as error:
            raise ValueError("Nhập URL HTTP với địa chỉ IPv4 nội bộ của camera.") from error
        if (parts.scheme != "http" or not any(ip in net for net in LAN)
                or parts.username is not None or parts.password is not None or parts.fragment
                or port == 0):
            raise ValueError("Chỉ hỗ trợ URL HTTP của camera trên mạng nội bộ.")
        self.url = parts.geturl()
        self.error = ""
        self._resources = ExitStack()
        self._chunks = None
        self._buffer = bytearray()

    def connect(self):
        self.release()
        self.error = ""
        try:
            client = self._resources.enter_context(httpx.Client(timeout=3, trust_env=False, follow_redirects=False))
            response = self._resources.enter_context(client.stream("GET", self.url))
            response.raise_for_status()
            if not response.headers.get("content-type", "").lower().startswith("multipart/x-mixed-replace"):
                self.error = "Địa chỉ không trả luồng MJPEG. Hãy kiểm tra đường dẫn /stream."
                self.release()
                return False
            self._chunks = response.iter_raw()
            return True
        except (httpx.HTTPError, httpx.InvalidURL):
            self.error = "Không kết nối được ESP32-CAM. Kiểm tra Wi-Fi và URL luồng hình."
            self.release()
            return False

    def get_frame(self):
        if self._chunks is None:
            return False, None
        deadline = monotonic() + 4
        try:
            while monotonic() < deadline:
                start = self._buffer.find(b"\xff\xd8")
                if start >= 0:
                    end = self._buffer.find(b"\xff\xd9", start + 2)
                    if end >= 0:
                        jpeg = bytes(self._buffer[start:end + 2])
                        del self._buffer[:end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None and frame.size:
                            return True, frame
                        continue
                    del self._buffer[:start]
                elif len(self._buffer) > 1:
                    del self._buffer[:-1]
                if len(self._buffer) > 2_000_000:
                    break
                self._buffer.extend(next(self._chunks))
        except (StopIteration, httpx.HTTPError, cv2.error):
            pass
        self.error = "ESP32-CAM không gửi được khung hình hoặc đã mất luồng."
        self.release()
        return False, None

    def is_opened(self):
        return self._chunks is not None

    def release(self):
        self._chunks = None
        self._buffer.clear()
        self._resources.close()
