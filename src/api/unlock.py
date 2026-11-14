"""One local webcam attempt with the demo lock."""
from contextlib import ExitStack, closing
import base64
import sqlite3
from typing import Literal

import cv2
from pydantic import BaseModel, Field

from fastapi import HTTPException

from src.api.enrollment import CAMERA_LOCK
from src.capture.esp32_cam import ESP32CamStream
from src.database.models import AuthPolicy


class CameraInput(BaseModel):
    source: Literal["webcam", "esp32"] = "webcam"
    url: str = Field(default="", max_length=512)


def network_camera(selection):
    try:
        return ESP32CamStream(selection.url)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def preview_camera(selection: CameraInput):
    if selection.source != "esp32":
        raise HTTPException(422, "Chọn ESP32-CAM để kiểm tra luồng hình.")
    if not CAMERA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Camera đang được dùng.")
    try:
        camera = network_camera(selection)
        with ExitStack() as stack:
            stack.callback(camera.release)
            if not camera.connect():
                raise HTTPException(502, camera.error)
            ok, frame = camera.get_frame()
            if not ok:
                raise HTTPException(502, camera.error)
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                raise HTTPException(502, "Không đọc được ảnh camera.")
            return {"jpeg": base64.b64encode(jpeg.tobytes()).decode("ascii")}
    finally:
        CAMERA_LOCK.release()


def verify(path, policy: AuthPolicy, selection: CameraInput | None = None):
    from scripts import demo_face_palm_webcam as webcam
    from src.api.accounts import has_admin_audit

    if not CAMERA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Webcam is busy with enrollment or verification.")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            if not has_admin_audit(conn):
                raise HTTPException(403, "Verification requires the shared demo database.")
        with ExitStack() as stack:
            demo = webcam.FacePalmDemo(webcam.DatabaseConnection(str(path)), face_threshold=webcam.FACE_THRESHOLD)
            remote = selection is not None and selection.source == "esp32"
            camera = network_camera(selection) if remote else webcam.WebcamCamera(camera_index=0)
            stack.callback(webcam.cv2.destroyAllWindows)
            stack.callback(camera.release)
            if not camera.connect():
                if remote:
                    raise HTTPException(502, camera.error)
                raise HTTPException(409, "Cannot open webcam. Close other apps using it and try again.")
            face = None
            palm = None
            if policy != AuthPolicy.PALM_ONLY:
                demo.face_templates = webcam.load_face_templates(demo.db)
                face = webcam.FaceRecognitionEngine()
                if not face.is_ready:
                    raise HTTPException(503, "Face models are unavailable.")
            if policy != AuthPolicy.FACE_ONLY:
                palm = stack.enter_context(webcam.PalmprintRecognitionEngine())
            result, hand, capture_error = webcam.verify_once(
                demo, policy, camera, face, palm,
                webcam.DEFAULT_FACE_MIN_SHARPNESS if policy == AuthPolicy.FACE_ONLY else 20.0, 2.0,
            )
            return {
                "request_id": result.request_id,
                "user_id": result.user_id,
                "hand_side": hand,
                "decision": result.decision,
                "actuation_status": result.actuation_status,
                "reason": result.reason,
                "capture_error": capture_error,
            }
    except (sqlite3.Error, OSError, RuntimeError, ValueError, webcam.cv2.error) as error:
        raise HTTPException(503, str(error)) from error
    finally:
        CAMERA_LOCK.release()
