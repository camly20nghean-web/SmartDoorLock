"""Single-modality enrollment through the local webcam demo."""
from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel


class Enrollment(BaseModel):
    sample: Literal["FACE", "LEFT", "RIGHT"]
    replace: bool = False


def check_target(conn, user_id, data):
    row = conn.execute("SELECT status FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Account not found.")
    if row[0] != "ACTIVE":
        raise HTTPException(409, "Enrollment requires an active account.")
    if data.sample == "FACE":
        existing = conn.execute(
            "SELECT 1 FROM biometric_templates WHERE user_id = ? AND biometric_type = 'FACE'", (user_id,),
        ).fetchone()
    else:
        existing = conn.execute(
            "SELECT 1 FROM demo_palm_templates WHERE user_id = ? AND hand_side = ?", (user_id, data.sample),
        ).fetchone()
    if existing and not data.replace:
        raise HTTPException(409, "This sample already exists. Confirm replacement and try again.")


def capture_sample(sample):
    from scripts import demo_face_palm_webcam as webcam

    camera = webcam.WebcamCamera(camera_index=0)
    try:
        if not camera.connect():
            raise HTTPException(409, "Cannot open webcam. Close other apps using it and try again.")
        if sample == "FACE":
            engine = webcam.FaceRecognitionEngine()
            if not engine.is_ready:
                raise HTTPException(503, "Face models are unavailable.")
            vectors = webcam.capture_face(camera, engine, 20.0, webcam.ENROLL_SAMPLES, time.monotonic() + 90)
            dimension = webcam.DEFAULT_FEATURE_DIMENSION
        else:
            with webcam.PalmprintRecognitionEngine() as engine:
                vectors, hand = webcam.capture_palm(
                    camera, engine, 2.0, webcam.ENROLL_SAMPLES, time.monotonic() + 90, sample,
                )
            if hand != sample:
                raise HTTPException(422, "The captured hand does not match the requested side.")
            dimension = webcam.PALM_FEATURE_DIMENSION
        if len(vectors) != webcam.ENROLL_SAMPLES:
            raise HTTPException(422, "Not enough accepted captures.")
        return webcam.unit_vector(webcam.np.mean(vectors, axis=0), dimension)
    except (webcam.CaptureStopped, OSError, ValueError, RuntimeError, webcam.cv2.error) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        camera.release()
        webcam.cv2.destroyAllWindows()


def enroll(path: Path, user_id, data):
    from src.api.accounts import account_transaction, has_admin_audit

    if not CAMERA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another enrollment is using the webcam.")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            if not has_admin_audit(conn):
                raise HTTPException(403, "Enrollment requires the shared demo database.")
            check_target(conn, user_id, data)
        vector = capture_sample(data.sample)
        from scripts import demo_face_palm_webcam as webcam

        with account_transaction(path) as conn:
            check_target(conn, user_id, data)
            if data.sample == "FACE":
                webcam.store_face(conn, user_id, vector, webcam.get_utc_now_iso())
            else:
                webcam.store_palm(conn, user_id, data.sample, vector)
        return {"user_id": user_id, "sample": data.sample}
    except sqlite3.Error as error:
        raise HTTPException(503, "Cannot access the demo database.") from error
    finally:
        CAMERA_LOCK.release()


CAMERA_LOCK = Lock()
