from contextlib import closing
import sqlite3

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np

from scripts import demo_face_palm_webcam as webcam
from scripts.prepare_account_admin_demo import prepare
from src.api import enrollment
from src.api.main import create_app
from src.database.connection import DatabaseConnection


def test_enroll_each_sample_and_preserve_other_templates(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite3"
    demo = webcam.FacePalmDemo(DatabaseConnection(str(source)))
    demo.users.create_user("P02", "Demo")
    shared = prepare(source, tmp_path / "shared.sqlite3")
    client = TestClient(create_app(shared), client=("127.0.0.1", 50000))
    capture_index = 0

    def capture(sample):
        dimension = webcam.DEFAULT_FEATURE_DIMENSION if sample == "FACE" else webcam.PALM_FEATURE_DIMENSION
        return np.eye(1, dimension, k=capture_index, dtype=np.float32)[0]

    monkeypatch.setattr(enrollment, "capture_sample", capture)

    def register(sample, replace=False):
        return client.post("/admin/enrollment", params={"user_id": "P02"},
                           json={"sample": sample, "replace": replace})

    def templates():
        with closing(sqlite3.connect(shared)) as conn:
            faces = dict(conn.execute("SELECT biometric_type, feature_data FROM biometric_templates"))
            palms = dict(conn.execute("SELECT hand_side, vector FROM demo_palm_templates"))
        return faces | palms

    assert register("FACE").status_code == 200
    assert set(templates()) == {"FACE"}
    assert register("LEFT").status_code == 200
    assert set(templates()) == {"FACE", "LEFT"}
    assert register("RIGHT").status_code == 200
    original = templates()
    assert register("LEFT").status_code == 409

    capture_index = 1
    assert register("LEFT", replace=True).status_code == 200
    updated = templates()
    assert updated["LEFT"] != original["LEFT"]
    assert updated["FACE"] == original["FACE"]
    assert updated["RIGHT"] == original["RIGHT"]

    def cancel(sample):
        raise HTTPException(422, "cancelled after 2 captures")

    monkeypatch.setattr(enrollment, "capture_sample", cancel)
    assert register("FACE", replace=True).status_code == 422
    assert templates() == updated
    user = client.get("/users").json()[0]
    assert user["has_face"] and user["has_left_palm"] and user["has_right_palm"]


def test_cancel_during_capture_releases_webcam(monkeypatch):
    calls = []

    class Camera:
        def __init__(self, camera_index):
            pass

        def connect(self):
            return True

        def release(self):
            calls.append("released")

    class FaceEngine:
        is_ready = True

    def cancel(*args):
        raise webcam.CaptureStopped("Cancelled at 2 of 5")

    monkeypatch.setattr(webcam, "WebcamCamera", Camera)
    monkeypatch.setattr(webcam, "FaceRecognitionEngine", FaceEngine)
    monkeypatch.setattr(webcam, "capture_face", cancel)
    monkeypatch.setattr(webcam.cv2, "destroyAllWindows", lambda: calls.append("closed"))
    import pytest

    with pytest.raises(HTTPException, match="Cancelled at 2 of 5"):
        enrollment.capture_sample("FACE")
    assert calls == ["released", "closed"]
