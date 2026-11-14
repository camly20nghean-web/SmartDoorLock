from contextlib import nullcontext

from fastapi.testclient import TestClient
import numpy as np
import pytest

from scripts import demo_face_palm_webcam as webcam
from scripts.prepare_account_admin_demo import prepare
from src.api.enrollment import CAMERA_LOCK
from src.api.main import create_app
from src.hardware.mock_lock import MockDoorLock


ORIGINAL_CAPTURE_PALM = webcam.capture_palm


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    db = webcam.DatabaseConnection(str(tmp_path / "original.sqlite3"))
    demo = webcam.FacePalmDemo(db)
    face = np.eye(1, webcam.DEFAULT_FEATURE_DIMENSION, dtype=np.float32)[0]
    palm = np.eye(1, webcam.PALM_FEATURE_DIMENSION, dtype=np.float32)[0]
    webcam.save_enrollment(db, "P02", "RIGHT", [face], [palm])
    shared = prepare(tmp_path / "original.sqlite3", tmp_path / "shared.sqlite3")

    class Camera:
        def __init__(self, camera_index):
            pass

        def connect(self):
            return True

        def release(self):
            pass

    class FaceEngine:
        is_ready = True

    monkeypatch.setattr(webcam, "WebcamCamera", Camera)
    monkeypatch.setattr(webcam.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(webcam, "FaceRecognitionEngine", FaceEngine)
    monkeypatch.setattr(webcam, "PalmprintRecognitionEngine", lambda: nullcontext())
    monkeypatch.setattr(webcam, "capture_face", lambda *args: [face])
    monkeypatch.setattr(webcam, "capture_palm", lambda *args: ([palm], "RIGHT"))
    calls = []
    original_unlock = MockDoorLock.unlock

    def unlock(lock):
        calls.append(True)
        return original_unlock(lock)

    monkeypatch.setattr(MockDoorLock, "unlock", unlock)
    return TestClient(create_app(shared), client=("127.0.0.1", 50000)), calls


@pytest.mark.parametrize("policy,method,side", [
    ("FACE_ONLY", "FACE", None), ("PALM_ONLY", "PALM", "RIGHT"), ("FACE_AND_PALM", "DUAL", "RIGHT"),
])
def test_dashboard_unlock_modes(demo_client, policy, method, side):
    client, calls = demo_client
    response = client.post("/admin/verify", params={"policy": policy})
    assert response.status_code == 200
    result = response.json()
    assert (result["user_id"], result["hand_side"], result["decision"], result["actuation_status"]) == (
        "P02", side, "GRANTED", "ACTUATED",
    )
    row, = client.get("/access-logs").json()["items"]
    assert (row["request_id"], row["auth_method"], row["hand_side"]) == (result["request_id"], method, side)
    assert calls == [True]


@pytest.mark.parametrize("failure", ["inactive", "wrong-hand", "cancelled", "expired", "busy"])
def test_dashboard_unlock_failures(demo_client, monkeypatch, failure):
    client, calls = demo_client
    if failure == "inactive":
        response = client.post("/admin/user-status", params={"user_id": "P02"},
                               json={"action": "DEACTIVATE", "confirmation": "P02"})
        assert response.status_code == 200
    elif failure == "wrong-hand":
        captured = webcam.capture_palm
        monkeypatch.setattr(webcam, "capture_palm", lambda *args: (captured(*args)[0], "LEFT"))
    elif failure in ("cancelled", "expired"):
        def stop(*args):
            raise webcam.CaptureStopped(failure, "RIGHT")
        monkeypatch.setattr(webcam, "capture_palm", stop)

    if failure == "busy":
        with CAMERA_LOCK:
            response = client.post("/admin/verify", params={"policy": "FACE_AND_PALM"})
        assert response.status_code == 409
        assert "busy" in response.json()["detail"]
    else:
        response = client.post("/admin/verify", params={"policy": "FACE_AND_PALM"})
        assert response.status_code == 200
        result = response.json()
        assert (result["decision"], result["actuation_status"]) == ("DENIED", "NOT_ACTUATED")
        if failure in ("cancelled", "expired"):
            assert result["capture_error"] == failure
    assert calls == []

def test_expired_session_is_audited_once(demo_client, monkeypatch):
    client, unlock_calls = demo_client
    from scripts import demo_face_palm_unlock as unlock_demo

    now = [100.0]
    from types import SimpleNamespace

    monkeypatch.setattr(unlock_demo, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(webcam, "time", SimpleNamespace(monotonic=lambda: now[0]))
    issue_session = webcam.FacePalmDemo.issue_session
    authenticate = webcam.FacePalmDemo.authenticate
    replay = []

    def issue_then_expire(demo, ttl_seconds):
        session_id = issue_session(demo, ttl_seconds)
        now[0] += ttl_seconds
        return session_id

    def authenticate_then_replay(demo, *args, **kwargs):
        result = authenticate(demo, *args, **kwargs)
        replay.append(authenticate(demo, *args, **kwargs))
        return result

    monkeypatch.setattr(webcam.FacePalmDemo, "issue_session", issue_then_expire)
    monkeypatch.setattr(webcam.FacePalmDemo, "authenticate", authenticate_then_replay)

    response = client.post("/admin/verify", params={"policy": "FACE_AND_PALM"})
    assert response.status_code == 200
    result = response.json()
    assert (result["decision"], result["actuation_status"], result["reason"]) == (
        "DENIED", "NOT_ACTUATED", "Session expired.",
    )
    row, = client.get("/access-logs").json()["items"]
    assert row["request_id"] == result["request_id"]
    assert (row["auth_method"], row["hand_side"], row["decision"], row["actuation_status"], row["reason"]) == (
        "DUAL", "RIGHT", "DENIED", "NOT_ACTUATED", "Session expired.",
    )
    assert replay[0].reason == "Unknown or already used session."
    assert unlock_calls == []

def test_esp32_disconnect_does_not_unlock_and_webcam_can_retry(demo_client, monkeypatch):
    from src.api import unlock

    client, calls = demo_client
    released = []

    class Camera:
        error = "ESP32-CAM đã mất luồng."

        def connect(self):
            return True

        def release(self):
            released.append(True)

        def get_frame(self):
            return False, None

    with monkeypatch.context() as remote:
        remote.setattr(unlock, "network_camera", lambda selection: Camera())
        remote.setattr(webcam, "capture_palm", ORIGINAL_CAPTURE_PALM)
        result = client.post("/admin/verify", params={"policy": "FACE_AND_PALM"},
                             json={"source": "esp32", "url": "http://192.168.1.5:81/stream"})
        assert result.status_code == 200
        assert result.json()["actuation_status"] == "NOT_ACTUATED"
        assert "mất luồng" in result.json()["capture_error"]
        assert calls == [] and released == [True]

    assert client.post("/admin/verify", params={"policy": "FACE_AND_PALM"}).json()["actuation_status"] == "ACTUATED"
