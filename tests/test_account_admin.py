from contextlib import closing
import sqlite3

from fastapi.testclient import TestClient
import numpy as np
import pytest

from scripts.demo_face_palm_unlock import FacePalmDemo
from scripts.prepare_account_admin_demo import prepare
from src.api.main import create_app
from src.biometrics.palm import PALM_FEATURE_DIMENSION
from src.database.connection import DatabaseConnection
from src.database.models import AccessDecision, ActuationStatus, AuthPolicy


@pytest.fixture
def account_demo(tmp_path):
    original = tmp_path / "original.sqlite3"
    seed = FacePalmDemo(DatabaseConnection(str(original)))
    seed.users.create_user("P01", "An")
    palm = np.eye(1, PALM_FEATURE_DIMENSION, dtype=np.float32)[0]
    seed.enroll_palm("P01", "LEFT", palm)
    target = prepare(original, tmp_path / "shared.sqlite3")
    return target, original, palm


def test_admin_status_controls_next_mock_unlock(account_demo, monkeypatch):
    path, original, palm = account_demo
    client = TestClient(create_app(path), client=("127.0.0.1", 50000))
    assert client.get("/users").headers["X-SmartDoorLock-Source"] == "shared-demo"
    demo = FacePalmDemo(DatabaseConnection(str(path)))
    face = np.eye(1, 128, dtype=np.float32)[0]
    demo.face_templates["P01"] = face
    unlock_calls = []
    unlock = demo.lock.unlock

    def count_unlock():
        unlock_calls.append(True)
        return unlock()

    monkeypatch.setattr(demo.lock, "unlock", count_unlock)

    def verify():
        try:
            result = demo.authenticate(demo.issue_session(), face, palm, "LEFT", policy=AuthPolicy.FACE_AND_PALM)
            return result.decision, result.actuation_status
        finally:
            demo.lock.lock()

    def change(action, confirmation=None):
        return client.post("/admin/user-status", params={"user_id": "P01"},
                           json={"action": action, "confirmation": confirmation})

    assert verify() == (AccessDecision.GRANTED, ActuationStatus.ACTUATED)
    assert change("DEACTIVATE", "wrong").status_code == 422
    assert change("DEACTIVATE", "P01").json()["status"] == "INACTIVE"
    assert verify() == (AccessDecision.DENIED, ActuationStatus.NOT_ACTUATED)
    assert len(unlock_calls) == 1
    assert client.get("/users").json()[0]["has_left_palm"]
    assert change("ACTIVATE").json()["status"] == "ACTIVE"
    assert verify() == (AccessDecision.GRANTED, ActuationStatus.ACTUATED)
    assert len(unlock_calls) == 2

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT decision, hand_side FROM access_logs ORDER BY log_id"
        ).fetchall() == [("GRANTED", "LEFT"), ("DENIED", "LEFT"), ("GRANTED", "LEFT")]

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT actor, user_id, action, old_status, new_status FROM account_admin_audit ORDER BY audit_id"
        ).fetchall() == [
            ("local-demo", "P01", "DEACTIVATE", "ACTIVE", "INACTIVE"),
            ("local-demo", "P01", "ACTIVATE", "INACTIVE", "ACTIVE"),
        ]
    with closing(sqlite3.connect(original)) as conn:
        assert conn.execute("SELECT status FROM users").fetchone() == ("ACTIVE",)


def test_admin_permissions_duplicate_and_audit_rollback(account_demo):
    path, original, _ = account_demo
    client = TestClient(create_app(path), client=("127.0.0.1", 50000))
    body = {"user_id": "P02", "full_name": "Test user"}
    readonly = TestClient(create_app(original), client=("127.0.0.1", 50000))
    remote = TestClient(create_app(path), client=("192.0.2.1", 50000))
    assert readonly.post("/admin/users", json=body).status_code == 403
    assert remote.post("/admin/users", json=body).status_code == 403
    assert client.post("/admin/users", json=body, headers={"Origin": "https://example.com"}).status_code == 403
    assert client.post("/admin/users", json=body).status_code == 201
    assert client.post("/admin/users", json=body).status_code == 409
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT action, old_status, new_status FROM account_admin_audit").fetchall() == [
            ("CREATE", None, "ACTIVE")
        ]
        conn.execute("""CREATE TRIGGER reject_audit BEFORE INSERT ON account_admin_audit
                        BEGIN SELECT RAISE(ABORT, 'test audit failure'); END""")
    assert client.post("/admin/user-status", params={"user_id": "P01"},
                       json={"action": "DEACTIVATE", "confirmation": "P01"}).status_code == 503
    assert client.post("/admin/users", json={**body, "user_id": "P03"}).status_code == 503
    assert [(user["user_id"], user["status"]) for user in client.get("/users").json()] == [
        ("P01", "ACTIVE"), ("P02", "ACTIVE")
    ]

