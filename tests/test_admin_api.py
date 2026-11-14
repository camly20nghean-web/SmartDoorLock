from contextlib import closing, contextmanager
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.database.schema import SCHEMA_SQL


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(SCHEMA_SQL)
        conn.executescript("""
            CREATE TABLE demo_palm_templates (
                user_id TEXT NOT NULL,
                hand_side TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                vector BLOB NOT NULL
            );

            INSERT INTO users (user_id, full_name, status, created_at, updated_at) VALUES
                ('P01', 'Đặng An', 'ACTIVE', '2026-11-12', '2026-11-12'),
                ('P02', 'Bình', 'SUSPENDED', '2026-11-12', '2026-11-12');
            INSERT INTO biometric_templates (
                user_id, biometric_type, feature_data, feature_format,
                feature_dimension, model_version, created_at
            ) VALUES ('P01', 'FACE', X'0102', 'FLOAT32_EMBEDDING', 128, 'sface', '2026-11-12');
            INSERT INTO demo_palm_templates VALUES
                ('P01', 'LEFT', 'lbp', X'0304'),
                ('P02', 'RIGHT', 'lbp', X'0506');

            INSERT INTO access_logs (
                request_id, user_id, auth_method, decision, actuation_status, timestamp, reason
            ) VALUES
                ('older', 'P01', 'PALM', 'DENIED', 'NOT_ACTUATED', '2026-11-12T09:00:00+00:00', 'old'),
                ('first', 'P01', 'PALM', 'DENIED', 'NOT_ACTUATED', '2026-11-12T10:00:00+00:00', 'rejected'),
                ('second', NULL, 'PALM', 'DENIED', 'NOT_ACTUATED', '2026-11-12T10:00:00+00:00', 'anonymous'),
                ('dual', 'P01', 'DUAL', 'GRANTED', 'FAILED', '2026-11-12T20:18:42.684784+00:00', 'actuator error');
        """)
    return path


def test_users_filter_and_biometric_columns_are_not_read(database, monkeypatch):
    read_database = main.read_database

    @contextmanager
    def block_vectors(path):
        with read_database(path) as conn:
            def authorize(action, table, column, database_name, source):
                if action == sqlite3.SQLITE_READ and column in {"feature_data", "vector"}:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            conn.set_authorizer(authorize)
            yield conn

    monkeypatch.setattr(main, "read_database", block_vectors)
    with TestClient(main.create_app(database)) as client:
        response = client.get("/users")
        assert response.status_code == 200
        assert response.json() == [
            {"user_id": "P01", "full_name": "Đặng An", "status": "ACTIVE",
             "has_face": True, "has_left_palm": True, "has_right_palm": False},
            {"user_id": "P02", "full_name": "Bình", "status": "SUSPENDED",
             "has_face": False, "has_left_palm": False, "has_right_palm": True},
        ]
        assert [row["user_id"] for row in client.get("/users", params={"q": "đẶNG"}).json()] == ["P01"]
        assert [row["user_id"] for row in client.get("/users", params={"q": "p02", "status": "SUSPENDED"}).json()] == ["P02"]
        assert client.get("/users", params={"q": "p02", "status": "ACTIVE"}).json() == []
        assert client.get("/users", params={"q": "%"}).json() == []


def test_logs_filter_and_paginate_in_stable_order(database):
    with TestClient(main.create_app(database)) as client:
        params = {"auth_method": "PALM", "decision": "DENIED", "limit": 1}
        for offset, request_id in enumerate(("second", "first", "older")):
            response = client.get("/access-logs", params={**params, "offset": offset})
            assert response.status_code == 200
            page = response.json()
            assert page["total"] == 3
            assert page["offset"] == offset and page["limit"] == 1
            assert [row["request_id"] for row in page["items"]] == [request_id]
        assert client.get("/access-logs", params={**params, "offset": 3}).json()["items"] == []
        row, = client.get("/access-logs", params={"auth_method": "DUAL"}).json()["items"]
        assert row["decision"] == "GRANTED" and row["actuation_status"] == "FAILED"
        assert row["timestamp"] == "2026-11-12T20:18:42.684784+00:00"
        assert row["timestamp_local"] == "13/11/2026 03:18:42"
        assert client.get("/access-logs", params={"limit": 101}).status_code == 422


@pytest.mark.parametrize("endpoint", ["/users", "/access-logs"])
def test_missing_database_is_reported_without_creating_it(tmp_path, monkeypatch, endpoint):
    path = tmp_path / "missing.sqlite3"
    monkeypatch.setenv("SMARTDOORLOCK_ADMIN_DB", str(path))
    with TestClient(main.create_app()) as client:
        response = client.get(endpoint)
    assert response.status_code == 503
    assert "database file not found" in response.json()["detail"]
    assert not path.exists()


def test_connection_rejects_writes_at_sqlite_level(database):
    with main.read_database(database) as conn:
        conn.execute("PRAGMA query_only = OFF")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM users")

def test_legacy_logs_and_hand_side_migration(database):
    from src.database.schema import migrate_access_log_hand_side

    with closing(sqlite3.connect(database)) as conn:
        conn.execute("ALTER TABLE access_logs DROP COLUMN hand_side")
        before = conn.execute("SELECT * FROM access_logs ORDER BY log_id").fetchall()
        with TestClient(main.create_app(database)) as client:
            assert all(row["hand_side"] is None for row in client.get("/access-logs").json()["items"])
        migrate_access_log_hand_side(conn)
        migrate_access_log_hand_side(conn)
        assert conn.execute("SELECT * FROM access_logs ORDER BY log_id").fetchall() == [
            (*row, None) for row in before
        ]
        conn.execute("UPDATE access_logs SET hand_side = 'RIGHT' WHERE request_id = 'second'")
        conn.execute("UPDATE access_logs SET hand_side = 'LEFT' WHERE request_id = 'dual'")
        conn.commit()
    with TestClient(main.create_app(database)) as client:
        rows = {row["request_id"]: row for row in client.get("/access-logs").json()["items"]}
        assert rows["second"]["hand_side"] == "RIGHT" and rows["second"]["decision"] == "DENIED"
        assert rows["dual"]["hand_side"] == "LEFT"
        assert rows["older"]["hand_side"] is None
