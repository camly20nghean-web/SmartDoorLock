"""Audited local account mutations for the demo."""
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from src.api.enrollment import Enrollment, enroll
from src.api.unlock import CameraInput, preview_camera, verify
from src.database.models import AuthPolicy, UserStatus, get_utc_now_iso

AUDIT_SCHEMA = """
CREATE TABLE account_admin_audit (
    audit_id INTEGER PRIMARY KEY,
    actor TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    action TEXT NOT NULL CHECK(action IN ('CREATE', 'DEACTIVATE', 'ACTIVATE')),
    old_status TEXT,
    new_status TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
"""
UserId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
FullName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class NewUser(BaseModel):
    user_id: UserId
    full_name: FullName


class StatusChange(BaseModel):
    action: Literal["DEACTIVATE", "ACTIVATE"]
    confirmation: UserId | None = None


def has_admin_audit(conn):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_admin_audit'"
    ).fetchone() is not None


@contextmanager
def account_transaction(path: Path):
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as conn, conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            if not has_admin_audit(conn):
                raise HTTPException(403, "Account changes require a dedicated demo database.")
            yield conn
    except sqlite3.Error as error:
        raise HTTPException(503, "Account change failed; no change was saved.") from error


def install_account_routes(app: FastAPI, path: Path, enabled: bool):
    def local_demo_only(request: Request):
        # Streamlit calls the API server-side. Reject browser cross-origin writes.
        if (not enabled or not request.client or request.client.host not in {"127.0.0.1", "::1"}
                or "origin" in request.headers):
            raise HTTPException(403, "Account changes require local access to the dedicated demo database.")

    def audit(conn, user_id, action, old, new, now):
        conn.execute(
            """INSERT INTO account_admin_audit (actor, user_id, action, old_status, new_status, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""", ("local-demo", user_id, action, old, new, now),
        )

    @app.post("/admin/users", status_code=201, dependencies=[Depends(local_demo_only)])
    def create_user(data: NewUser):
        with account_transaction(path) as conn:
            if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (data.user_id,)).fetchone():
                raise HTTPException(409, "User ID already exists.")
            now = get_utc_now_iso()
            conn.execute(
                """INSERT INTO users (user_id, full_name, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""", (data.user_id, data.full_name, UserStatus.ACTIVE, now, now),
            )
            audit(conn, data.user_id, "CREATE", None, UserStatus.ACTIVE, now)
        return {"user_id": data.user_id, "status": UserStatus.ACTIVE}

    @app.post("/admin/user-status", dependencies=[Depends(local_demo_only)])
    def change_status(user_id: UserId, data: StatusChange):
        deactivate = data.action == "DEACTIVATE"
        if deactivate and data.confirmation != user_id:
            raise HTTPException(422, "Confirm the exact user ID to deactivate.")
        old, new = (UserStatus.ACTIVE, UserStatus.INACTIVE) if deactivate else (UserStatus.INACTIVE, UserStatus.ACTIVE)
        with account_transaction(path) as conn:
            row = conn.execute("SELECT status FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "User not found.")
            if row[0] != old:
                raise HTTPException(409, "User status changed or does not allow this action.")
            now = get_utc_now_iso()
            conn.execute("UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?", (new, now, user_id))
            audit(conn, user_id, data.action, old, new, now)
        return {"user_id": user_id, "status": new}



    @app.post("/admin/enrollment", dependencies=[Depends(local_demo_only)])
    def enroll_user(user_id: UserId, data: Enrollment):
        return enroll(path, user_id, data)


    @app.post("/admin/verify", dependencies=[Depends(local_demo_only)])
    def verify_user(policy: AuthPolicy, camera: CameraInput | None = None):
        return verify(path, policy, camera)

    @app.post("/admin/camera-preview", dependencies=[Depends(local_demo_only)])
    def camera_preview(camera: CameraInput):
        return preview_camera(camera)
