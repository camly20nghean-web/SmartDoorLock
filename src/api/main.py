"""Local administration API for SmartDoorLock demos."""
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, computed_field

from src.database.schema import has_access_log_hand_side
from src.api.accounts import has_admin_audit, install_account_routes
from src.database.models import AccessDecision, ActuationStatus, AuthMethod, UserStatus

DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / "data" / "admin_api_demo.sqlite3"
VIETNAM_TIME = timezone(timedelta(hours=7))


class UserSummary(BaseModel):
    user_id: str
    full_name: str
    status: UserStatus
    has_face: bool
    has_left_palm: bool
    has_right_palm: bool


class AccessLogSummary(BaseModel):
    timestamp: str
    request_id: str
    user_id: str | None
    auth_method: AuthMethod
    decision: AccessDecision
    reason: str
    actuation_status: ActuationStatus
    hand_side: Literal["LEFT", "RIGHT"] | None = None

    @computed_field(description="Vietnam time in DD/MM/YYYY HH:MM:SS format (UTC+07:00)")
    @property
    def timestamp_local(self) -> str | None:
        try:
            recorded_at = datetime.fromisoformat(self.timestamp)
        except ValueError:
            return None
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        return recorded_at.astimezone(VIETNAM_TIME).strftime("%d/%m/%Y %H:%M:%S")


class AccessLogPage(BaseModel):
    items: list[AccessLogSummary]
    limit: int
    offset: int
    total: int


@contextmanager
def read_database(path: Path):
    try:
        if not path.is_file():
            raise HTTPException(503, f"Admin database file not found: {path}")
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            yield conn
    except sqlite3.Error as error:
        raise HTTPException(503, "Cannot read the admin database. Check the SQLite file and schema.") from error


def create_app(database_path: Path | str | None = None) -> FastAPI:
    path = Path(database_path or os.environ.get("SMARTDOORLOCK_ADMIN_DB") or DEFAULT_DATABASE).resolve()
    app = FastAPI(title="SmartDoorLock Admin API")
    demo_source = False
    if path.is_file():
        try:
            with read_database(path) as conn:
                demo_source = has_admin_audit(conn)
        except HTTPException:
            pass
    install_account_routes(app, path, demo_source)

    def source_headers(response: Response):
        response.headers["X-SmartDoorLock-Source"] = "shared-demo" if demo_source else "read-only"
        response.headers["X-SmartDoorLock-Account-Management"] = "enabled" if demo_source else "disabled"

    @app.get("/users", response_model=list[UserSummary])
    def users(
        response: Response,
        q: Annotated[str | None, Query(max_length=100, description="Substring of user ID or full name")] = None,
        status: UserStatus | None = None,
    ):
        source_headers(response)
        search = q.strip().casefold() if q else ""
        with read_database(path) as conn:
            conn.create_function("casefold", 1, str.casefold, deterministic=True)
            rows = conn.execute("""
                SELECT u.user_id, u.full_name, u.status,
                    EXISTS (
                        SELECT 1 FROM biometric_templates t
                        WHERE t.user_id = u.user_id AND t.biometric_type = 'FACE'
                    ) AS has_face,
                    EXISTS (
                        SELECT 1 FROM demo_palm_templates p
                        WHERE p.user_id = u.user_id AND p.hand_side = 'LEFT'
                    ) AS has_left_palm,
                    EXISTS (
                        SELECT 1 FROM demo_palm_templates p
                        WHERE p.user_id = u.user_id AND p.hand_side = 'RIGHT'
                    ) AS has_right_palm
                FROM users u
                WHERE (:status IS NULL OR u.status = :status)
                    AND (:search = '' OR instr(casefold(u.user_id), :search) > 0
                        OR instr(casefold(u.full_name), :search) > 0)
                ORDER BY u.user_id
            """, {"status": status, "search": search}).fetchall()
        return [dict(row) for row in rows]

    @app.get("/access-logs", response_model=AccessLogPage)
    def access_logs(
        response: Response,
        auth_method: AuthMethod | None = None,
        decision: AccessDecision | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
    ):
        source_headers(response)
        params = {"method": auth_method, "decision": decision, "limit": limit, "offset": offset}
        filters = """
            WHERE (:method IS NULL OR auth_method = :method)
                AND (:decision IS NULL OR decision = :decision)
        """
        with read_database(path) as conn:
            conn.execute("BEGIN")
            total = conn.execute("SELECT COUNT(*) FROM access_logs " + filters, params).fetchone()[0]
            hand_column = "hand_side" if has_access_log_hand_side(conn) else "NULL AS hand_side"
            rows = conn.execute(f"""
                SELECT timestamp, request_id, user_id, auth_method, decision, reason, actuation_status, {hand_column}
                FROM access_logs
            """ + filters + """
                ORDER BY timestamp DESC, log_id DESC
                LIMIT :limit OFFSET :offset
            """, params).fetchall()
        return {"items": [dict(row) for row in rows], "limit": limit, "offset": offset, "total": total}

    return app


app = create_app()
