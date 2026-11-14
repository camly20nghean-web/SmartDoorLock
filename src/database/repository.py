"""Data Access Object (DAO) repositories for users, biometric templates, and access logs."""
import logging
import sqlite3
from typing import List, Optional

from .connection import DatabaseConnection
from .models import (
    AccessDecision,
    AccessLogRecord,
    ActuationStatus,
    AuthMethod,
    BiometricTemplateRecord,
    BiometricType,
    FeatureFormat,
    UserRecord,
    UserStatus,
    get_utc_now_iso,
)

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository handling CRUD, status toggles, and searching for users."""

    def __init__(self, db: DatabaseConnection) -> None:
        self.db = db

    def create_user(
        self,
        user_id: str,
        full_name: str,
        status: UserStatus = UserStatus.ACTIVE
    ) -> UserRecord:
        """
        Register a new user in the database.
        
        Raises:
            ValueError: If user_id already exists or invalid data provided.
        """
        user_id = user_id.strip()
        full_name = full_name.strip()
        if not user_id:
            raise ValueError("user_id cannot be empty.")
        if not full_name:
            raise ValueError("full_name cannot be empty.")

        now_utc = get_utc_now_iso()
        status_val = status.value if isinstance(status, UserStatus) else status

        sql = """
        INSERT INTO users (user_id, full_name, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?);
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute(sql, (user_id, full_name, status_val, now_utc, now_utc))
                conn.commit()
            logger.info("Created user '%s' (%s) status=%s", user_id, full_name, status_val)
            return UserRecord(
                user_id=user_id,
                full_name=full_name,
                status=UserStatus(status_val),
                created_at=now_utc,
                updated_at=now_utc
            )
        except sqlite3.IntegrityError as e:
            logger.warning("Failed to create user '%s': duplicate user_id. %s", user_id, e)
            raise ValueError(f"User with user_id '{user_id}' already exists.") from e

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Fetch user by unique user_id."""
        sql = "SELECT * FROM users WHERE user_id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_user(row)

    def list_users(self, status: Optional[UserStatus] = None) -> List[UserRecord]:
        """List users, optionally filtered by status."""
        sql = "SELECT * FROM users"
        params = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value if isinstance(status, UserStatus) else status,)
        sql += " ORDER BY created_at ASC;"

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [self._row_to_user(r) for r in cursor.fetchall()]

    def search_users(self, keyword: str) -> List[UserRecord]:
        """Search users matching keyword in user_id or full_name."""
        keyword = keyword.strip()
        if not keyword:
            return self.list_users()

        pattern = f"%{keyword}%"
        sql = """
        SELECT * FROM users
        WHERE user_id LIKE ? OR full_name LIKE ?
        ORDER BY created_at ASC;
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (pattern, pattern))
            return [self._row_to_user(r) for r in cursor.fetchall()]

    def update_user(
        self,
        user_id: str,
        full_name: Optional[str] = None,
        status: Optional[UserStatus] = None
    ) -> bool:
        """Update full_name and/or status of an existing user."""
        clauses = []
        params = []
        if full_name is not None:
            full_name_clean = full_name.strip()
            if not full_name_clean:
                raise ValueError("full_name cannot be empty.")
            clauses.append("full_name = ?")
            params.append(full_name_clean)

        if status is not None:
            status_val = status.value if isinstance(status, UserStatus) else status
            clauses.append("status = ?")
            params.append(status_val)

        if not clauses:
            return True

        now_utc = get_utc_now_iso()
        clauses.append("updated_at = ?")
        params.append(now_utc)

        params.append(user_id)
        sql = f"UPDATE users SET {', '.join(clauses)} WHERE user_id = ?;"

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            conn.commit()
            return cursor.rowcount > 0

    def deactivate_user(self, user_id: str) -> bool:
        """Set user status to INACTIVE, revoking authorization."""
        return self.update_user(user_id, status=UserStatus.INACTIVE)

    def activate_user(self, user_id: str) -> bool:
        """Set user status to ACTIVE, enabling authorization."""
        return self.update_user(user_id, status=UserStatus.ACTIVE)

    def suspend_user(self, user_id: str) -> bool:
        """Set user status to SUSPENDED."""
        return self.update_user(user_id, status=UserStatus.SUSPENDED)

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete user record (cascading to templates)."""
        sql = "DELETE FROM users WHERE user_id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            full_name=row["full_name"],
            status=UserStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )


class BiometricTemplateRepository:
    """Repository handling biometric template records linked to users."""

    def __init__(self, db: DatabaseConnection) -> None:
        self.db = db

    def create_template(
        self,
        user_id: str,
        biometric_type: BiometricType,
        feature_data: bytes,
        feature_format: FeatureFormat,
        feature_dimension: int,
        model_version: str
    ) -> BiometricTemplateRecord:
        """
        Store a biometric feature vector with full metadata.
        
        Raises:
            sqlite3.IntegrityError: If user_id does not exist (foreign key check).
            ValueError: If feature data, dimension, or model metadata is invalid.
        """
        if not feature_data:
            raise ValueError("feature_data cannot be empty.")
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be greater than zero.")
        if not model_version.strip():
            raise ValueError("model_version cannot be empty.")

        now_utc = get_utc_now_iso()
        b_type_val = biometric_type.value if isinstance(biometric_type, BiometricType) else biometric_type
        fmt_val = feature_format.value if isinstance(feature_format, FeatureFormat) else feature_format

        sql = """
        INSERT INTO biometric_templates (
            user_id, biometric_type, feature_data, feature_format, feature_dimension, model_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (
                user_id,
                b_type_val,
                feature_data,
                fmt_val,
                int(feature_dimension),
                model_version.strip(),
                now_utc
            ))
            conn.commit()
            template_id = cursor.lastrowid

        return BiometricTemplateRecord(
            template_id=template_id,
            user_id=user_id,
            biometric_type=BiometricType(b_type_val),
            feature_data=feature_data,
            feature_format=FeatureFormat(fmt_val),
            feature_dimension=int(feature_dimension),
            model_version=model_version.strip(),
            created_at=now_utc
        )

    def get_templates_by_user(
        self,
        user_id: str,
        biometric_type: Optional[BiometricType] = None
    ) -> List[BiometricTemplateRecord]:
        """Fetch all biometric templates registered for a given user."""
        sql = "SELECT * FROM biometric_templates WHERE user_id = ?"
        params = [user_id]
        if biometric_type is not None:
            sql += " AND biometric_type = ?"
            params.append(biometric_type.value if isinstance(biometric_type, BiometricType) else biometric_type)
        sql += " ORDER BY created_at ASC;"

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [self._row_to_template(r) for r in cursor.fetchall()]

    def count_templates(self, user_id: Optional[str] = None) -> int:
        """Count registered templates."""
        sql = "SELECT COUNT(*) as cnt FROM biometric_templates"
        params = ()
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params = (user_id,)

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return int(cursor.fetchone()["cnt"])

    def delete_templates_by_user(self, user_id: str) -> int:
        """Delete all templates belonging to a specific user."""
        sql = "DELETE FROM biometric_templates WHERE user_id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (user_id,))
            conn.commit()
            return cursor.rowcount

    @staticmethod
    def _row_to_template(row: sqlite3.Row) -> BiometricTemplateRecord:
        return BiometricTemplateRecord(
            template_id=row["template_id"],
            user_id=row["user_id"],
            biometric_type=BiometricType(row["biometric_type"]),
            feature_data=row["feature_data"],
            feature_format=FeatureFormat(row["feature_format"]),
            feature_dimension=int(row["feature_dimension"]),
            model_version=row["model_version"],
            created_at=row["created_at"]
        )


class AccessLogRepository:
    """Repository handling audit trail logging with strict UTC timestamps and privacy."""

    def __init__(self, db: DatabaseConnection) -> None:
        self.db = db

    def log_event(
        self,
        request_id: str,
        auth_method: AuthMethod,
        decision: AccessDecision,
        actuation_status: ActuationStatus,
        reason: str,
        user_id: Optional[str] = None,
        *, hand_side: Optional[str] = None,
    ) -> int:
        """
        Record an access event into the audit trail.
        
        Decouples authorization decision from physical actuator result.
        Never stores raw biometric or image data in the log.
        """
        now_utc = get_utc_now_iso()
        m_val = auth_method.value if isinstance(auth_method, AuthMethod) else auth_method
        d_val = decision.value if isinstance(decision, AccessDecision) else decision
        a_val = actuation_status.value if isinstance(actuation_status, ActuationStatus) else actuation_status

        sql = """
        INSERT INTO access_logs (
            request_id, user_id, auth_method, decision, actuation_status, timestamp, reason, hand_side
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, (
                request_id,
                user_id,
                m_val,
                d_val,
                a_val,
                now_utc,
                reason.strip(),
                hand_side if m_val in ("PALM", "DUAL") else None,
            ))
            conn.commit()
            log_id = cursor.lastrowid

        logger.info(
            "AccessLog #%d | req=%s | user=%s | method=%s | decision=%s | actuation=%s | reason=%s",
            log_id, request_id, user_id or "ANONYMOUS", m_val, d_val, a_val, reason
        )
        return log_id

    def get_recent_logs(
        self,
        limit: int = 20,
        user_id: Optional[str] = None
    ) -> List[AccessLogRecord]:
        """Fetch latest N access records ordered chronologically descending."""
        sql = "SELECT * FROM access_logs"
        params = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY log_id DESC LIMIT ?;"
        params.append(max(1, limit))

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [self._row_to_log(r) for r in cursor.fetchall()]

    def count_logs(self, decision: Optional[AccessDecision] = None) -> int:
        """Count total access records, optionally filtered by decision."""
        sql = "SELECT COUNT(*) as cnt FROM access_logs"
        params = ()
        if decision is not None:
            sql += " WHERE decision = ?"
            params = (decision.value if isinstance(decision, AccessDecision) else decision,)

        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return int(cursor.fetchone()["cnt"])

    @staticmethod
    def _row_to_log(row: sqlite3.Row) -> AccessLogRecord:
        return AccessLogRecord(
            log_id=row["log_id"],
            request_id=row["request_id"],
            user_id=row["user_id"],
            auth_method=AuthMethod(row["auth_method"]),
            decision=AccessDecision(row["decision"]),
            actuation_status=ActuationStatus(row["actuation_status"]),
            timestamp=row["timestamp"],
            reason=row["reason"],
            hand_side=row["hand_side"] if "hand_side" in row.keys() else None,
        )
