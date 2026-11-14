"""Database layer package exporting models, connection, repositories, and schemas."""
from .connection import DatabaseConnection
from .models import (
    AccessDecision,
    AccessLogRecord,
    ActuationStatus,
    AuthMethod,
    AuthorizationOutcome,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricTemplateRecord,
    BiometricType,
    FeatureFormat,
    UserRecord,
    UserStatus,
    generate_request_id,
    get_utc_now_iso,
)
from .repository import (
    AccessLogRepository,
    BiometricTemplateRepository,
    UserRepository,
)
from .schema import (
    ACCESS_LOGS_TABLE_SQL,
    BIOMETRIC_TEMPLATES_TABLE_SQL,
    SCHEMA_SQL,
    USERS_TABLE_SQL,
)

__all__ = [
    "DatabaseConnection",
    "UserRecord",
    "UserStatus",
    "BiometricTemplateRecord",
    "BiometricType",
    "FeatureFormat",
    "AccessLogRecord",
    "AuthMethod",
    "AccessDecision",
    "ActuationStatus",
    "AuthPolicy",
    "BiometricAuthClaim",
    "AuthorizationOutcome",
    "generate_request_id",
    "get_utc_now_iso",
    "UserRepository",
    "BiometricTemplateRepository",
    "AccessLogRepository",
    "USERS_TABLE_SQL",
    "BIOMETRIC_TEMPLATES_TABLE_SQL",
    "ACCESS_LOGS_TABLE_SQL",
    "SCHEMA_SQL",
]
