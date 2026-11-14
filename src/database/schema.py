"""Database schema definitions for SQLite adhering to Phase 2 specifications.

Enforces:
- Foreign key cascading and set null rules.
- Check constraints on enumerated values.
- Explicit biometric metadata (feature_format, feature_dimension, model_version).
- Audit trail decoupling authorization decision from physical actuation status.
"""

USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

BIOMETRIC_TEMPLATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS biometric_templates (
    template_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    biometric_type TEXT NOT NULL CHECK(biometric_type IN ('FACE', 'PALM')),
    feature_data BLOB NOT NULL,
    feature_format TEXT NOT NULL CHECK(feature_format IN ('FLOAT32_EMBEDDING', 'GABOR_VECTOR', 'BINARY_CODE')),
    feature_dimension INTEGER NOT NULL CHECK(feature_dimension > 0),
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
"""

ACCESS_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS access_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    auth_method TEXT NOT NULL CHECK(auth_method IN ('FACE', 'PALM', 'DUAL', 'MANUAL')),
    decision TEXT NOT NULL CHECK(decision IN ('GRANTED', 'DENIED')),
    actuation_status TEXT NOT NULL CHECK(actuation_status IN ('ACTUATED', 'NOT_ACTUATED', 'FAILED')),
    timestamp TEXT NOT NULL,
    reason TEXT NOT NULL,
    hand_side TEXT CHECK(hand_side IN ('LEFT', 'RIGHT')),
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_templates_user ON biometric_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_user ON access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_logs_request ON access_logs(request_id);
"""

SCHEMA_SQL = (
    USERS_TABLE_SQL
    + "\n"
    + BIOMETRIC_TEMPLATES_TABLE_SQL
    + "\n"
    + ACCESS_LOGS_TABLE_SQL
    + "\n"
    + CREATE_INDEXES_SQL
)


def has_access_log_hand_side(conn):
    return any(column[1] == "hand_side" for column in conn.execute("PRAGMA table_info(access_logs)"))


def migrate_access_log_hand_side(conn):
    """Add nullable hand metadata to an existing access log table."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        if not has_access_log_hand_side(conn):
            conn.execute("ALTER TABLE access_logs ADD COLUMN hand_side TEXT CHECK(hand_side IN ('LEFT', 'RIGHT'))")
