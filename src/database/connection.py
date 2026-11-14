"""SQLite connection manager enforcing foreign keys and schema migration."""
import logging
from pathlib import Path
import sqlite3
from typing import Optional

from .schema import SCHEMA_SQL, migrate_access_log_hand_side

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages SQLite database connection lifecycle and schema integrity."""

    def __init__(self, db_path: str = "smart_door_lock.db") -> None:
        self.db_path = db_path
        self._shared_memory_conn: Optional[sqlite3.Connection] = None
        self._ensure_parent_directory()

        if self.db_path == ":memory:":
            # For in-memory testing, retain a persistent connection to keep tables in RAM
            self._shared_memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_memory_conn.row_factory = sqlite3.Row
            self._shared_memory_conn.execute("PRAGMA foreign_keys = ON;")

    def _ensure_parent_directory(self) -> None:
        """Create parent directory if database is file-based."""
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if parent and str(parent) != ".":
                parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """
        Open or return connection with foreign keys strictly enabled.
        
        Returns:
            sqlite3.Connection: Ready-to-use SQLite connection instance.
        """
        if self._shared_memory_conn is not None:
            return self._shared_memory_conn

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enforce foreign key constraints on every connection
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Execute DDL statements to ensure all tables and indexes exist."""
        conn = self.get_connection()
        conn.executescript(SCHEMA_SQL)
        migrate_access_log_hand_side(conn)
        logger.info("Database schema initialized successfully at '%s'.", self.db_path)

    def close(self) -> None:
        """Close shared in-memory connection if active."""
        if self._shared_memory_conn is not None:
            self._shared_memory_conn.close()
            self._shared_memory_conn = None
