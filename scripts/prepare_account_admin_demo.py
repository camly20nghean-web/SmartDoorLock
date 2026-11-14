"""Create a new SQLite copy for account management and webcam verification."""
from contextlib import closing
from pathlib import Path
import argparse
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.accounts import AUDIT_SCHEMA

SOURCE = ROOT / "data" / "face_palm_webcam_demo" / "demo.sqlite3"
TARGET = ROOT / "data" / "account_admin_demo.sqlite3"


def prepare(source=SOURCE, target=TARGET):
    if not source.is_file():
        raise FileNotFoundError(source)
    # Exclusive creation avoids overwriting a source or an earlier demo.
    target.touch(exist_ok=False)
    try:
        with (
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original,
            closing(sqlite3.connect(target.resolve().as_uri() + "?mode=rw", uri=True)) as copy,
            copy,
        ):
            original.backup(copy)
            copy.execute(AUDIT_SCHEMA)
    except Exception:
        target.unlink()
        raise
    return target


def prepare_empty(target=TARGET):
    from scripts.demo_face_palm_unlock import FacePalmDemo
    from src.database.connection import DatabaseConnection

    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=False)
    demo = FacePalmDemo(DatabaseConnection(str(target)))
    with closing(demo.db.get_connection()) as conn, conn:
        conn.execute(AUDIT_SCHEMA)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", action="store_true", help="Create an empty database for a fresh clone")
    args = parser.parse_args()
    path = prepare_empty() if args.empty else prepare()
    print(f"Shared account demo database: {path.relative_to(ROOT)}")
