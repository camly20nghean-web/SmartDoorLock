"""CLI management tool for users, database initialization, and audit log inspection.

Usage examples:
    python scripts/manage_users.py init-db
    python scripts/manage_users.py add-user --user-id NV001 --name "Nguyen Van A"
    python scripts/manage_users.py list-users [--status ACTIVE]
    python scripts/manage_users.py search-users --keyword "Nguyen"
    python scripts/manage_users.py deactivate-user --user-id NV001
    python scripts/manage_users.py activate-user --user-id NV001
    python scripts/manage_users.py delete-user --user-id NV001
    python scripts/manage_users.py view-logs [--limit 20]
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.database.connection import DatabaseConnection
from src.database.models import UserStatus
from src.database.repository import AccessLogRepository, UserRepository


def get_db_path(args=None) -> str:
    """Extract SQLite database file path from CLI args or settings."""
    if args and getattr(args, "db_path", None):
        return args.db_path
    settings = get_settings()
    raw = settings.database_path
    if raw.startswith("sqlite:///"):
        return raw.replace("sqlite:///", "")
    return raw


def handle_init_db(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    print(f"✅ Database initialized successfully at '{db_path}'.")


def handle_add_user(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    status = UserStatus(args.status.upper())
    try:
        user = repo.create_user(user_id=args.user_id, full_name=args.name, status=status)
        print(f"✅ User created: [{user.user_id}] {user.full_name} (Status: {user.status.value})")
    except ValueError as e:
        print(f"❌ Error: {e}")


def handle_list_users(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    status_filter = UserStatus(args.status.upper()) if args.status else None
    users = repo.list_users(status=status_filter)
    if not users:
        print("ℹ️ No users found.")
        return

    print("=" * 80)
    print(f"{'USER_ID':<15} {'FULL NAME':<30} {'STATUS':<12} {'CREATED AT (UTC)'}")
    print("=" * 80)
    for u in users:
        ts = u.created_at[:19] if u.created_at else "N/A"
        print(f"{u.user_id:<15} {u.full_name:<30} {u.status.value:<12} {ts}")
    print("=" * 80)


def handle_search_users(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    users = repo.search_users(keyword=args.keyword)
    if not users:
        print(f"ℹ️ No users matching keyword '{args.keyword}'.")
        return

    print("=" * 80)
    print(f"{'USER_ID':<15} {'FULL NAME':<30} {'STATUS':<12} {'UPDATED AT (UTC)'}")
    print("=" * 80)
    for u in users:
        ts = u.updated_at[:19] if u.updated_at else "N/A"
        print(f"{u.user_id:<15} {u.full_name:<30} {u.status.value:<12} {ts}")
    print("=" * 80)


def handle_update_user(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    status = UserStatus(args.status.upper()) if args.status else None
    if repo.update_user(user_id=args.user_id, full_name=args.name, status=status):
        print(f"✅ User '{args.user_id}' updated successfully.")
    else:
        print(f"❌ User '{args.user_id}' not found.")


def handle_deactivate(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    if repo.deactivate_user(user_id=args.user_id):
        print(f"✅ User '{args.user_id}' has been DEACTIVATED (Status: INACTIVE).")
    else:
        print(f"❌ User '{args.user_id}' not found.")


def handle_activate(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    if repo.activate_user(user_id=args.user_id):
        print(f"✅ User '{args.user_id}' has been ACTIVATED (Status: ACTIVE).")
    else:
        print(f"❌ User '{args.user_id}' not found.")


def handle_delete(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    repo = UserRepository(db)

    if repo.delete_user(user_id=args.user_id):
        print(f"✅ User '{args.user_id}' permanently deleted.")
    else:
        print(f"❌ User '{args.user_id}' not found.")


def handle_view_logs(args):
    db_path = get_db_path(args)
    db = DatabaseConnection(db_path=db_path)
    db.init_db()
    log_repo = AccessLogRepository(db)

    logs = log_repo.get_recent_logs(limit=args.limit, user_id=args.user_id)
    if not logs:
        print("ℹ️ No access logs found.")
        return

    print("=" * 105)
    print(f"{'LOG_ID':<7} {'REQUEST_ID':<16} {'TIMESTAMP (UTC)':<20} {'USER_ID':<12} {'METHOD':<8} {'DECISION':<9} {'ACTUATION':<13} {'REASON'}")
    print("=" * 105)
    for l in logs:
        ts = l.timestamp[:19] if l.timestamp else "N/A"
        uid = l.user_id if l.user_id else "(anonymous)"
        print(f"{l.log_id:<7} {l.request_id:<16} {ts:<20} {uid:<12} {l.auth_method.value:<8} {l.decision.value:<9} {l.actuation_status.value:<13} {l.reason}")
    print("=" * 105)


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--db-path", default=argparse.SUPPRESS, help="Custom SQLite database path for isolated environments")

    parser = argparse.ArgumentParser(description="SmartDoorLock User and Database Manager CLI.", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-db
    subparsers.add_parser("init-db", parents=[parent_parser], help="Initialize SQLite tables.")

    # add-user
    add_parser = subparsers.add_parser("add-user", parents=[parent_parser], help="Register a new authorized user.")
    add_parser.add_argument("--user-id", required=True, help="Unique identifier (e.g. NV001)")
    add_parser.add_argument("--name", required=True, help="Full name of user")
    add_parser.add_argument("--status", default="ACTIVE", choices=["ACTIVE", "INACTIVE", "SUSPENDED"], help="Account status")

    # list-users
    list_parser = subparsers.add_parser("list-users", parents=[parent_parser], help="List registered users.")
    list_parser.add_argument("--status", choices=["ACTIVE", "INACTIVE", "SUSPENDED"], help="Filter by status")

    # search-users
    search_parser = subparsers.add_parser("search-users", parents=[parent_parser], help="Search users by name or ID.")
    search_parser.add_argument("--keyword", required=True, help="Search term")

    # update-user
    update_parser = subparsers.add_parser("update-user", parents=[parent_parser], help="Update user details.")
    update_parser.add_argument("--user-id", required=True, help="Target user ID")
    update_parser.add_argument("--name", help="New full name")
    update_parser.add_argument("--status", choices=["ACTIVE", "INACTIVE", "SUSPENDED"], help="New status")

    # deactivate-user
    deact_parser = subparsers.add_parser("deactivate-user", parents=[parent_parser], help="Revoke access privileges for a user.")
    deact_parser.add_argument("--user-id", required=True, help="User ID to deactivate")

    # activate-user
    act_parser = subparsers.add_parser("activate-user", parents=[parent_parser], help="Restore access privileges for a user.")
    act_parser.add_argument("--user-id", required=True, help="User ID to activate")

    # delete-user
    del_parser = subparsers.add_parser("delete-user", parents=[parent_parser], help="Delete user permanently.")
    del_parser.add_argument("--user-id", required=True, help="User ID to delete")

    # view-logs
    logs_parser = subparsers.add_parser("view-logs", parents=[parent_parser], help="Display audit access logs.")
    logs_parser.add_argument("--limit", type=int, default=20, help="Number of records to show")
    logs_parser.add_argument("--user-id", default=None, help="Filter by user ID")

    args = parser.parse_args()

    commands = {
        "init-db": handle_init_db,
        "add-user": handle_add_user,
        "list-users": handle_list_users,
        "search-users": handle_search_users,
        "update-user": handle_update_user,
        "deactivate-user": handle_deactivate,
        "activate-user": handle_activate,
        "delete-user": handle_delete,
        "view-logs": handle_view_logs,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)


if __name__ == "__main__":
    main()
