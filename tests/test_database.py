"""Unit tests for SQLite database schema, foreign keys, cascading rules, and repositories."""
import sqlite3
import pytest

from src.database.connection import DatabaseConnection
from src.database.models import (
    AccessDecision,
    ActuationStatus,
    AuthMethod,
    BiometricType,
    FeatureFormat,
    UserStatus,
)
from src.database.repository import (
    AccessLogRepository,
    BiometricTemplateRepository,
    UserRepository,
)


@pytest.fixture
def db():
    """Create an isolated in-memory SQLite database instance with foreign keys enabled."""
    db_conn = DatabaseConnection(db_path=":memory:")
    db_conn.init_db()
    return db_conn


@pytest.fixture
def user_repo(db):
    return UserRepository(db)


@pytest.fixture
def template_repo(db):
    return BiometricTemplateRepository(db)


@pytest.fixture
def log_repo(db):
    return AccessLogRepository(db)


class TestUserRepositoryCRUD:
    """Test suite covering User CRUD operations, status management, and search."""

    def test_create_and_get_user(self, user_repo):
        user = user_repo.create_user("NV001", "Nguyen Van A", status=UserStatus.ACTIVE)
        assert user.user_id == "NV001"
        assert user.full_name == "Nguyen Van A"
        assert user.status == UserStatus.ACTIVE
        assert user.is_active is True

        fetched = user_repo.get_user("NV001")
        assert fetched is not None
        assert fetched.user_id == "NV001"
        assert fetched.full_name == "Nguyen Van A"

    def test_duplicate_user_id_raises_value_error(self, user_repo):
        user_repo.create_user("NV002", "Tran Van B")
        with pytest.raises(ValueError) as exc_info:
            user_repo.create_user("NV002", "Duplicate ID User")
        assert "already exists" in str(exc_info.value)

    def test_invalid_empty_fields_raise_value_error(self, user_repo):
        with pytest.raises(ValueError):
            user_repo.create_user("   ", "Name")
        with pytest.raises(ValueError):
            user_repo.create_user("ID01", "   ")

    def test_list_users_filtering_by_status(self, user_repo):
        user_repo.create_user("U1", "Active User 1", status=UserStatus.ACTIVE)
        user_repo.create_user("U2", "Inactive User", status=UserStatus.INACTIVE)
        user_repo.create_user("U3", "Suspended User", status=UserStatus.SUSPENDED)
        user_repo.create_user("U4", "Active User 2", status=UserStatus.ACTIVE)

        all_users = user_repo.list_users()
        assert len(all_users) == 4

        active_users = user_repo.list_users(status=UserStatus.ACTIVE)
        assert len(active_users) == 2
        assert {u.user_id for u in active_users} == {"U1", "U4"}

        suspended = user_repo.list_users(status=UserStatus.SUSPENDED)
        assert len(suspended) == 1
        assert suspended[0].user_id == "U3"

    def test_search_users(self, user_repo):
        user_repo.create_user("NV101", "Nguyen Van An")
        user_repo.create_user("NV102", "Le Thi Binh")
        user_repo.create_user("QL201", "Nguyen Hoang Cuong")

        # Search by name
        results = user_repo.search_users("Nguyen")
        assert len(results) == 2
        assert {u.user_id for u in results} == {"NV101", "QL201"}

        # Search by ID
        results_id = user_repo.search_users("QL")
        assert len(results_id) == 1
        assert results_id[0].user_id == "QL201"

    def test_update_user_and_status_transitions(self, user_repo):
        user_repo.create_user("U10", "Old Name", status=UserStatus.ACTIVE)

        # Update name
        user_repo.update_user("U10", full_name="New Name")
        assert user_repo.get_user("U10").full_name == "New Name"

        # Deactivate
        user_repo.deactivate_user("U10")
        assert user_repo.get_user("U10").status == UserStatus.INACTIVE
        assert user_repo.get_user("U10").is_active is False

        # Reactivate
        user_repo.activate_user("U10")
        assert user_repo.get_user("U10").status == UserStatus.ACTIVE
        assert user_repo.get_user("U10").is_active is True

        # Suspend
        user_repo.suspend_user("U10")
        assert user_repo.get_user("U10").status == UserStatus.SUSPENDED


class TestBiometricTemplatesAndForeignKeys:
    """Test suite covering biometric metadata, constraints, and cascading deletes."""

    def test_create_template_with_metadata(self, user_repo, template_repo):
        user_repo.create_user("NV001", "Nguyen Van A")
        dummy_feature = b"\x00\x01\x02\x03" * 128  # 512 bytes

        template = template_repo.create_template(
            user_id="NV001",
            biometric_type=BiometricType.FACE,
            feature_data=dummy_feature,
            feature_format=FeatureFormat.FLOAT32_EMBEDDING,
            feature_dimension=128,
            model_version="insightface-buffalo_l-v1"
        )

        assert template.template_id is not None
        assert template.user_id == "NV001"
        assert template.biometric_type == BiometricType.FACE
        assert template.feature_format == FeatureFormat.FLOAT32_EMBEDDING
        assert template.feature_dimension == 128
        assert template.model_version == "insightface-buffalo_l-v1"
        assert template.created_at is not None

    def test_foreign_key_enforcement_nonexistent_user(self, template_repo):
        """Inserting template for a nonexistent user must raise IntegrityError."""
        dummy_feature = b"\x00" * 64
        with pytest.raises(sqlite3.IntegrityError):
            template_repo.create_template(
                user_id="GHOST_USER",
                biometric_type=BiometricType.PALM,
                feature_data=dummy_feature,
                feature_format=FeatureFormat.GABOR_VECTOR,
                feature_dimension=64,
                model_version="palm-gabor-v1"
            )

    def test_cascading_delete_removes_templates(self, user_repo, template_repo):
        """Deleting a user must automatically delete all their biometric templates."""
        user_repo.create_user("NV005", "User To Delete")
        template_repo.create_template(
            user_id="NV005",
            biometric_type=BiometricType.FACE,
            feature_data=b"\x01" * 32,
            feature_format=FeatureFormat.FLOAT32_EMBEDDING,
            feature_dimension=32,
            model_version="v1"
        )
        template_repo.create_template(
            user_id="NV005",
            biometric_type=BiometricType.PALM,
            feature_data=b"\x02" * 32,
            feature_format=FeatureFormat.GABOR_VECTOR,
            feature_dimension=32,
            model_version="v1"
        )

        assert template_repo.count_templates(user_id="NV005") == 2

        # Delete user
        user_repo.delete_user("NV005")

        # Verify cascading deletion
        assert template_repo.count_templates(user_id="NV005") == 0


class TestAccessLogRepository:
    """Test suite covering audit logging, UTC timestamps, and ON DELETE SET NULL."""

    def test_log_access_and_retrieve(self, user_repo, log_repo):
        # Create user first so foreign key is satisfied
        user_repo.create_user("NV001", "Nguyen Van A")

        log_id = log_repo.log_event(
            request_id="req_test_001",
            user_id="NV001",
            auth_method=AuthMethod.FACE,
            decision=AccessDecision.GRANTED,
            actuation_status=ActuationStatus.ACTUATED,
            reason="Face verification matched"
        )
        assert log_id > 0

        # Also test logging for an unauthenticated / anonymous stranger (user_id=None)
        log_id_anon = log_repo.log_event(
            request_id="req_test_anon",
            user_id=None,
            auth_method=AuthMethod.FACE,
            decision=AccessDecision.DENIED,
            actuation_status=ActuationStatus.NOT_ACTUATED,
            reason="Unrecognized face"
        )
        assert log_id_anon > log_id

        logs = log_repo.get_recent_logs(limit=5)
        assert len(logs) == 2
        record = logs[1]  # Older record
        assert record.request_id == "req_test_001"
        assert record.user_id == "NV001"
        assert record.decision == AccessDecision.GRANTED
        assert record.actuation_status == ActuationStatus.ACTUATED
        assert record.reason == "Face verification matched"
        assert record.timestamp is not None

    def test_user_deletion_sets_log_user_id_to_null(self, user_repo, log_repo):
        """Deleting user preserves historical audit trail while setting user_id to NULL."""
        user_repo.create_user("AUDIT_USER", "Audit Target")
        log_repo.log_event(
            request_id="req_audit_01",
            user_id="AUDIT_USER",
            auth_method=AuthMethod.DUAL,
            decision=AccessDecision.GRANTED,
            actuation_status=ActuationStatus.ACTUATED,
            reason="Dual auth successful"
        )

        # Delete user
        user_repo.delete_user("AUDIT_USER")

        logs = log_repo.get_recent_logs(limit=5)
        assert len(logs) == 1
        assert logs[0].request_id == "req_audit_01"
        # user_id must now be NULL (preserving audit history)
        assert logs[0].user_id is None
        assert logs[0].decision == AccessDecision.GRANTED

def test_face_only_does_not_record_hand_side(tmp_path):
    db = DatabaseConnection(str(tmp_path / "face.sqlite3"))
    db.init_db()
    logs = AccessLogRepository(db)
    logs.log_event("face", AuthMethod.FACE, AccessDecision.DENIED,
                   ActuationStatus.NOT_ACTUATED, "No match", hand_side="RIGHT")
    assert logs.get_recent_logs()[0].hand_side is None
