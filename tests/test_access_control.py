"""Unit and integration tests for AccessControlService, policies, and security safeguards."""
import time
import pytest

from src.database.connection import DatabaseConnection
from src.database.models import (
    AccessDecision,
    ActuationStatus,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricType,
    UserStatus,
)
from src.database.repository import AccessLogRepository, UserRepository
from src.hardware.base import LockState
from src.hardware.mock_lock import MockDoorLock
from src.services.access_control import AccessControlService


@pytest.fixture
def service_env():
    """Build isolated AccessControlService environment with in-memory DB and test lock."""
    db = DatabaseConnection(db_path=":memory:")
    db.init_db()

    user_repo = UserRepository(db)
    log_repo = AccessLogRepository(db)
    lock = MockDoorLock(default_auto_relock_seconds=2.0)

    # Seed test users
    user_repo.create_user("NV001", "Active Employee", status=UserStatus.ACTIVE)
    user_repo.create_user("NV002", "Inactive Employee", status=UserStatus.INACTIVE)
    user_repo.create_user("NV003", "Suspended Employee", status=UserStatus.SUSPENDED)
    user_repo.create_user("NV999", "Another Active Employee", status=UserStatus.ACTIVE)

    service = AccessControlService(
        user_repo=user_repo,
        log_repo=log_repo,
        lock=lock,
        auto_relock_seconds=2.0,
        claim_max_age_seconds=10.0,
        is_test_or_demo_mode=True
    )

    return service, lock, user_repo, log_repo


def make_claim(
    user_id: str,
    biometric_type: BiometricType,
    is_matched: bool = True,
    score: float = 0.88,
    session_id: str = "sess_001",
    timestamp: float = None,
    is_trusted_source: bool = True
) -> BiometricAuthClaim:
    """Helper creating a test BiometricAuthClaim."""
    return BiometricAuthClaim(
        user_id=user_id,
        biometric_type=biometric_type,
        is_matched=is_matched,
        similarity_score=score,
        session_id=session_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        is_trusted_source=is_trusted_source
    )


class TestAccessControlPolicies:
    """Test suite covering FACE_ONLY, PALM_ONLY, and dual FACE_AND_PALM authorization policies."""

    def test_face_only_policy_success(self, service_env):
        service, lock, _, log_repo = service_env
        claim = make_claim("NV001", BiometricType.FACE)

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)

        assert outcome.is_granted is True
        assert outcome.decision == AccessDecision.GRANTED
        assert outcome.actuation_status == ActuationStatus.ACTUATED
        assert lock.get_state() == LockState.UNLOCKED

        logs = log_repo.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].decision == AccessDecision.GRANTED
        assert logs[0].user_id == "NV001"
        lock.lock()

    def test_palm_only_policy_success(self, service_env):
        service, lock, _, log_repo = service_env
        claim = make_claim("NV001", BiometricType.PALM)

        outcome = service.process_access_request(policy=AuthPolicy.PALM_ONLY, palm_claim=claim)

        assert outcome.is_granted is True
        assert outcome.decision == AccessDecision.GRANTED
        assert outcome.actuation_status == ActuationStatus.ACTUATED
        lock.lock()

    def test_dual_face_and_palm_policy_success(self, service_env):
        service, lock, _, log_repo = service_env
        shared_session = "dual_session_101"
        face_claim = make_claim("NV001", BiometricType.FACE, session_id=shared_session)
        palm_claim = make_claim("NV001", BiometricType.PALM, session_id=shared_session)

        outcome = service.process_access_request(
            policy=AuthPolicy.FACE_AND_PALM,
            face_claim=face_claim,
            palm_claim=palm_claim
        )

        assert outcome.is_granted is True
        assert outcome.decision == AccessDecision.GRANTED
        assert outcome.actuation_status == ActuationStatus.ACTUATED
        lock.lock()

    def test_dual_policy_identity_divergence_rejected(self, service_env):
        """Face belonging to User A and Palm belonging to User B must be strictly DENIED."""
        service, lock, _, log_repo = service_env
        shared_session = "dual_session_hack"
        face_claim = make_claim("NV001", BiometricType.FACE, session_id=shared_session)
        palm_claim = make_claim("NV999", BiometricType.PALM, session_id=shared_session)

        outcome = service.process_access_request(
            policy=AuthPolicy.FACE_AND_PALM,
            face_claim=face_claim,
            palm_claim=palm_claim
        )

        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert outcome.actuation_status == ActuationStatus.NOT_ACTUATED
        assert lock.get_state() == LockState.LOCKED
        assert "Identity divergence" in outcome.reason

    def test_dual_policy_session_divergence_rejected(self, service_env):
        """Claims from different verification sessions must be strictly DENIED."""
        service, lock, _, log_repo = service_env
        face_claim = make_claim("NV001", BiometricType.FACE, session_id="session_morning")
        palm_claim = make_claim("NV001", BiometricType.PALM, session_id="session_afternoon")

        outcome = service.process_access_request(
            policy=AuthPolicy.FACE_AND_PALM,
            face_claim=face_claim,
            palm_claim=palm_claim
        )

        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert "Session divergence" in outcome.reason


class TestSecuritySafeguardsAndAccountStatus:
    """Test suite covering untrusted inputs, account status invariance, and expired claims."""

    def test_untrusted_biometric_source_rejected(self, service_env):
        """Claims from an unverified or forged pipeline (is_trusted_source=False) must be DENIED."""
        service, lock, _, _ = service_env
        untrusted_claim = make_claim("NV001", BiometricType.FACE, is_trusted_source=False)

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=untrusted_claim)

        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert "not from an authenticated trusted pipeline" in outcome.reason
        assert lock.get_state() == LockState.LOCKED

    def test_expired_claim_rejected(self, service_env):
        """Claims older than max_age_seconds must be DENIED."""
        service, lock, _, _ = service_env
        expired_claim = make_claim(
            "NV001",
            BiometricType.FACE,
            timestamp=time.time() - 60.0  # 60s ago > 10s max age
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=expired_claim)

        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert "expired" in outcome.reason

    def test_inactive_and_suspended_users_denied(self, service_env):
        service, lock, _, _ = service_env

        # Inactive user
        inactive_claim = make_claim("NV002", BiometricType.FACE)
        outcome_inactive = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=inactive_claim)
        assert outcome_inactive.is_granted is False
        assert outcome_inactive.decision == AccessDecision.DENIED
        assert "INACTIVE" in outcome_inactive.reason

        # Suspended user
        suspended_claim = make_claim("NV003", BiometricType.FACE)
        outcome_suspended = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=suspended_claim)
        assert outcome_suspended.is_granted is False
        assert outcome_suspended.decision == AccessDecision.DENIED
        assert "SUSPENDED" in outcome_suspended.reason

    def test_actuation_suppressed_when_not_in_test_or_demo_mode(self, service_env):
        """When is_test_or_demo_mode is False, authorization is granted but physical actuation is suppressed."""
        service, lock, _, _ = service_env
        service.is_test_or_demo_mode = False

        claim = make_claim("NV001", BiometricType.FACE)
        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)

        assert outcome.is_granted is True
        assert outcome.decision == AccessDecision.GRANTED
        assert outcome.actuation_status == ActuationStatus.NOT_ACTUATED
        assert lock.get_state() == LockState.LOCKED  # Lock was not physically fired!

