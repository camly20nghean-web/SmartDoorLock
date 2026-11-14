"""Central Access Control Service decoupling authorization policy from lock actuation."""
import logging
from typing import Optional

from src.database.models import (
    AccessDecision,
    ActuationStatus,
    AuthMethod,
    AuthorizationOutcome,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricType,
    UserStatus,
    generate_request_id,
)
from src.database.repository import AccessLogRepository, UserRepository
from src.hardware.base import BaseDoorLock

logger = logging.getLogger(__name__)


class AccessControlService:
    """
    Decoupled authorization policy engine.
    
    Security Principles:
    1. Separation of Concerns: Decouples authorization determination from physical hardware actuation.
    2. Zero Trust Input: Rejects claims from untrusted sources, expired sessions, or session mismatches.
    3. Status Invariance: Deactivated or suspended users are denied regardless of match scores.
    4. Controlled Actuation: Physical actuation is isolated and only executed in verified environments.
    """

    def __init__(
        self,
        user_repo: UserRepository,
        log_repo: AccessLogRepository,
        lock: Optional[BaseDoorLock] = None,
        auto_relock_seconds: float = 5.0,
        claim_max_age_seconds: float = 30.0,
        is_test_or_demo_mode: bool = False
    ) -> None:
        self.user_repo = user_repo
        self.log_repo = log_repo
        self.lock = lock
        self.auto_relock_seconds = auto_relock_seconds
        self.claim_max_age_seconds = claim_max_age_seconds
        self.is_test_or_demo_mode = is_test_or_demo_mode

    def evaluate_authorization(
        self,
        policy: AuthPolicy,
        face_claim: Optional[BiometricAuthClaim] = None,
        palm_claim: Optional[BiometricAuthClaim] = None
    ) -> tuple[AccessDecision, Optional[str], str]:
        """
        Evaluate biometric claims against the configured authorization policy.
        
        Returns:
            tuple[AccessDecision, Optional[str], str]: (decision, resolved_user_id, reason)
        """
        if policy == AuthPolicy.FACE_ONLY:
            return self._evaluate_single_claim(face_claim, expected_type=BiometricType.FACE)

        if policy == AuthPolicy.PALM_ONLY:
            return self._evaluate_single_claim(palm_claim, expected_type=BiometricType.PALM)

        if policy == AuthPolicy.FACE_AND_PALM:
            return self._evaluate_dual_claims(face_claim, palm_claim)

        return AccessDecision.DENIED, None, f"Unsupported authorization policy: {policy}"

    def process_access_request(
        self,
        policy: AuthPolicy,
        face_claim: Optional[BiometricAuthClaim] = None,
        palm_claim: Optional[BiometricAuthClaim] = None,
        request_id: Optional[str] = None
    ) -> AuthorizationOutcome:
        """
        Execute full access control workflow: evaluate policy, record audit log, and optionally actuate lock.
        
        Returns:
            AuthorizationOutcome: Complete audit-ready outcome record.
        """
        req_id = request_id or generate_request_id()
        decision, user_id, reason = self.evaluate_authorization(policy, face_claim, palm_claim)

        # Map policy to audit auth_method
        auth_method_map = {
            AuthPolicy.FACE_ONLY: AuthMethod.FACE,
            AuthPolicy.PALM_ONLY: AuthMethod.PALM,
            AuthPolicy.FACE_AND_PALM: AuthMethod.DUAL,
        }
        auth_method = auth_method_map.get(policy, AuthMethod.MANUAL)

        actuation_status = ActuationStatus.NOT_ACTUATED

        # Controlled actuation: Only trigger lock if access GRANTED and demo/test mode is explicitly active
        if decision == AccessDecision.GRANTED and self.lock is not None:
            if self.is_test_or_demo_mode:
                try:
                    unlock_ok = self.lock.unlock(duration_seconds=self.auto_relock_seconds)
                    actuation_status = ActuationStatus.ACTUATED if unlock_ok else ActuationStatus.FAILED
                except Exception as e:
                    logger.error("[AccessControl] Exception during lock actuation: %s", e)
                    actuation_status = ActuationStatus.FAILED
            else:
                logger.warning(
                    "[AccessControl] Actuation suppressed: is_test_or_demo_mode is False. "
                    "In production Phase 6, actuation will be delegated to verified hardware bridge."
                )

        # Log event to database (Audit Trail)
        self.log_repo.log_event(
            request_id=req_id,
            user_id=user_id,
            auth_method=auth_method,
            decision=decision,
            actuation_status=actuation_status,
            reason=reason
        )

        return AuthorizationOutcome(
            request_id=req_id,
            decision=decision,
            actuation_status=actuation_status,
            user_id=user_id,
            policy=policy,
            reason=reason
        )

    def _evaluate_single_claim(
        self,
        claim: Optional[BiometricAuthClaim],
        expected_type: BiometricType
    ) -> tuple[AccessDecision, Optional[str], str]:
        """Validate a single biometric claim against security criteria."""
        if claim is None:
            return AccessDecision.DENIED, None, f"Missing required {expected_type.value} claim."

        if claim.biometric_type != expected_type:
            return AccessDecision.DENIED, claim.user_id, f"Mismatched modality: expected {expected_type.value}."

        if not claim.is_trusted_source:
            return AccessDecision.DENIED, claim.user_id, "Rejected: Claim is not from an authenticated trusted pipeline."

        if claim.is_expired(self.claim_max_age_seconds):
            return AccessDecision.DENIED, claim.user_id, "Rejected: Biometric claim timestamp has expired."

        if not claim.is_matched or not claim.user_id:
            return AccessDecision.DENIED, claim.user_id, f"{expected_type.value} verification score rejected."

        # Verify user account validity in repository
        user = self.user_repo.get_user(claim.user_id)
        if not user:
            return AccessDecision.DENIED, claim.user_id, f"User '{claim.user_id}' does not exist in database."

        if user.status != UserStatus.ACTIVE:
            return AccessDecision.DENIED, claim.user_id, f"User '{claim.user_id}' account is {user.status.value}."

        return AccessDecision.GRANTED, user.user_id, f"{expected_type.value} verification successful."

    def _evaluate_dual_claims(
        self,
        face_claim: Optional[BiometricAuthClaim],
        palm_claim: Optional[BiometricAuthClaim]
    ) -> tuple[AccessDecision, Optional[str], str]:
        """Evaluate dual biometric authentication (FACE_AND_PALM)."""
        # Validate face claim first
        face_dec, face_uid, face_reason = self._evaluate_single_claim(face_claim, BiometricType.FACE)
        if face_dec != AccessDecision.GRANTED:
            return AccessDecision.DENIED, face_uid, f"FACE check failed: {face_reason}"

        # Validate palm claim
        palm_dec, palm_uid, palm_reason = self._evaluate_single_claim(palm_claim, BiometricType.PALM)
        if palm_dec != AccessDecision.GRANTED:
            return AccessDecision.DENIED, palm_uid, f"PALM check failed: {palm_reason}"

        assert face_claim is not None and palm_claim is not None

        # Cross-validation: Both claims MUST identify the exact same individual
        if face_claim.user_id != palm_claim.user_id:
            return (
                AccessDecision.DENIED,
                face_claim.user_id,
                f"Identity divergence: Face ({face_claim.user_id}) != Palm ({palm_claim.user_id})."
            )

        # Cross-validation: Both claims MUST belong to the exact same verification session
        if face_claim.session_id != palm_claim.session_id:
            return (
                AccessDecision.DENIED,
                face_claim.user_id,
                f"Session divergence: Face session ({face_claim.session_id}) != Palm session ({palm_claim.session_id})."
            )

        return (
            AccessDecision.GRANTED,
            face_claim.user_id,
            "Dual verification (Face + Palmprint) successful."
        )

