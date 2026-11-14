"""Data models and enumerations for entities, policies, and audit logs."""
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Optional
import uuid


def get_utc_now_iso() -> str:
    """Return current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    """Generate unique tracking UUID string for each access request."""
    return f"req_{uuid.uuid4().hex[:12]}"


class UserStatus(str, Enum):
    """User account privilege and active status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class BiometricType(str, Enum):
    """Supported biometric modality."""
    FACE = "FACE"
    PALM = "PALM"


class FeatureFormat(str, Enum):
    """Serialization representation of biometric feature vectors."""
    FLOAT32_EMBEDDING = "FLOAT32_EMBEDDING"
    GABOR_VECTOR = "GABOR_VECTOR"
    BINARY_CODE = "BINARY_CODE"


class AuthMethod(str, Enum):
    """Authentication channel or mechanism."""
    FACE = "FACE"
    PALM = "PALM"
    DUAL = "DUAL"
    MANUAL = "MANUAL"


class AccessDecision(str, Enum):
    """Authorization determination emitted by policy engine."""
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class ActuationStatus(str, Enum):
    """Physical hardware / mock lock trigger result."""
    ACTUATED = "ACTUATED"
    NOT_ACTUATED = "NOT_ACTUATED"
    FAILED = "FAILED"


class AuthPolicy(str, Enum):
    """Multi-factor biometric access policy."""
    FACE_ONLY = "FACE_ONLY"
    PALM_ONLY = "PALM_ONLY"
    FACE_AND_PALM = "FACE_AND_PALM"


@dataclass
class UserRecord:
    """User profile record mapped to the 'users' table."""
    user_id: str
    full_name: str
    status: UserStatus = UserStatus.ACTIVE
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """True if user account has active authorization privileges."""
        return self.status == UserStatus.ACTIVE


@dataclass
class BiometricTemplateRecord:
    """Biometric feature template mapped to 'biometric_templates'."""
    user_id: str
    biometric_type: BiometricType
    feature_data: bytes
    feature_format: FeatureFormat
    feature_dimension: int
    model_version: str
    created_at: Optional[str] = None
    template_id: Optional[int] = None


@dataclass
class AccessLogRecord:
    """Audit trail record mapped to 'access_logs'."""
    request_id: str
    auth_method: AuthMethod
    decision: AccessDecision
    actuation_status: ActuationStatus
    reason: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    log_id: Optional[int] = None
    hand_side: Optional[str] = None


@dataclass(frozen=True)
class BiometricAuthClaim:
    """
    Biometric verification result presented to the access control policy engine.
    
    Security note:
    Only claims originating from an authorized, trusted internal pipeline
    (is_trusted_source=True) within an active session will be evaluated for access.
    """
    user_id: Optional[str]
    biometric_type: BiometricType
    is_matched: bool
    similarity_score: float
    session_id: str
    timestamp: float
    is_trusted_source: bool = False

    def is_expired(self, max_age_seconds: float = 30.0) -> bool:
        """Check if the claim timestamp has exceeded the maximum validity window."""
        return (time.time() - self.timestamp) > max_age_seconds


@dataclass(frozen=True)
class AuthorizationOutcome:
    """Comprehensive result of an authorization request evaluation."""
    request_id: str
    decision: AccessDecision
    actuation_status: ActuationStatus
    user_id: Optional[str]
    policy: AuthPolicy
    reason: str

    @property
    def is_granted(self) -> bool:
        return self.decision == AccessDecision.GRANTED
