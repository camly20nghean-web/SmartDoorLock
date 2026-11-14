"""Unit and integration tests for FaceRecognitionEngine, quality assessment, and AccessControl."""
import math
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.biometrics.face import (
    DEFAULT_FEATURE_DIMENSION,
    DEFAULT_MODEL_VERSION,
    FaceQualityAssessment,
    FaceRecognitionEngine,
)
from src.database.connection import DatabaseConnection
from src.database.models import (
    AccessDecision,
    ActuationStatus,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricType,
    FeatureFormat,
    UserStatus,
)
from src.database.repository import (
    AccessLogRepository,
    BiometricTemplateRepository,
    UserRepository,
)
from src.hardware.base import LockState
from src.hardware.mock_lock import MockDoorLock
from src.services.access_control import AccessControlService


class TestCosineSimilarityMath:
    """Rigorous mathematical tests for Cosine Similarity computation."""

    @classmethod
    def setup_class(cls):
        cls.engine = FaceRecognitionEngine(similarity_threshold=0.60)

    def test_identical_vectors_yield_one(self):
        vec = np.random.randn(128).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        score = self.engine.compute_similarity(vec, vec)
        assert math.isclose(score, 1.0, rel_tol=1e-5)

    def test_orthogonal_vectors_yield_zero(self):
        vec1 = np.zeros(128, dtype=np.float32)
        vec1[0] = 1.0

        vec2 = np.zeros(128, dtype=np.float32)
        vec2[1] = 1.0

        score = self.engine.compute_similarity(vec1, vec2)
        assert math.isclose(score, 0.0, abs_tol=1e-5)

    def test_opposite_vectors_yield_minus_one(self):
        vec = np.random.randn(128).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        neg_vec = -vec

        score = self.engine.compute_similarity(vec, neg_vec)
        assert math.isclose(score, -1.0, rel_tol=1e-5)

    def test_dimension_mismatch_returns_zero(self):
        vec128 = np.ones(128, dtype=np.float32)
        vec512 = np.ones(512, dtype=np.float32)

        score = self.engine.compute_similarity(vec128, vec512)
        assert score == 0.0

    def test_nan_and_inf_handling_returns_zero(self):
        vec_nan = np.ones(128, dtype=np.float32)
        vec_nan[5] = np.nan

        vec_inf = np.ones(128, dtype=np.float32)
        vec_inf[10] = np.inf

        vec_valid = np.ones(128, dtype=np.float32)

        assert self.engine.compute_similarity(vec_nan, vec_valid) == 0.0
        assert self.engine.compute_similarity(vec_valid, vec_inf) == 0.0

    def test_none_or_empty_inputs_return_zero(self):
        vec = np.ones(128, dtype=np.float32)
        assert self.engine.compute_similarity(None, vec) == 0.0
        assert self.engine.compute_similarity(vec, None) == 0.0


class TestFaceQualityAssessment:
    """Test facial image quality metrics (illumination, blur, resolution)."""

    @classmethod
    def setup_class(cls):
        cls.engine = FaceRecognitionEngine()

    def test_acceptable_quality_frame(self):
        # Create a textured 100x100 face region inside a 200x200 image
        image = np.random.randint(80, 160, (200, 200, 3), dtype=np.uint8)
        face_box = np.array([50, 50, 100, 100], dtype=np.float32)

        quality = self.engine.assess_face_quality(image, face_box)
        assert quality.is_acceptable is True
        assert 35.0 <= quality.brightness <= 230.0
        assert quality.sharpness >= 40.0

    def test_too_dark_illumination_rejected(self):
        # Dark image
        dark_image = np.ones((200, 200, 3), dtype=np.uint8) * 15
        face_box = np.array([20, 20, 80, 80], dtype=np.float32)

        quality = self.engine.assess_face_quality(dark_image, face_box)
        assert quality.is_acceptable is False
        assert "dark" in quality.rejection_reason

    def test_overexposed_illumination_rejected(self):
        bright_image = np.ones((200, 200, 3), dtype=np.uint8) * 250
        face_box = np.array([20, 20, 80, 80], dtype=np.float32)

        quality = self.engine.assess_face_quality(bright_image, face_box)
        assert quality.is_acceptable is False
        assert "overexposed" in quality.rejection_reason

    def test_blurry_image_rejected(self):
        # Uniform flat image has zero variance (completely blurry)
        flat_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        face_box = np.array([20, 20, 80, 80], dtype=np.float32)

        quality = self.engine.assess_face_quality(flat_image, face_box)
        assert quality.is_acceptable is False
        assert "blurry" in quality.rejection_reason

    def test_small_resolution_rejected(self):
        # Face box width 30px < 60px minimum
        image = np.random.randint(80, 160, (100, 100, 3), dtype=np.uint8)
        small_box = np.array([10, 10, 30, 30], dtype=np.float32)

        quality = self.engine.assess_face_quality(image, small_box)
        assert quality.is_acceptable is False
        assert "small" in quality.rejection_reason


class TestFaceDetectorConstraints:
    """Test face count constraints: 0 faces, 1 face, and multiple faces."""

    @patch.object(FaceRecognitionEngine, "detect_all_faces")
    def test_zero_faces_detected(self, mock_detect):
        mock_detect.return_value = np.empty((0, 15), dtype=np.float32)
        engine = FaceRecognitionEngine()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        face_box, reason = engine.detect_single_face(dummy_img)
        assert face_box is None
        assert "No face detected" in reason

    @patch.object(FaceRecognitionEngine, "detect_all_faces")
    def test_multiple_faces_detected_strictly_rejected(self, mock_detect):
        # 2 faces detected
        mock_detect.return_value = np.zeros((2, 15), dtype=np.float32)
        engine = FaceRecognitionEngine()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        face_box, reason = engine.detect_single_face(dummy_img)
        assert face_box is None
        assert "Multiple faces" in reason

    @patch.object(FaceRecognitionEngine, "detect_all_faces")
    def test_single_face_accepted(self, mock_detect):
        # Exactly 1 face detected
        mock_detect.return_value = np.array([[10, 10, 80, 80] + [0]*11], dtype=np.float32)
        engine = FaceRecognitionEngine()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        face_box, reason = engine.detect_single_face(dummy_img)
        assert face_box is not None
        assert reason == "OK"


class TestDatabaseAndAccessControlIntegration:
    """Integration test connecting SFace templates in SQLite to AccessControlService."""

    @pytest.fixture
    def test_env(self):
        db = DatabaseConnection(db_path=":memory:")
        db.init_db()

        user_repo = UserRepository(db)
        template_repo = BiometricTemplateRepository(db)
        log_repo = AccessLogRepository(db)
        lock = MockDoorLock(default_auto_relock_seconds=2.0)

        # Seed users
        user_repo.create_user("NV001", "Active User", status=UserStatus.ACTIVE)
        user_repo.create_user("NV002", "Inactive User", status=UserStatus.INACTIVE)

        # Generate a synthetic 128-D SFace embedding
        enrolled_vec = np.random.randn(128).astype(np.float32)
        enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)

        # Save to SQLite
        template_repo.create_template(
            user_id="NV001",
            biometric_type=BiometricType.FACE,
            feature_data=enrolled_vec.tobytes(),
            feature_format=FeatureFormat.FLOAT32_EMBEDDING,
            feature_dimension=DEFAULT_FEATURE_DIMENSION,
            model_version=DEFAULT_MODEL_VERSION
        )

        service = AccessControlService(
            user_repo=user_repo,
            log_repo=log_repo,
            lock=lock,
            auto_relock_seconds=2.0,
            is_test_or_demo_mode=True
        )

        return service, lock, user_repo, template_repo, log_repo, enrolled_vec

    def test_template_stored_with_accurate_metadata(self, test_env):
        _, _, _, template_repo, _, _ = test_env

        templates = template_repo.get_templates_by_user("NV001", biometric_type=BiometricType.FACE)
        assert len(templates) == 1
        tmpl = templates[0]

        assert tmpl.feature_format == FeatureFormat.FLOAT32_EMBEDDING
        assert tmpl.feature_dimension == 128
        assert tmpl.model_version == "opencv_sface_2021dec"

        # Deserialize back to numpy float32
        recovered = np.frombuffer(tmpl.feature_data, dtype=np.float32)
        assert recovered.shape == (128,)

    def test_active_user_high_similarity_grants_access(self, test_env):
        service, lock, _, _, log_repo, _ = test_env

        # Create verified claim with high similarity (0.85 > 0.60)
        claim = BiometricAuthClaim(
            user_id="NV001",
            biometric_type=BiometricType.FACE,
            is_matched=True,
            similarity_score=0.85,
            session_id="test_sess_01",
            timestamp=time.time(),
            is_trusted_source=True
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)
        assert outcome.is_granted is True
        assert outcome.decision == AccessDecision.GRANTED
        assert outcome.actuation_status == ActuationStatus.ACTUATED
        assert lock.get_state() == LockState.UNLOCKED

        logs = log_repo.get_recent_logs(limit=1)
        assert logs[0].decision == AccessDecision.GRANTED
        lock.lock()

    def test_inactive_user_rejected_despite_high_score(self, test_env):
        service, lock, _, _, _, _ = test_env

        claim = BiometricAuthClaim(
            user_id="NV002",  # Inactive user
            biometric_type=BiometricType.FACE,
            is_matched=True,
            similarity_score=0.95,
            session_id="test_sess_02",
            timestamp=time.time(),
            is_trusted_source=True
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)
        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert lock.get_state() == LockState.LOCKED

    def test_empty_database_zero_templates_strictly_denied(self):
        """Verify that an empty database with 0 enrolled templates strictly denies any detected face."""
        db = DatabaseConnection(":memory:")
        db.init_db()
        user_repo = UserRepository(db)
        log_repo = AccessLogRepository(db)
        lock = MockDoorLock()
        service = AccessControlService(
            user_repo=user_repo,
            log_repo=log_repo,
            lock=lock,
            is_test_or_demo_mode=True
        )

        # Simulating live search with empty cached_templates
        cached_templates = []
        probe_vector = np.random.randn(128).astype(np.float32)
        probe_vector = probe_vector / np.linalg.norm(probe_vector)

        best_score = -1.0
        best_user_id = None
        for uid, _, _, tmpl_vec in cached_templates:
            score = float(np.dot(probe_vector, tmpl_vec))
            if score > best_score:
                best_score = score
                best_user_id = uid

        # No match found
        assert best_user_id is None
        assert best_score == -1.0

        claim = BiometricAuthClaim(
            user_id=best_user_id,
            biometric_type=BiometricType.FACE,
            is_matched=False,
            similarity_score=max(0.0, best_score),
            session_id="unreg_test_sess",
            timestamp=time.time(),
            is_trusted_source=True
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)
        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert outcome.actuation_status == ActuationStatus.NOT_ACTUATED
        assert lock.get_state() == LockState.LOCKED

        logs = log_repo.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].decision == AccessDecision.DENIED
        assert logs[0].user_id is None
        assert logs[0].actuation_status == ActuationStatus.NOT_ACTUATED

        assert "INACTIVE" in outcome.reason

    def test_stranger_low_score_rejected(self, test_env):
        service, lock, _, _, _, _ = test_env

        claim = BiometricAuthClaim(
            user_id=None,
            biometric_type=BiometricType.FACE,
            is_matched=False,
            similarity_score=0.25,
            session_id="test_sess_03",
            timestamp=time.time(),
            is_trusted_source=True
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)
        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert lock.get_state() == LockState.LOCKED

    def test_empty_database_zero_templates_strictly_denied(self):
        """Verify that an empty database with 0 enrolled templates strictly denies any detected face."""
        db = DatabaseConnection(":memory:")
        db.init_db()
        user_repo = UserRepository(db)
        log_repo = AccessLogRepository(db)
        lock = MockDoorLock()
        service = AccessControlService(
            user_repo=user_repo,
            log_repo=log_repo,
            lock=lock,
            is_test_or_demo_mode=True
        )

        # Simulating live search with empty cached_templates
        cached_templates = []
        probe_vector = np.random.randn(128).astype(np.float32)
        probe_vector = probe_vector / np.linalg.norm(probe_vector)

        best_score = -1.0
        best_user_id = None
        for uid, _, _, tmpl_vec in cached_templates:
            score = float(np.dot(probe_vector, tmpl_vec))
            if score > best_score:
                best_score = score
                best_user_id = uid

        # No match found
        assert best_user_id is None
        assert best_score == -1.0

        claim = BiometricAuthClaim(
            user_id=best_user_id,
            biometric_type=BiometricType.FACE,
            is_matched=False,
            similarity_score=max(0.0, best_score),
            session_id="unreg_test_sess",
            timestamp=time.time(),
            is_trusted_source=True
        )

        outcome = service.process_access_request(policy=AuthPolicy.FACE_ONLY, face_claim=claim)
        assert outcome.is_granted is False
        assert outcome.decision == AccessDecision.DENIED
        assert outcome.actuation_status == ActuationStatus.NOT_ACTUATED
        assert lock.get_state() == LockState.LOCKED

        logs = log_repo.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].decision == AccessDecision.DENIED
        assert logs[0].user_id is None
        assert logs[0].actuation_status == ActuationStatus.NOT_ACTUATED


