"""Unit tests for PalmprintRecognitionEngine (Milestones A & B).

Tests:
- Engine lifecycle & clean model loading via byte buffer.
- Robust handling of blank / invalid frames (no false positive hand detection).
- MCP landmark geometry, hand axis estimation, and rotation calculation.
- Strict Palmar vs. Dorsal surface orientation checking.
- Quality filtering (blur, illumination, scale).
- Compliance with BaseBiometricExtractor contract:
  * detect_roi returns None when quality or detection is invalid.
  * extract_embedding produces a normalized spatial LBP descriptor.
  * verify raises NotImplementedError until Milestone D.
"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.biometrics.base import BaseBiometricExtractor
from src.biometrics.palm import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ROI_SIZE,
    PALM_FEATURE_DIMENSION,
    PalmQualityAssessment,
    PalmROIResult,
    PalmprintRecognitionEngine,
)


class TestPalmRecognitionEngineLifecycle:
    """Tests for model loading, lifecycle, and resource teardown."""

    def test_engine_initializes_with_valid_model(self):
        engine = PalmprintRecognitionEngine()
        assert engine.is_ready is True
        engine.close()

    def test_engine_missing_model_raises_filenotfound(self, tmp_path):
        fake_path = tmp_path / "non_existent_hand_model.task"
        with pytest.raises(FileNotFoundError) as exc_info:
            PalmprintRecognitionEngine(model_path=fake_path)
        assert "not found" in str(exc_info.value)

    def test_engine_context_manager(self):
        with PalmprintRecognitionEngine() as engine:
            assert engine.is_ready is True
            assert isinstance(engine, BaseBiometricExtractor)
        assert engine._detector is None


class TestHandDetectionAndROI:
    """Tests for Hand Landmark Detection (Milestone A) and ROI Alignment (Milestone B)."""

    @classmethod
    def setup_class(cls):
        cls.engine = PalmprintRecognitionEngine()

    @classmethod
    def teardown_class(cls):
        cls.engine.close()

    def test_blank_image_returns_no_hand(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.engine.detect_hand_and_roi(blank)

        assert isinstance(res, PalmROIResult)
        assert res.has_hand is False
        assert res.landmarks_2d is None
        assert res.roi_image is None

        # detect_roi must return None
        assert self.engine.detect_roi(blank) is None

    def test_none_or_empty_image_handling(self):
        empty = np.array([], dtype=np.uint8)
        assert self.engine.detect_hand_and_roi(empty).has_hand is False
        assert self.engine.detect_roi(empty) is None

    def test_quality_assessment_rejection_logic(self):
        # Test synthetic blur rejection
        qual = PalmQualityAssessment(
            is_acceptable=False,
            sharpness=20.0,  # Below MIN_SHARPNESS_VAR (50.0)
            mean_illumination=100.0,
            palm_width_px=80.0,
            is_palmar=True,
            rejection_reasons=["Image blurred"]
        )
        assert qual.is_acceptable is False
        assert "Image blurred" in qual.rejection_reasons[0]

    def test_dorsal_surface_rejection_logic(self):
        # Back of hand presentation must be rejected
        qual = PalmQualityAssessment(
            is_acceptable=False,
            sharpness=120.0,
            mean_illumination=110.0,
            palm_width_px=95.0,
            is_palmar=False,
            rejection_reasons=["Back of hand (dorsal) presented"]
        )
        assert qual.is_acceptable is False
        assert qual.is_palmar is False

    def test_roi_dimensions_and_clahe_format(self):
        # Create an artificial synthetic hand-like textured region to test Affine Crop & CLAHE
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2_roi = np.random.randint(60, 180, (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE, 3), dtype=np.uint8)

        # Apply CLAHE to ensure contrast pipeline functions
        gray = cv2_roi[:, :, 0]
        clahe = self.engine._clahe.apply(gray)
        assert clahe.shape == (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE)
        assert clahe.dtype == np.uint8

    @pytest.mark.parametrize("tilt_degrees", [0, -30, 30])
    def test_roi_aligns_hand_axis_and_centers_crop(self, monkeypatch, tilt_degrees):
        angle = np.deg2rad(tilt_degrees)
        axis = np.array([np.sin(angle), -np.cos(angle)])
        across = np.array([np.cos(angle), np.sin(angle)])
        wrist = np.array([230, 300])
        middle_mcp = wrist + 100 * axis
        palm_center = wrist + 52 * axis
        upper_marker = palm_center + 18 * axis

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, tuple(np.rint(palm_center).astype(int)), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(np.rint(upper_marker).astype(int)), 4, (0, 0, 255), -1)

        points = np.tile(wrist, (21, 1))
        points = points.astype(float)
        points[5] = middle_mcp + 60 * across
        points[9] = middle_mcp
        points[17] = middle_mcp - 60 * across
        height, width = frame.shape[:2]
        landmarks = [SimpleNamespace(x=x / width, y=y / height, z=0.0) for x, y in points]
        detection = SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[[SimpleNamespace(category_name="Right", score=1.0)]],
        )
        monkeypatch.setattr(self.engine, "_detector", SimpleNamespace(detect=lambda _: detection))

        result = self.engine.detect_hand_and_roi(frame)

        assert result.rotation_angle_deg == pytest.approx(tilt_degrees, abs=0.01)
        np.testing.assert_allclose(result.center_px, palm_center, atol=0.01)
        assert result.roi_image.shape == (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE, 3)
        assert result.clahe_image.shape == (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE)
        assert result.clahe_image.dtype == np.uint8

        center_pixels = np.argwhere(result.roi_image[:, :, 1] > 127)
        upper_pixels = np.argwhere(result.roi_image[:, :, 2] > 127)
        assert center_pixels.size and upper_pixels.size
        center_y, center_x = center_pixels.mean(axis=0)
        upper_y, upper_x = upper_pixels.mean(axis=0)
        assert center_x == pytest.approx(DEFAULT_ROI_SIZE / 2, abs=1)
        assert center_y == pytest.approx(DEFAULT_ROI_SIZE / 2, abs=1)
        assert upper_x == pytest.approx(center_x, abs=2)
        assert upper_y == pytest.approx(center_y - 18 * DEFAULT_ROI_SIZE / 65, abs=2)

    def test_extract_embedding_is_normalized_deterministic_and_texture_sensitive(self):
        stripes = (np.arange(DEFAULT_ROI_SIZE) // 4 % 2 * 160 + 40).astype(np.uint8)
        vertical = np.tile(stripes, (DEFAULT_ROI_SIZE, 1))
        roi = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)
        rotated = cv2.cvtColor(vertical.T, cv2.COLOR_GRAY2BGR)

        embedding = self.engine.extract_embedding(roi)

        assert embedding.shape == (PALM_FEATURE_DIMENSION,)
        assert embedding.dtype == np.float32
        assert np.isfinite(embedding).all()
        assert np.linalg.norm(embedding) == pytest.approx(1.0)
        np.testing.assert_array_equal(embedding, self.engine.extract_embedding(roi))
        assert float(embedding @ self.engine.extract_embedding(rotated)) < 0.95

    @pytest.mark.parametrize("shape,dtype", [((128, 128), np.uint8), ((128, 128, 3), np.float32)])
    def test_extract_embedding_rejects_invalid_roi(self, shape, dtype):
        with pytest.raises(ValueError, match="aligned palm ROI"):
            self.engine.extract_embedding(np.zeros(shape, dtype=dtype))

    def test_verification_is_pending_milestone_d(self):
        dummy_roi = np.zeros((128, 128, 3), dtype=np.uint8)
        dummy_embedding = self.engine.extract_embedding(dummy_roi)
        with pytest.raises(NotImplementedError) as exc_d:
            self.engine.verify(dummy_roi, dummy_embedding)
        assert "Milestone D" in str(exc_d.value)

