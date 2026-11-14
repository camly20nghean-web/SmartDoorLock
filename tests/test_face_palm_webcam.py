import argparse
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import demo_face_palm_webcam as webcam
from scripts.demo_face_palm_unlock import FacePalmDemo
from src.biometrics.face import FaceRecognitionEngine
from src.biometrics.palm import (
    PALM_FEATURE_DIMENSION,
    PalmQualityAssessment,
    PalmROIResult,
    PalmprintRecognitionEngine,
)
from src.database.connection import DatabaseConnection


@pytest.mark.parametrize(
    "palm_min_sharpness,probe_sharpness,expected",
    [(50.0, 50.0, "ACTUATED"), (50.0, 49.9, "NOT_ACTUATED"), (2.0, 2.3, "ACTUATED")],
)
def test_enroll_reopen_and_verify_quality_gate(
    tmp_path, monkeypatch, capsys, palm_min_sharpness, probe_sharpness, expected
):
    db_path = str(tmp_path / "webcam.sqlite3")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = Mock()
    camera.get_frame.return_value = (True, frame)
    face = Mock()
    face.detect_single_face.return_value = (np.ones(15), "OK")
    face.assess_face_quality.return_value = SimpleNamespace(is_acceptable=True)
    face.extract_embedding.return_value = np.eye(1, 128, dtype=np.float32)[0]
    palm = Mock()
    palm.extract_embedding.return_value = np.eye(1, PALM_FEATURE_DIMENSION, dtype=np.float32)[0]

    def palm_result(sharpness):
        acceptable = sharpness >= palm_min_sharpness
        return PalmROIResult(
            has_hand=True, handedness="Left", is_palmar=True, roi_image=frame[:128, :128],
            quality=PalmQualityAssessment(
                acceptable, sharpness, 100, 80, True, [] if acceptable else ["Image blurred"],
            ),
        )

    palm.detect_hand_and_roi.return_value = palm_result(max(probe_sharpness, palm_min_sharpness))
    monkeypatch.setattr(webcam.cv2, "imshow", Mock())
    key = Mock(return_value=ord(" "))
    monkeypatch.setattr(webcam.cv2, "waitKey", key)
    monkeypatch.setattr(webcam.time, "sleep", Mock())

    enroll = SimpleNamespace(
        command="enroll", user="P01", hand="LEFT",
        face_min_sharpness=20.0, palm_min_sharpness=palm_min_sharpness,
    )
    demo = FacePalmDemo(DatabaseConnection(db_path), face_threshold=webcam.FACE_THRESHOLD)
    assert webcam.run_webcam(enroll, demo, face, palm, camera) == 0

    reopened = FacePalmDemo(DatabaseConnection(db_path), face_threshold=webcam.FACE_THRESHOLD)
    reopened.face_templates = webcam.load_face_templates(reopened.db)
    palm.detect_hand_and_roi.return_value = palm_result(probe_sharpness)
    if expected == "NOT_ACTUATED":
        key.side_effect = [ord(" "), ord(" "), ord("q")]

    try:
        verify = SimpleNamespace(
            command="verify", face_min_sharpness=20.0, palm_min_sharpness=palm_min_sharpness,
        )
        assert webcam.run_webcam(verify, reopened, face, palm, camera) == (0 if expected == "ACTUATED" else 1)
        with closing(reopened.db.get_connection()) as conn:
            assert conn.execute("SELECT actuation_status FROM access_logs").fetchone()[0] == expected
        assert reopened.lock.is_locked() == (expected == "NOT_ACTUATED")
        assert all(call.kwargs["min_sharpness"] == 20.0 for call in face.assess_face_quality.call_args_list)
        assert all(call.kwargs["min_sharpness"] == palm_min_sharpness
                   for call in palm.detect_hand_and_roi.call_args_list)
        output = capsys.readouterr().out
        assert "min sharpness 20" in output
        if expected == "NOT_ACTUATED":
            assert palm.extract_embedding.call_count == webcam.ENROLL_SAMPLES
            assert "Sharpness 49.9 / min 50.0" in output
    finally:
        reopened.lock.lock()


@pytest.mark.parametrize("hand,expected", [("Left", "ACTUATED"), ("Right", "NOT_ACTUATED")])
def test_verify_palm_uses_enrolled_hand_without_face(tmp_path, monkeypatch, hand, expected):
    demo = FacePalmDemo(DatabaseConnection(str(tmp_path / "webcam.sqlite3")))
    demo.users.create_user("P01", "P01")
    vector = np.eye(1, PALM_FEATURE_DIMENSION, dtype=np.float32)[0]
    demo.enroll_palm("P01", "LEFT", vector)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = Mock()
    camera.get_frame.return_value = (True, frame)
    palm = Mock()
    palm.detect_hand_and_roi.return_value = PalmROIResult(
        has_hand=True, handedness=hand, is_palmar=True, roi_image=frame[:128, :128],
        quality=PalmQualityAssessment(True, 2.5, 100, 80, True),
    )
    palm.extract_embedding.return_value = vector
    monkeypatch.setattr(webcam.cv2, "imshow", Mock())
    monkeypatch.setattr(webcam.cv2, "waitKey", Mock(return_value=ord(" ")))
    monkeypatch.setattr(webcam.time, "sleep", Mock())

    try:
        args = SimpleNamespace(command="verify-palm", palm_min_sharpness=2.0)
        assert webcam.run_webcam(args, demo, None, palm, camera) == (0 if expected == "ACTUATED" else 1)
        assert demo.lock.is_locked() == (expected == "NOT_ACTUATED")
        with closing(demo.db.get_connection()) as conn:
            audit = conn.execute("SELECT auth_method, actuation_status FROM access_logs").fetchone()
        assert tuple(audit) == ("PALM", expected)
    finally:
        demo.lock.lock()


def test_face_at_23_3_requires_lower_demo_threshold(monkeypatch):
    engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
    face = np.array([0, 0, 100, 100])
    monkeypatch.setattr(engine, "detect_single_face", lambda frame: (face, "OK"))
    rows, columns = np.indices((100, 100))
    stronger_pattern = (rows < 15) | ((rows == 15) & (columns < 30))
    gray = (100 + ((rows + columns) % 2) * (1 + stronger_pattern)).astype(np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)

    candidate, _, reason = webcam.face_candidate(engine, frame, webcam.DEFAULT_FACE_MIN_SHARPNESS)
    assert candidate is None
    assert "sharpness: 23.3 < 40.0" in reason
    candidate, _, _ = webcam.face_candidate(engine, frame, 20.0)
    assert candidate is face


@pytest.mark.parametrize(
    "handedness,brightness,required_hand,rejection",
    [
        ("Right", 100, "RIGHT", None),
        ("Left", 100, "LEFT", "Back of hand"),
        ("Right", 10, "RIGHT", "Insufficient lighting"),
        ("Right", 100, "LEFT", "Show your LEFT palm"),
    ],
)
def test_palm_at_2_3_uses_requested_threshold_without_bypassing_other_checks(
    monkeypatch, handedness, brightness, required_hand, rejection
):
    engine = PalmprintRecognitionEngine.__new__(PalmprintRecognitionEngine)
    engine._clahe = webcam.cv2.createCLAHE()
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    points = np.tile([320.0, 300.0], (21, 1))
    points[5], points[9], points[17] = [375, 200], [320, 200], [265, 200]
    landmarks = [SimpleNamespace(x=x / 640, y=y / 480, z=0.0) for x, y in points]
    engine._detector = Mock()
    engine._detector.detect.return_value = SimpleNamespace(
        hand_landmarks=[landmarks],
        handedness=[[SimpleNamespace(category_name=handedness, score=1.0)]],
    )
    # Fix the measured Laplacian variance while exercising the engine's real ROI and quality checks.
    amplitude = np.sqrt(2.3)
    monkeypatch.setattr(webcam.cv2, "Laplacian", lambda *_: np.array([-amplitude, amplitude]))

    candidate, _, reason = webcam.palm_candidate(engine, frame, required_hand)
    assert candidate is None
    assert "Sharpness 2.3 / min 50.0" in reason and "Image blurred" in reason

    candidate, hand, reason = webcam.palm_candidate(engine, frame, required_hand, min_sharpness=2.0)
    assert "Sharpness 2.3 / min 2.0" in reason and "Image blurred" not in reason
    if rejection:
        assert candidate is None and rejection in reason
    else:
        assert candidate is not None and hand == required_hand


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_sharpness_argument_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite, positive"):
        webcam.positive_sharpness(value)
