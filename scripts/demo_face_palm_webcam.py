"""Enroll and verify live face and palm samples with an isolated mock lock."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_face_palm_unlock import DEMO_PALM_THRESHOLD, FacePalmDemo, unit_vector
from src.biometrics.face import DEFAULT_FEATURE_DIMENSION, DEFAULT_MODEL_VERSION, FaceRecognitionEngine
from src.biometrics.palm import (
    MIN_SHARPNESS_VAR,
    PALM_FEATURE_DIMENSION,
    PALM_FEATURE_VERSION,
    PalmprintRecognitionEngine,
)
from src.capture.webcam import WebcamCamera
from src.database.connection import DatabaseConnection
from src.database.models import ActuationStatus, AuthPolicy, get_utc_now_iso

DB_PATH = PROJECT_ROOT / "data" / "face_palm_webcam_demo" / "demo.sqlite3"
FACE_THRESHOLD = 0.60
DEFAULT_FACE_MIN_SHARPNESS = 40.0
SESSION_SECONDS = 30
ENROLL_SAMPLES = 5
WINDOW = "SmartDoorLock FACE_AND_PALM | MockDoorLock"


class CaptureStopped(RuntimeError):
    def __init__(self, message, hand_side=None):
        super().__init__(message)
        self.hand_side = hand_side


def face_candidate(engine, frame, min_sharpness):
    face, reason = engine.detect_single_face(frame)
    if face is None:
        return None, None, reason
    quality = engine.assess_face_quality(frame, face, min_sharpness=min_sharpness)
    if not quality.is_acceptable:
        return None, None, quality.rejection_reason
    return face, None, "Face ready"


def palm_candidate(engine, frame, required_hand=None, *, min_sharpness=MIN_SHARPNESS_VAR):
    result = engine.detect_hand_and_roi(frame, min_sharpness=min_sharpness)
    quality = result.quality
    hand = {"Left": "LEFT", "Right": "RIGHT"}.get(result.handedness)
    if quality is None:
        return None, hand, f"Show one palm facing the camera | Min sharpness {min_sharpness:g}"
    sharpness = f"Sharpness {quality.sharpness:.1f} / min {min_sharpness}"
    reasons = list(quality.rejection_reasons)
    if required_hand and hand != required_hand:
        reasons.append(f"Show your {required_hand} palm")
    if reasons or not quality.is_acceptable or result.roi_image is None or not result.is_palmar or hand is None:
        return None, hand, f"{sharpness} | " + "; ".join(reasons or ["Palm quality rejected"])
    return result.roi_image, hand, f"{hand} palm ready | {sharpness}"


def capture(camera, title, inspect, extract, count, deadline):
    vectors = []
    last_status = "No frame inspected"
    last_hand = None
    while time.monotonic() < deadline:
        ok, frame = camera.get_frame()
        if not ok or frame is None:
            detail = getattr(camera, "error", "")
            if not isinstance(detail, str) or not detail:
                detail = "camera stopped delivering frames"
            raise CaptureStopped(f"{title}: {detail}", last_hand)
        candidate, hand, last_status = inspect(frame)
        if hand is not None:
            last_hand = hand
        preview = frame.copy()
        instructions = "SPACE: capture | Q/Esc: cancel" if candidate is not None else "Q/Esc: cancel"
        lines = [
            f"{title} | {len(vectors)}/{count} | {max(0, deadline - time.monotonic()):.0f}s",
            instructions,
            *last_status.split(" | "),
        ]
        for index, line in enumerate(lines):
            cv2.putText(preview, line, (10, 24 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise CaptureStopped(f"{title}: cancelled; last result: {last_status}", last_hand)
        if key == ord(" ") and candidate is not None and time.monotonic() < deadline:
            vectors.append(extract(frame, candidate))
            print(f"{title}: {len(vectors)}/{count} | {last_status}", flush=True)
            if len(vectors) == count:
                return vectors, hand
    raise CaptureStopped(f"{title}: timed out; last result: {last_status}", last_hand)


def save_enrollment(db, user_id, hand, faces, palms):
    """Commit the account and both templates together after quality-gated capture."""
    face = unit_vector(np.mean(faces, axis=0), DEFAULT_FEATURE_DIMENSION)
    palm = unit_vector(np.mean(palms, axis=0), PALM_FEATURE_DIMENSION)
    stamp = get_utc_now_iso()
    with closing(db.get_connection()) as conn, conn:
        conn.execute(
            "INSERT INTO users (user_id, full_name, created_at, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_id) DO NOTHING", (user_id, user_id, stamp, stamp),
        )
        if conn.execute("SELECT status FROM users WHERE user_id = ?", (user_id,)).fetchone()[0] != "ACTIVE":
            raise ValueError("Only an active demo user can enroll.")
        store_face(conn, user_id, face, stamp)
        store_palm(conn, user_id, hand, palm)


def store_face(conn, user_id, face, stamp):
    conn.execute(
        "DELETE FROM biometric_templates WHERE user_id = ? AND biometric_type = 'FACE' AND model_version = ?",
        (user_id, DEFAULT_MODEL_VERSION),
    )
    conn.execute(
        "INSERT INTO biometric_templates "
        "(user_id, biometric_type, feature_data, feature_format, feature_dimension, model_version, created_at) "
        "VALUES (?, 'FACE', ?, 'FLOAT32_EMBEDDING', ?, ?, ?)",
        (user_id, face.tobytes(), DEFAULT_FEATURE_DIMENSION, DEFAULT_MODEL_VERSION, stamp),
    )


def store_palm(conn, user_id, hand, palm):
    conn.execute(
        "INSERT INTO demo_palm_templates (user_id, hand_side, feature_version, vector) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (user_id, hand_side, feature_version) DO UPDATE SET vector = excluded.vector",
        (user_id, hand, PALM_FEATURE_VERSION, palm.tobytes()),
    )


def load_face_templates(db):
    with closing(db.get_connection()) as conn:
        rows = conn.execute(
            "SELECT user_id, feature_data FROM biometric_templates WHERE biometric_type = 'FACE' "
            "AND feature_format = 'FLOAT32_EMBEDDING' AND feature_dimension = ? AND model_version = ?",
            (DEFAULT_FEATURE_DIMENSION, DEFAULT_MODEL_VERSION),
        ).fetchall()
    return {
        row["user_id"]: unit_vector(np.frombuffer(row["feature_data"], dtype="<f4"), DEFAULT_FEATURE_DIMENSION)
        for row in rows
    }


def capture_face(camera, engine, sharpness, count, deadline):
    return capture(
        camera, f"Face (min sharpness {sharpness:g})",
        lambda frame: face_candidate(engine, frame, sharpness),
        lambda frame, face: engine.extract_embedding(engine.align_and_crop(frame, face)),
        count, deadline,
    )[0]


def capture_palm(camera, engine, sharpness, count, deadline, required_hand=None):
    return capture(
        camera, "Palm",
        lambda frame: palm_candidate(engine, frame, required_hand, min_sharpness=sharpness),
        lambda frame, roi: engine.extract_embedding(roi), count, deadline,
    )


def verify_once(demo, policy, camera, face_engine, palm_engine, face_sharpness, palm_sharpness):
    """Capture the selected factors within one demo authentication session."""
    deadline = time.monotonic() + SESSION_SECONDS
    session = demo.issue_session(SESSION_SECONDS)
    face = None
    palm = None
    hand = None
    capture_error = None
    try:
        if policy != AuthPolicy.PALM_ONLY:
            face = capture_face(camera, face_engine, face_sharpness, 1, deadline)[0]
        if policy != AuthPolicy.FACE_ONLY:
            palms, hand = capture_palm(camera, palm_engine, palm_sharpness, 1, deadline)
            palm = palms[0]
    except CaptureStopped as error:
        hand = error.hand_side
        capture_error = str(error)
    result = demo.authenticate(session, face, palm, hand, policy=policy)
    return result, hand, capture_error


def run_webcam(args, demo, face_engine, palm_engine, camera):
    palm_only = args.command == "verify-palm"

    if args.command == "enroll":
        faces = capture_face(camera, face_engine, args.face_min_sharpness, ENROLL_SAMPLES, time.monotonic() + 90)
        palms, hand = capture_palm(camera, palm_engine, args.palm_min_sharpness, ENROLL_SAMPLES, time.monotonic() + 90, args.hand)
        save_enrollment(demo.db, args.user, hand, faces, palms)
        print(f"Enrolled {args.user}: {ENROLL_SAMPLES} face and {ENROLL_SAMPLES} {hand} palm captures.")
        return 0

    policy = AuthPolicy.PALM_ONLY if palm_only else AuthPolicy.FACE_AND_PALM
    result, _, capture_error = verify_once(
        demo, policy, camera, face_engine, palm_engine,
        DEFAULT_FACE_MIN_SHARPNESS if palm_only else args.face_min_sharpness, args.palm_min_sharpness,
    )
    if capture_error:
        print(f"{result.decision.value} / {result.actuation_status.value} | {capture_error} | {result.reason}")
        return 1

    print(f"{result.decision.value} / {result.actuation_status.value} | {result.reason}")
    if result.actuation_status == ActuationStatus.ACTUATED:
        print("MockDoorLock unlocked. Waiting for automatic relock.", flush=True)
        time.sleep(demo.lock.default_auto_relock_seconds)
        return 0
    return 1


def positive_sharpness(value):
    try:
        threshold = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a finite, positive number") from None
    if not np.isfinite(threshold) or threshold <= 0:
        raise argparse.ArgumentTypeError("expected a finite, positive number")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--db-path", type=Path, help="Existing SQLite demo database to use instead of the default")
    subcommands = parser.add_subparsers(dest="command", required=True)
    enroll = subcommands.add_parser("enroll")
    enroll.add_argument("--user", required=True, type=str.strip)
    enroll.add_argument("--hand", choices=("LEFT", "RIGHT"), required=True)
    verify = subcommands.add_parser("verify")
    verify_palm = subcommands.add_parser("verify-palm", help="Verify an enrolled palm with MockDoorLock")
    for command in (enroll, verify):
        command.add_argument(
            "--face-min-sharpness", type=positive_sharpness, default=DEFAULT_FACE_MIN_SHARPNESS,
            help="Minimum face sharpness for this demo (default: %(default)s)",
        )
    for command in (enroll, verify, verify_palm):
        command.add_argument(
            "--palm-min-sharpness", type=positive_sharpness, default=MIN_SHARPNESS_VAR,
            help="Minimum palm sharpness for this demo (default: %(default)s)",
        )
    args = parser.parse_args()
    if args.command == "enroll" and not args.user:
        parser.error("--user must not be empty")

    db_path = args.db_path.resolve() if args.db_path else DB_PATH
    if args.db_path and not db_path.is_file():
        parser.error("--db-path must point to an existing demo database")
    print(f"Demo database: {db_path}", flush=True)
    palm_only = args.command == "verify-palm"
    thresholds = f"palm {DEMO_PALM_THRESHOLD:.2f}"
    if not palm_only:
        thresholds = f"face {FACE_THRESHOLD:.2f}, {thresholds}"
    print(f"Cosine thresholds: {thresholds} | MockDoorLock only", flush=True)
    demo = None
    camera = WebcamCamera(camera_index=args.camera_id)
    try:
        demo = FacePalmDemo(DatabaseConnection(str(db_path)), face_threshold=FACE_THRESHOLD)
        if args.command == "verify":
            demo.face_templates = load_face_templates(demo.db)
            if not demo.face_templates:
                raise ValueError("No compatible face templates. Run enroll first.")
        if palm_only and not any(demo.palm_templates(hand) for hand in ("LEFT", "RIGHT")):
            raise ValueError("No compatible palm templates. Run enroll first.")
        if not camera.connect():
            raise CaptureStopped(f"Cannot open camera {args.camera_id}")
        face_engine = None if palm_only else FaceRecognitionEngine()
        if face_engine is not None and not face_engine.is_ready:
            raise RuntimeError("YuNet/SFace models are unavailable")
        with PalmprintRecognitionEngine() as palm_engine:
            return run_webcam(args, demo, face_engine, palm_engine, camera)
    except (OSError, RuntimeError, ValueError, sqlite3.Error, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if demo is not None:
            demo.lock.lock()


if __name__ == "__main__":
    sys.exit(main())
