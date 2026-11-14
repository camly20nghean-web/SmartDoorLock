"""Live Webcam Face Recognition & Access Control Demonstration.

Features:
- Real-time face detection & 5-point landmark rendering via OpenCV YuNet.
- 128-D SFace embedding extraction & Cosine Similarity search against SQLite enrolled templates.
- Enforces single face requirement (multiple faces strictly denied).
- Secure pipeline generating authenticated BiometricAuthClaim(is_trusted_source=True).
- Integrated with AccessControlService in demo mode (controlling MockDoorLock only).
- Live HUD displaying empirical inference latency (ms), real FPS, similarity score, and lock status.
- Never displays user identity when similarity threshold is not satisfied.

Usage:
    python scripts/demo_face_unlock.py [--threshold 0.60] [--camera-id 0]
"""
import argparse
from pathlib import Path
import sys
import time
import uuid
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.biometrics.face import FaceRecognitionEngine
from src.capture.webcam import WebcamCamera
from src.database.connection import DatabaseConnection
from src.database.models import (
    AccessDecision,
    AuthPolicy,
    BiometricAuthClaim,
    BiometricType,
)
from src.database.repository import AccessLogRepository, BiometricTemplateRepository, UserRepository
from src.hardware.mock_lock import MockDoorLock
from src.services.access_control import AccessControlService


def get_db_path() -> str:
    raw = get_settings().database_path
    if raw.startswith("sqlite:///"):
        return raw.replace("sqlite:///", "")
    return raw


def run_demo(threshold: float = 0.60, camera_id: int = 0, db_path: str = None) -> int:
    """Run interactive real-time face verification loop."""
    resolved_db_path = db_path if db_path else get_db_path()
    db = DatabaseConnection(db_path=resolved_db_path)
    db.init_db()

    user_repo = UserRepository(db)
    template_repo = BiometricTemplateRepository(db)
    log_repo = AccessLogRepository(db)

    # 1. Initialize Mock Lock & Access Control Service (Test/Demo mode explicitly enabled)
    lock = MockDoorLock(default_auto_relock_seconds=4.0)
    service = AccessControlService(
        user_repo=user_repo,
        log_repo=log_repo,
        lock=lock,
        auto_relock_seconds=4.0,
        is_test_or_demo_mode=True  # Strictly controls MockDoorLock
    )

    # 2. Initialize Face Recognition Engine
    engine = FaceRecognitionEngine(similarity_threshold=threshold)
    if not engine.is_ready:
        print("\n❌ ERROR: Face recognition models not found. Please run: python scripts/download_models.py")
        return 1

    # 3. Load all enrolled face templates from database into memory for fast cosine search
    enrolled_templates = template_repo.get_templates_by_user("", biometric_type=None)
    # Fetch all face templates directly
    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT t.*, u.full_name, u.status FROM biometric_templates t "
            "JOIN users u ON t.user_id = u.user_id "
            "WHERE t.biometric_type = 'FACE';"
        )
        template_rows = cursor.fetchall()

    cached_templates: list[tuple[str, str, str, np.ndarray]] = []
    for row in template_rows:
        vec = np.frombuffer(row["feature_data"], dtype=np.float32)
        cached_templates.append((row["user_id"], row["full_name"], row["status"], vec))

    print("=" * 70)
    print("  SMART DOOR LOCK - LIVE FACE RECOGNITION DEMO (Phase 3)")
    print("=" * 70)
    print(f"Enrolled face templates in DB : {len(cached_templates)}")
    for uid, name, stat, _ in cached_templates:
        print(f"  • [{uid}] {name} (Status: {stat})")
    print(f"Cosine Similarity Threshold   : {threshold:.2f}")
    print("Hardware Actuator Target      : MockDoorLock (Software Simulation)")
    print("Controls                      : Press 'q' or 'ESC' to exit safely")
    print("=" * 70)

    # 4. Connect to Webcam
    camera = WebcamCamera(camera_index=camera_id, width=640, height=480, fps=30)
    if not camera.connect():
        print(f"\n❌ ERROR: Cannot access camera device {camera_id}.")
        return 1

    window_name = "SmartDoorLock - Live Face Unlock Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Metrics tracking
    frame_count = 0
    start_time = time.time()
    current_fps = 0.0
    last_auth_attempt_time = 0.0
    auth_cooldown_seconds = 2.0  # Prevent spamming unlock calls

    last_verification_status = "IDLE - STAND IN FRONT OF CAMERA"
    last_status_color = (255, 255, 255)

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = camera.get_frame()
            if not ret or frame is None:
                continue

            frame_count += 1
            now = time.time()
            if (now - start_time) >= 1.0:
                current_fps = frame_count / (now - start_time)
                frame_count = 0
                start_time = now

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # 5. Measure actual inference latency
            t_infer_start = time.perf_counter()
            faces = engine.detect_all_faces(frame)
            num_faces = len(faces)

            detected_box = None
            landmarks = None

            if num_faces == 1:
                face_data = faces[0]
                box = [int(v) for v in face_data[:4]]
                x, y, fw, fh = box
                detected_box = box

                # Extract 5 landmarks
                landmarks = [
                    (int(face_data[4]), int(face_data[5])),    # right eye
                    (int(face_data[6]), int(face_data[7])),    # left eye
                    (int(face_data[8]), int(face_data[9])),    # nose tip
                    (int(face_data[10]), int(face_data[11])),  # right mouth
                    (int(face_data[12]), int(face_data[13])),  # left mouth
                ]

                # Check face quality
                quality = engine.assess_face_quality(frame, face_data)

                if quality.is_acceptable:
                    # Align and extract embedding
                    aligned = engine.align_and_crop(frame, face_data)
                    probe_vec = engine.extract_embedding(aligned)

                    # Match against enrolled templates
                    best_match_user_id = None
                    best_match_name = None
                    best_score = -1.0

                    for uid, name, stat, enrolled_vec in cached_templates:
                        sim = engine.compute_similarity(probe_vec, enrolled_vec)
                        if sim > best_score:
                            best_score = sim
                            best_match_user_id = uid
                            best_match_name = name

                    # Evaluate authentication request through secure AccessControlService
                    if (now - last_auth_attempt_time) >= auth_cooldown_seconds:
                        session_id = f"sess_{uuid.uuid4().hex[:8]}"

                        if best_score >= threshold and best_match_user_id is not None:
                            # Construct trusted internal claim
                            trusted_claim = BiometricAuthClaim(
                                user_id=best_match_user_id,
                                biometric_type=BiometricType.FACE,
                                is_matched=True,
                                similarity_score=best_score,
                                session_id=session_id,
                                timestamp=now,
                                is_trusted_source=True  # Sourced directly from verified pipeline
                            )

                            outcome = service.process_access_request(
                                policy=AuthPolicy.FACE_ONLY,
                                face_claim=trusted_claim
                            )

                            if outcome.decision == AccessDecision.GRANTED:
                                last_verification_status = f"ACCESS GRANTED: {best_match_name} (Score: {best_score:.2f})"
                                last_status_color = (0, 255, 0)  # Green
                            else:
                                last_verification_status = f"ACCESS DENIED: {outcome.reason}"
                                last_status_color = (0, 0, 255)  # Red
                        else:
                            # Unknown face or below threshold: Never show name
                            last_verification_status = f"ACCESS DENIED: Unknown Face (Score: {max(0.0, best_score):.2f} < {threshold:.2f})"
                            last_status_color = (0, 0, 255)  # Red

                        last_auth_attempt_time = now
                else:
                    last_verification_status = f"QUALITY: {quality.rejection_reason}"
                    last_status_color = (0, 165, 255)  # Orange

            elif num_faces > 1:
                last_verification_status = f"SECURITY ALERT: {num_faces} faces detected! Only 1 face allowed."
                last_status_color = (0, 0, 255)

            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_infer_start) * 1000.0

            # 6. Render Visual Annotations
            if detected_box is not None:
                x, y, fw, fh = detected_box
                box_color = (0, 255, 0) if "GRANTED" in last_verification_status else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), box_color, 2)

                # Draw facial landmarks
                if landmarks:
                    for lx, ly in landmarks:
                        cv2.circle(display_frame, (lx, ly), 3, (0, 255, 255), -1)

            # 7. Render Comprehensive HUD
            hud_bg = display_frame[:75, :]
            cv2.rectangle(display_frame, (0, 0), (w, 75), (20, 20, 20), -1)

            lock_state = lock.get_state().value
            lock_color = (0, 255, 0) if lock_state == "UNLOCKED" else (0, 0, 255)

            hud_line1 = f"FPS: {current_fps:.1f} | Infer: {infer_ms:.1f}ms | Lock: [{lock_state}]"
            cv2.putText(display_frame, hud_line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"[{lock_state}]", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lock_color, 2)

            cv2.putText(display_frame, last_verification_status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, last_status_color, 2)

            # Bottom info bar
            cv2.rectangle(display_frame, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(display_frame, "YuNet + SFace (OpenCV Zoo) | Press 'q' or 'ESC' to exit", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Demo exited by user.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released and preview closed cleanly.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Live Face Unlock Demo using YuNet + SFace.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Cosine similarity threshold (default: 0.60)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--db-path", default=None, help="Custom SQLite database path for isolated testing")
    args = parser.parse_args()

    sys.exit(run_demo(threshold=args.threshold, camera_id=args.camera_id, db_path=args.db_path))


if __name__ == "__main__":
    main()

