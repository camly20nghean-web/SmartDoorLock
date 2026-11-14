"""Interactive face biometric template enrollment via Webcam.

Captures exactly 5 high-quality face samples from webcam, verifies quality,
computes normalized mean embedding, and records into SQLite table biometric_templates.

Usage:
    python scripts/enroll_face.py --user-id NV001 [--samples 5] [--camera-id 0]
"""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.biometrics.face import (
    DEFAULT_FEATURE_DIMENSION,
    DEFAULT_MODEL_VERSION,
    FaceRecognitionEngine,
)
from src.capture.webcam import WebcamCamera
from src.database.connection import DatabaseConnection
from src.database.models import BiometricType, FeatureFormat
from src.database.repository import BiometricTemplateRepository, UserRepository


def get_db_path() -> str:
    raw = get_settings().database_path
    if raw.startswith("sqlite:///"):
        return raw.replace("sqlite:///", "")
    return raw


def enroll_user_face(user_id: str, required_samples: int = 5, camera_id: int = 0) -> int:
    """Run interactive webcam enrollment session for user_id."""
    db_path = get_db_path()
    db = DatabaseConnection(db_path=db_path)
    db.init_db()

    user_repo = UserRepository(db)
    template_repo = BiometricTemplateRepository(db)

    # 1. Verify user exists and is active
    user = user_repo.get_user(user_id)
    if not user:
        print(f"\n❌ ERROR: User ID '{user_id}' does not exist in database.")
        print("   Please create the user first using: python scripts/manage_users.py add-user --user-id ... --name ...")
        return 1

    if not user.is_active:
        print(f"\n❌ ERROR: User '{user_id}' account is {user.status.value}. Only ACTIVE users can enroll biometrics.")
        return 1

    # 2. Initialize Face Engine
    engine = FaceRecognitionEngine()
    if not engine.is_ready:
        print("\n❌ ERROR: Face recognition models not found. Please run: python scripts/download_models.py")
        return 1

    # 3. Connect to Webcam
    camera = WebcamCamera(camera_index=camera_id, width=640, height=480, fps=30)
    if not camera.connect():
        print(f"\n❌ ERROR: Cannot access camera {camera_id}.")
        return 1

    window_name = f"SmartDoorLock - Face Enrollment: {user.full_name} ({user_id})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("=" * 65)
    print(f"  STARTING FACE ENROLLMENT: {user.full_name} ({user_id})")
    print("=" * 65)
    print(f"Target samples required : {required_samples}")
    print("Instructions            : Look directly at the camera.")
    print("                          Tilt your head slightly between samples.")
    print("Exit anytime            : Press 'q' or 'ESC'")
    print("=" * 65)

    collected_embeddings: list[np.ndarray] = []
    last_capture_time = 0.0
    capture_cooldown_seconds = 0.8  # Enforces temporal diversity between samples

    try:
        while len(collected_embeddings) < required_samples:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                continue

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # Detect faces
            faces = engine.detect_all_faces(frame)
            num_faces = len(faces)

            status_msg = ""
            status_color = (0, 0, 255)  # Red default

            if num_faces == 0:
                status_msg = "No face detected. Please position face in front of camera."
            elif num_faces > 1:
                status_msg = f"Multiple faces ({num_faces})! Only 1 face allowed."
            else:
                face_data = faces[0]
                box = [int(v) for v in face_data[:4]]
                x, y, fw, fh = box

                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (0, 255, 255), 2)

                # Assess quality
                quality = engine.assess_face_quality(frame, face_data)
                now = time.time()

                if not quality.is_acceptable:
                    status_msg = quality.rejection_reason
                    status_color = (0, 165, 255)  # Orange
                elif (now - last_capture_time) < capture_cooldown_seconds:
                    remaining = capture_cooldown_seconds - (now - last_capture_time)
                    status_msg = f"Hold steady... next sample in {remaining:.1f}s"
                    status_color = (255, 255, 0)  # Cyan
                else:
                    # Capture sample
                    aligned = engine.align_and_crop(frame, face_data)
                    embedding = engine.extract_embedding(aligned)
                    collected_embeddings.append(embedding)
                    last_capture_time = now

                    status_msg = f"Sample {len(collected_embeddings)}/{required_samples} captured successfully!"
                    status_color = (0, 255, 0)  # Green
                    cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (0, 255, 0), 3)

            # Render HUD
            progress_text = f"Enrolling: {len(collected_embeddings)}/{required_samples} samples"
            cv2.putText(display_frame, progress_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, status_msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Enrollment cancelled by user.")
                return 1

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(collected_embeddings) < required_samples:
        print("\n❌ Enrollment incomplete.")
        return 1

    # 4. Compute mean embedding and L2 normalize
    print("\nProcessing captured facial vectors...")
    stacked = np.stack(collected_embeddings, axis=0)  # Shape (5, 128)
    mean_vec = np.mean(stacked, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_vec)
    if norm > 1e-6:
        normalized_mean_vec = mean_vec / norm
    else:
        normalized_mean_vec = mean_vec

    feature_bytes = normalized_mean_vec.tobytes()

    # 5. Persist to database
    template = template_repo.create_template(
        user_id=user_id,
        biometric_type=BiometricType.FACE,
        feature_data=feature_bytes,
        feature_format=FeatureFormat.FLOAT32_EMBEDDING,
        feature_dimension=DEFAULT_FEATURE_DIMENSION,
        model_version=DEFAULT_MODEL_VERSION
    )

    print("=" * 65)
    print("✅ FACE BIOMETRIC TEMPLATE ENROLLED SUCCESSFULLY!")
    print(f"   Template ID       : {template.template_id}")
    print(f"   User ID           : {template.user_id} ({user.full_name})")
    print(f"   Feature Dimension : {template.feature_dimension} (float32)")
    print(f"   Model Version     : {template.model_version}")
    print(f"   Created At (UTC)  : {template.created_at}")
    print("=" * 65)
    return 0


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Enroll face biometric template into SQLite.")
    parser.add_argument("--user-id", required=True, help="User ID to enroll")
    parser.add_argument("--samples", type=int, default=5, help="Number of distinct facial samples to average (default: 5)")
    parser.add_argument("--camera-id", type=int, default=settings.camera_index, help="Webcam device ID")
    args = parser.parse_args()

    sys.exit(enroll_user_face(args.user_id, args.samples, args.camera_id))


if __name__ == "__main__":
    main()

