"""Manual webcam verification and live preview script.

Usage:
    python scripts/preview_webcam.py [--camera-id 0] [--width 640] [--height 480]
"""
import argparse
import sys
import time
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.capture.webcam import WebcamCamera


def run_preview(camera_id: int, width: int, height: int, fps: int) -> int:
    """Run interactive webcam display window with real-time performance metrics."""
    print("=" * 60)
    print("  SMART DOOR LOCK - WEBCAM PREVIEW UTILITY (Phase 1)")
    print("=" * 60)
    print(f"Target Camera Device ID : {camera_id}")
    print(f"Requested Resolution    : {width}x{height} @ {fps} FPS")
    print("Controls                : Press 'q' or 'ESC' to exit safely")
    print("=" * 60)

    camera = WebcamCamera(camera_index=camera_id, width=width, height=height, fps=fps)

    if not camera.connect():
        print(f"\n[ERROR] Unable to connect to camera {camera_id}.")
        print("Suggestions:")
        print(" 1. Check if the webcam is in use by another app (Zoom, Teams, Camera app).")
        print(" 2. Try passing another camera ID (e.g. python scripts/preview_webcam.py --camera-id 1).")
        print(" 3. Verify privacy settings allow camera access in Windows Settings.")
        return 1

    window_name = "SmartDoorLock - Camera Preview (Press 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                print("\n[WARNING] Failed to grab frame from camera. Exiting preview loop...")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            h, w = frame.shape[:2]

            # Render overlay information
            hud_text_fps = f"FPS: {current_fps:.1f} | Res: {w}x{h}"
            hud_text_help = "Status: CAMERA READY | Press 'q' to exit"

            cv2.putText(frame, hud_text_fps, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, hud_text_help, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            # Check key press (q or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] User requested exit. Closing camera stream...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and preview window closed cleanly.")

    return 0


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Webcam preview tool for SmartDoorLock.")
    parser.add_argument("--camera-id", type=int, default=settings.camera_index, help="Webcam device index")
    parser.add_argument("--width", type=int, default=settings.camera_frame_width, help="Frame width")
    parser.add_argument("--height", type=int, default=settings.camera_frame_height, help="Frame height")
    parser.add_argument("--fps", type=int, default=settings.camera_fps, help="Target FPS")
    args = parser.parse_args()

    sys.exit(run_preview(args.camera_id, args.width, args.height, args.fps))


if __name__ == "__main__":
    main()

