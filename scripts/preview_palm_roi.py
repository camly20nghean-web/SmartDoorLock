"""Visual Inspection Tool for Hand Landmark Detection (Milestone A) and Palm ROI Extraction (Milestone B).

Features:
- Live Webcam feed via WebcamCamera (BaseCamera abstraction).
- 21 3D Landmark skeleton visualization.
- Anatomical MCP joints (5, 9, 13, 17) & Wrist (0) highlight.
- Principal hand axis and rotation angle estimation.
- Palmar vs. Dorsal (Palm vs. Back of hand) surface orientation indicator.
- Oriented Palm ROI bounding polygon overlaid on live hand.
- Picture-in-Picture (PiP) live display of the aligned 128x128 Palm ROI and CLAHE enhanced texture.
- Real-time quality diagnostics: Sharpness, illumination, palm width.

Usage:
    python scripts/preview_palm_roi.py [--camera-id 0]
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

from src.biometrics.palm import PalmprintRecognitionEngine
from src.capture.webcam import WebcamCamera

# MediaPipe Hand Landmark skeleton connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm Knuckle Base (MCP line)
    (5, 9), (9, 13), (13, 17)
]


def run_preview(camera_id: int = 0) -> int:
    camera = WebcamCamera(camera_index=camera_id)
    if not camera.connect():
        print(f"❌ Failed to open camera (ID: {camera_id}).")
        return 1

    try:
        engine = PalmprintRecognitionEngine()
    except Exception as e:
        print(f"❌ Failed to initialize PalmprintRecognitionEngine: {e}")
        camera.release()
        return 1

    window_name = "SmartDoorLock - Milestone A & B: Hand Landmark & Palm ROI Inspection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("=" * 70)
    print("  SMART DOOR LOCK - PALM ROI INSPECTION (Milestones A & B)")
    print("=" * 70)
    print("• Present your hand to the webcam (cự ly 25cm - 40cm).")
    print("• Verify the 21 landmarks, Palm Center, and oriented Palm ROI box.")
    print("• Observe the 128x128 cropped Palm ROI & CLAHE view at top-right.")
    print("• Press 'q' or 'ESC' to exit cleanly.")
    print("=" * 70)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = camera.get_frame()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Analyze hand and extract ROI
            res = engine.detect_hand_and_roi(frame)

            display_frame = frame.copy()

            if res.has_hand and res.landmarks_2d is not None:
                pts = res.landmarks_2d.astype(int)

                # 1. Draw Skeleton Connections
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    pt1 = tuple(pts[p1_idx])
                    pt2 = tuple(pts[p2_idx])
                    cv2.line(display_frame, pt1, pt2, (180, 180, 180), 2)

                # 2. Draw Landmark Points
                for idx, pt in enumerate(pts):
                    # Highlight key joints: 0 (Wrist) in Blue, MCPs (5, 9, 13, 17) in Yellow
                    if idx == 0:
                        cv2.circle(display_frame, tuple(pt), 7, (255, 100, 0), -1)
                        cv2.putText(display_frame, "Wrist", (pt[0] - 25, pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                    elif idx in (5, 9, 13, 17):
                        cv2.circle(display_frame, tuple(pt), 6, (0, 255, 255), -1)
                    else:
                        cv2.circle(display_frame, tuple(pt), 4, (0, 180, 255), -1)

                # 3. Draw Hand Axis (Wrist 0 -> Middle MCP 9)
                cv2.line(display_frame, tuple(pts[0]), tuple(pts[9]), (0, 255, 255), 2)

                # 4. Draw Palm Center and ROI Box
                if res.center_px and res.roi_box_corners is not None:
                    cx, cy = int(res.center_px[0]), int(res.center_px[1])
                    cv2.circle(display_frame, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.putText(display_frame, "Palm Center", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    # Box color based on quality acceptance
                    box_color = (0, 255, 0) if (res.quality and res.quality.is_acceptable) else (0, 140, 255)
                    cv2.polylines(display_frame, [res.roi_box_corners], isClosed=True, color=box_color, thickness=2)

                # 5. Picture-in-Picture (PiP) for Extracted ROI & CLAHE
                if res.roi_image is not None and res.clahe_image is not None:
                    pip_w, pip_h = 120, 120
                    pip_roi = cv2.resize(res.roi_image, (pip_w, pip_h))
                    clahe_bgr = cv2.cvtColor(cv2.resize(res.clahe_image, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)

                    # Top-right PiP placement
                    # Box 1: Aligned Color ROI
                    x_start_1 = w - (pip_w * 2 + 30)
                    y_start = 45
                    display_frame[y_start:y_start + pip_h, x_start_1:x_start_1 + pip_w] = pip_roi
                    cv2.rectangle(display_frame, (x_start_1 - 1, y_start - 1), (x_start_1 + pip_w + 1, y_start + pip_h + 1), (0, 255, 0), 1)
                    cv2.putText(display_frame, "Aligned ROI", (x_start_1, y_start - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                    # Box 2: CLAHE Enhanced Texture
                    x_start_2 = w - (pip_w + 15)
                    display_frame[y_start:y_start + pip_h, x_start_2:x_start_2 + pip_w] = clahe_bgr
                    cv2.rectangle(display_frame, (x_start_2 - 1, y_start - 1), (x_start_2 + pip_w + 1, y_start + pip_h + 1), (0, 255, 0), 1)
                    cv2.putText(display_frame, "CLAHE Texture", (x_start_2, y_start - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # 6. Status Diagnostics Overlay (Top-Left)
                surface_str = "PALMAR (Mat long)" if res.is_palmar else "DORSAL (Mu ban tay)"
                surface_color = (0, 255, 0) if res.is_palmar else (0, 0, 255)

                cv2.putText(display_frame, f"Hand: {res.handedness} ({res.handedness_score * 100:.1f}%)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Surface: {surface_str}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, surface_color, 2)
                cv2.putText(display_frame, f"Hand Tilt: {res.rotation_angle_deg:+.1f} deg", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                if res.quality:
                    cv2.putText(display_frame, f"Sharpness: {res.quality.sharpness:.1f}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.putText(display_frame, f"Illumination: {res.quality.mean_illumination:.1f}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.putText(display_frame, f"Palm Width: {res.quality.palm_width_px:.1f} px", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                    if not res.quality.is_acceptable:
                        for idx, reason in enumerate(res.quality.rejection_reasons):
                            cv2.putText(display_frame, f"WARNING: {reason}", (15, 220 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            else:
                cv2.putText(display_frame, "No hand detected. Please show your palm.", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            # Top Header Bar
            cv2.rectangle(display_frame, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(display_frame, "SmartDoorLock | Milestones A & B: Hand Landmark & Palm ROI", (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 110, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Bottom Footer Bar
            cv2.rectangle(display_frame, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(display_frame, "MediaPipe Hand Landmarker (21 pts) + Anatomical Palmar ROI | Press 'q' or 'ESC' to exit", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Preview exited by user.")
                break

    finally:
        engine.close()
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released cleanly.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Milestone A & B: Hand Landmark and Palm ROI Visualizer.")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    sys.exit(run_preview(camera_id=args.camera_id))


if __name__ == "__main__":
    main()

