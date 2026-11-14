"""Helper script to download official OpenCV Zoo ONNX model weights for Face Recognition.

Downloads:
1. YuNet Face Detector (~300 KB): face_detection_yunet_2023mar.onnx
2. SFace Face Recognizer (~37 MB): face_recognition_sface_2021dec.onnx
"""
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "description": "YuNet Face Detector (OpenCV Zoo)",
        "expected_min_bytes": 200_000,
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "description": "SFace Feature Extractor (OpenCV Zoo)",
        "expected_min_bytes": 30_000_000,
    },
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "description": "MediaPipe Hand Landmarker (Google)",
        "expected_min_bytes": 7_000_000,
    },
}


def download_file(url: str, target_path: Path, description: str, min_bytes: int) -> bool:
    """Download file with live progress indicator."""
    if target_path.exists() and target_path.stat().st_size >= min_bytes:
        print(f"✅ {target_path.name} already exists ({target_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return True

    print(f"⬇️ Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {target_path}")

    def progress_callback(blocks, block_size, total_size):
        downloaded = blocks * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   Progress: {percent:5.1f}% [{mb_down:.2f}/{mb_total:.2f} MB]")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(target_path), reporthook=progress_callback)
        sys.stdout.write("\n")
        actual_size = target_path.stat().st_size
        if actual_size < min_bytes:
            print(f"❌ File downloaded is suspiciously small ({actual_size} bytes). Download may have failed.")
            return False
        print(f"✅ Download complete: {target_path.name} ({actual_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        sys.stdout.write("\n")
        print(f"❌ Error downloading {target_path.name}: {e}")
        return False


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  SMART DOOR LOCK - MODEL WEIGHTS DOWNLOADER (Phase 3)")
    print("=" * 65)
    print(f"Target directory: {WEIGHTS_DIR}\n")

    all_success = True
    for filename, meta in MODELS.items():
        dest = WEIGHTS_DIR / filename
        ok = download_file(meta["url"], dest, meta["description"], meta["expected_min_bytes"])
        if not ok:
            all_success = False

    print("=" * 65)
    if all_success:
        print("🎉 All face recognition models are ready for Phase 3!")
    else:
        print("⚠️ Some models failed to download. Please check your internet connection and retry.")
    print("=" * 65)


if __name__ == "__main__":
    main()

