"""Collect and compare local palm descriptors for Milestone C."""
import argparse
from pathlib import Path
import re
import sys
from uuid import uuid4

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biometrics.palm import (
    DEFAULT_ROI_SIZE,
    MAX_MEAN_ILLUMINATION,
    MIN_MEAN_ILLUMINATION,
    MIN_PALM_WIDTH_PX,
    MIN_SHARPNESS_VAR,
    PALM_FEATURE_DIMENSION,
    PALM_FEATURE_VERSION,
    PalmprintRecognitionEngine,
)
from src.capture.webcam import WebcamCamera

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "palm_milestone_c"


def measure_roi(frame, result):
    """Measure a unit-scale crop of the same region used by the engine."""
    points = result.landmarks_2d
    palm_width = result.quality.palm_width_px
    palm_length = np.linalg.norm((points[5] + points[17]) / 2 - points[0])
    # Match the engine's ROI geometry. Round only the diagnostic raster size.
    source_side = float(0.65 * min(palm_width, palm_length))
    side = max(1, round(source_side))
    center = np.asarray(result.center_px)
    transform = cv2.getRotationMatrix2D(tuple(center), result.rotation_angle_deg, 1.0)
    transform[:, 2] += side / 2 - center
    native = cv2.warpAffine(frame, transform, (side, side), borderMode=cv2.BORDER_REPLICATE)
    native_gray = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)
    native_sharpness = cv2.Laplacian(native_gray, cv2.CV_64F).var()
    height, width = frame.shape[:2]
    return width, height, source_side, native_sharpness


def capture(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.subject) or args.count < 1:
        raise ValueError("Use an anonymous subject code and a positive sample count.")

    camera = WebcamCamera(camera_index=args.camera_id)
    if not camera.connect():
        raise RuntimeError(f"Cannot open camera {args.camera_id}.")

    embeddings = []
    sharpness = []
    diagnostics = []
    window = "Milestone C - Space: capture, Q/Esc: finish"
    try:
        with PalmprintRecognitionEngine() as engine:
            while len(embeddings) < args.count:
                ok, frame = camera.get_frame()
                if not ok or frame is None:
                    print("Camera stopped delivering frames.")
                    break

                result = engine.detect_hand_and_roi(frame)
                quality = result.quality
                allowed = (
                    result.roi_image is not None
                    and result.handedness == args.hand
                    and result.is_palmar
                    and quality is not None
                    and quality.palm_width_px >= MIN_PALM_WIDTH_PX
                    and MIN_MEAN_ILLUMINATION <= quality.mean_illumination <= MAX_MEAN_ILLUMINATION
                )
                measurement = None
                if args.condition and result.roi_image is not None:
                    measurement = measure_roi(frame, result)
                if result.roi_box_corners is not None:
                    cv2.polylines(frame, [result.roi_box_corners], True, (0, 220, 255), 2)
                status = "Show the selected palm, centered and facing the camera."
                if quality is not None:
                    status = f"{result.handedness} | Sharpness {quality.sharpness:.1f}"
                    status += " | Space to save" if allowed else " | Reposition hand"
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                cv2.putText(frame, f"{args.role} {len(embeddings)}/{args.count}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                if measurement is not None:
                    width, height, side, native_sharpness = measurement
                    detail = (
                        f"{width}x{height} | ROI {side:.1f}px -> {DEFAULT_ROI_SIZE}px"
                        f" | Sharpness {native_sharpness:.2f} -> {quality.sharpness:.2f}"
                    )
                    cv2.putText(frame, detail, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and allowed:
                    embeddings.append(engine.extract_embedding(result.roi_image))
                    sharpness.append(quality.sharpness)
                    print(f"Capture {len(embeddings)}: Sharpness {quality.sharpness:.2f}")
                    if measurement is not None:
                        diagnostics.append(measurement)
                        print(detail)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not embeddings:
        print("No samples saved.")
        return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    session = uuid4().hex
    path = args.data_dir / f"{session}.npz"
    extra = {}
    if diagnostics:
        measurements = np.asarray(diagnostics)
        extra = {
            "condition": args.condition,
            "frame_width": measurements[:, 0].astype(int),
            "frame_height": measurements[:, 1].astype(int),
            "source_roi_side": measurements[:, 2],
            "native_sharpness": measurements[:, 3],
        }
    with path.open("xb") as output:
        np.savez_compressed(
            output,
            version=PALM_FEATURE_VERSION,
            subject=args.subject.upper(),
            hand=args.hand,
            role=args.role,
            session=session,
            capture_number=np.arange(1, len(embeddings) + 1),
            sharpness=np.asarray(sharpness),
            embeddings=np.stack(embeddings),
            **extra,
        )
    print(f"Saved {len(embeddings)} descriptors to {path}")
    return 0


def evaluate(args):
    sessions = {"reference": [], "probe": []}
    seen_sessions = set()
    conditions = {}
    for path in sorted(args.data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if data["version"].item() != PALM_FEATURE_VERSION:
                raise ValueError(f"Incompatible feature version in {path.name}.")
            role, session = data["role"].item(), data["session"].item()
            if role not in sessions or session in seen_sessions:
                raise ValueError(f"Invalid role or duplicate session in {path.name}.")
            seen_sessions.add(session)
            vectors, sharpness = data["embeddings"], data["sharpness"]
            if (
                vectors.ndim != 2 or vectors.shape[1] != PALM_FEATURE_DIMENSION
                or sharpness.shape != (len(vectors),) or not len(vectors)
                or not np.isfinite(vectors).all() or not np.isfinite(sharpness).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
            ):
                raise ValueError(f"Invalid sample data in {path.name}.")
            condition = data["condition"].item() if "condition" in data else None
            sessions[role].append((data["subject"].item(), data["hand"].item(), vectors, sharpness, condition))
            if condition:
                batch = np.column_stack(
                    (data["frame_width"], data["frame_height"], data["source_roi_side"],
                     data["native_sharpness"], sharpness)
                )
                conditions.setdefault((role, condition), []).append(batch)

    for role, batches in sessions.items():
        print(f"{role}: {sum(len(vectors) for _, _, vectors, _, _ in batches)} samples in {len(batches)} sessions")
    for (role, condition), batches in sorted(conditions.items()):
        metrics = np.concatenate(batches)
        resolutions = ", ".join(f"{w:g}x{h:g}" for w, h in np.unique(metrics[:, :2], axis=0))
        side, native, standard = np.median(metrics[:, 2:], axis=0)
        print(f"{role} {condition}: n={len(metrics)}, frame {resolutions}, "
              f"median ROI {side:.1f}px, sharpness {native:.2f} native / {standard:.2f} at {DEFAULT_ROI_SIZE}px")

    scores = {(kind, quality): [] for kind in ("same person", "different people")
              for quality in ("both sharp", "at least one below threshold")}
    within_person = {}
    for ref_subject, ref_hand, ref_vectors, ref_sharpness, ref_condition in sessions["reference"]:
        for probe_subject, probe_hand, probe_vectors, probe_sharpness, probe_condition in sessions["probe"]:
            if ref_hand != probe_hand:
                continue
            kind = "same person" if ref_subject == probe_subject else "different people"
            if args.within_person_only and kind == "different people":
                continue
            similarity = np.clip(ref_vectors @ probe_vectors.T, 0.0, 1.0)
            if kind == "same person" and ref_condition and probe_condition:
                key = (ref_subject, ref_hand, ref_condition, probe_condition)
                within_person.setdefault(key, []).extend(similarity.ravel())
            both_sharp = (
                (ref_sharpness[:, None] >= MIN_SHARPNESS_VAR)
                & (probe_sharpness[None, :] >= MIN_SHARPNESS_VAR)
            )
            scores[kind, "both sharp"].extend(similarity[both_sharp])
            scores[kind, "at least one below threshold"].extend(similarity[~both_sharp])

    print(f"Same-hand-side cosine similarity (sharpness threshold {MIN_SHARPNESS_VAR:g})")
    for (kind, quality), values in scores.items():
        if args.within_person_only and kind == "different people":
            continue
        if values:
            print(f"{kind}, {quality}: n={len(values)}, "
                  f"min={min(values):.4f}, median={np.median(values):.4f}, max={max(values):.4f}")
        else:
            print(f"{kind}, {quality}: no pairs")
    for (subject, hand, reference, probe), values in sorted(within_person.items()):
        print(f"{subject} {hand} {reference} -> {probe}: n={len(values)}, "
              f"min={min(values):.4f}, median={np.median(values):.4f}, max={max(values):.4f}")

    if args.within_person_only:
        if any(values for (kind, _), values in scores.items() if kind == "same person"):
            return 0
        print("Capture separate reference and probe sessions with the same subject and hand.")
        return 1
    if not all(any(values for (label, _), values in scores.items() if label == kind)
               for kind in ("same person", "different people")):
        print("Capture reference and probe sessions of the same hand side for at least two people.")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="Capture one session from the webcam")
    capture_parser.add_argument("--subject", required=True)
    capture_parser.add_argument("--hand", choices=("Left", "Right"), required=True)
    capture_parser.add_argument("--role", choices=("reference", "probe"), required=True)
    capture_parser.add_argument("--count", type=int, default=5)
    capture_parser.add_argument("--camera-id", type=int, default=0)
    capture_parser.add_argument("--condition", help="Capture label and enable ROI diagnostics")
    capture_parser.set_defaults(run=capture)
    evaluate_parser = commands.add_parser("evaluate", help="Compare saved reference and probe sessions")
    evaluate_parser.add_argument("--within-person-only", action="store_true")
    evaluate_parser.set_defaults(run=evaluate)
    args = parser.parse_args()
    try:
        return args.run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
