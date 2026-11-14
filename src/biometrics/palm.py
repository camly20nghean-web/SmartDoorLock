"""Palmprint Recognition Engine (Milestones A–C).

Implements:
- Milestone A: MediaPipe Hand Landmarker detection, 21 3D landmarks, Handedness classification.
- Milestone B: Anatomically-grounded Palm Region of Interest (ROI) extraction:
  * Uses Metacarpophalangeal (MCP) joints (Index 5, Middle 9, Pinky 17) and Wrist (0)
  * Orientation estimation and rotation alignment (eliminating hand tilt)
  * Rigorous palmar vs. dorsal (palm vs. back of hand) surface orientation verification
  * Canonical left/right hand normalization
  * Quality assessment (sharpness, illumination, minimum scale)
  * Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Milestone C: Spatial Local Binary Pattern (LBP) descriptor.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mediapipe as mp

from .base import BaseBiometricExtractor, BiometricMatchResult

# Dimensions and Default Paths
DEFAULT_ROI_SIZE = 128
PALM_FEATURE_VERSION = "palm-lbp-grid4-clahe-v1"
LBP_GRID_SIZE = 4
LBP_BINS = 256
PALM_FEATURE_DIMENSION = LBP_GRID_SIZE ** 2 * LBP_BINS
DEFAULT_MODEL_FILENAME = "hand_landmarker.task"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / DEFAULT_MODEL_FILENAME

# Quality Thresholds
MIN_SHARPNESS_VAR = 50.0       # Minimum Laplacian variance (blur detection)
MIN_MEAN_ILLUMINATION = 35.0   # Minimum acceptable average brightness
MAX_MEAN_ILLUMINATION = 220.0  # Maximum acceptable average brightness
MIN_PALM_WIDTH_PX = 50.0       # Minimum palm width between Index and Pinky MCPs


@dataclass
class PalmQualityAssessment:
    """Assessment of hand and palm image quality."""
    is_acceptable: bool
    sharpness: float
    mean_illumination: float
    palm_width_px: float
    is_palmar: bool
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class PalmROIResult:
    """Detailed outcome of Hand Detection & Palm ROI Extraction."""
    has_hand: bool
    landmarks_2d: Optional[np.ndarray] = None          # (21, 2) in image pixel coordinates
    landmarks_3d: Optional[np.ndarray] = None          # (21, 3) normalized coordinates
    handedness: Optional[str] = None                   # 'Left' or 'Right'
    handedness_score: float = 0.0
    is_palmar: bool = False                            # True = palm facing camera, False = back of hand
    roi_image: Optional[np.ndarray] = None             # (128, 128, 3) BGR aligned palm ROI
    clahe_image: Optional[np.ndarray] = None           # (128, 128) Grayscale CLAHE enhanced
    center_px: Optional[Tuple[float, float]] = None    # (x, y) center of palm ROI in original image
    roi_box_corners: Optional[np.ndarray] = None       # (4, 2) polygon corners on original image
    rotation_angle_deg: float = 0.0                    # In-plane rotation angle (degrees)
    quality: Optional[PalmQualityAssessment] = None


class PalmprintRecognitionEngine(BaseBiometricExtractor):
    """
    Palmprint Biometric Authentication Engine.
    
    Adheres strictly to the BaseBiometricExtractor contract:
    - Milestone A: Detects single hand and extracts 21 geometric landmarks.
    - Milestone B: Performs geometric alignment and extracts central palmar ROI.
    - Milestone C: Extracts spatial LBP histograms from the aligned palm ROI.
    - Milestone D: Matching (pending).
    """

    def __init__(self, model_path: Optional[Path] = None, num_hands: int = 2):
        resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Hand Landmarker model not found at '{resolved_path}'. "
                "Please run: python scripts/download_models.py"
            )

        # Read model as byte buffer to bypass Windows path encoding issues
        model_buffer = resolved_path.read_bytes()
        base_options = mp_python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    def close(self):
        """Release underlying MediaPipe detector resources cleanly."""
        if self._detector:
            self._detector.close()
            self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def detect_hand_and_roi(
        self, image: np.ndarray, *, min_sharpness: float = MIN_SHARPNESS_VAR
    ) -> PalmROIResult:
        """
        Execute full hand detection and anatomically-grounded ROI extraction.
        
        Args:
            image (np.ndarray): Input BGR frame.
            min_sharpness (float): Minimum Laplacian variance for this capture.
            
        Returns:
            PalmROIResult: Complete analysis with landmarks, ROI, and quality metrics.
        """
        if image is None or image.size == 0:
            return PalmROIResult(has_hand=False)

        h, w = image.shape[:2]

        # 1. Convert BGR image to RGB for MediaPipe Tasks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 2. Run detection
        detection_result = self._detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return PalmROIResult(has_hand=False)

        # 3. Security check: Single hand enforcement
        if len(detection_result.hand_landmarks) > 1:
            quality = PalmQualityAssessment(
                is_acceptable=False,
                sharpness=0.0,
                mean_illumination=0.0,
                palm_width_px=0.0,
                is_palmar=False,
                rejection_reasons=["Multiple hands detected. Only single hand is permitted for authentication."]
            )
            return PalmROIResult(has_hand=True, quality=quality)

        # 4. Extract landmarks & handedness
        norm_landmarks = detection_result.hand_landmarks[0]
        handedness_cat = detection_result.handedness[0][0]
        handedness = handedness_cat.category_name  # 'Left' or 'Right'
        handedness_score = handedness_cat.score

        # Convert normalized coordinates to pixel space
        landmarks_2d = np.zeros((21, 2), dtype=np.float32)
        landmarks_3d = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(norm_landmarks):
            landmarks_2d[i] = [lm.x * w, lm.y * h]
            landmarks_3d[i] = [lm.x, lm.y, lm.z]

        # 5. Anatomical keypoints
        # 0: Wrist
        # 5: Index Finger MCP joint
        # 9: Middle Finger MCP joint
        # 13: Ring Finger MCP joint
        # 17: Pinky Finger MCP joint
        p_wrist = landmarks_2d[0]
        p_index_mcp = landmarks_2d[5]
        p_middle_mcp = landmarks_2d[9]
        p_pinky_mcp = landmarks_2d[17]

        # 6. Verify Palmar vs. Dorsal (Palm vs. Back of hand) orientation
        # MediaPipe provides handedness in a mirror-corrected coordinate frame.
        # We determine palmar orientation via the cross product of palm surface vectors:
        # V1: from Wrist (0) to Index MCP (5)
        # V2: from Wrist (0) to Pinky MCP (17)
        # Normal vector Z component: (V1_x * V2_y - V1_y * V2_x)
        # For an anatomical Right hand facing camera: Index MCP is to the camera's Left (smaller X than Pinky)
        # For an anatomical Left hand facing camera: Index MCP is to the camera's Right (larger X than Pinky)
        v1 = landmarks_3d[5] - landmarks_3d[0]
        v2 = landmarks_3d[17] - landmarks_3d[0]
        normal_z = v1[0] * v2[1] - v1[1] * v2[0]

        # Check surface orientation consistency with handedness
        if handedness == "Right":
            is_palmar = normal_z < 0.0
        else:
            is_palmar = normal_z > 0.0

        # 7. Compute Hand Principal Axis and Rotation Angle
        # Vector from Wrist (0) to Middle Finger MCP (9) represents the long axis of the hand
        axis_vec = p_middle_mcp - p_wrist
        # In image coordinates, Y increases downward. An upright hand has Middle MCP above Wrist (dy < 0).
        # We compute angle with vertical (0, -1):
        angle_rad = math.atan2(axis_vec[0], -axis_vec[1])
        angle_deg = math.degrees(angle_rad)

        # 8. Define Anatomical Palm Center and ROI Dimensions
        # Central MCP base midpoint
        p_mcp_mid = (p_index_mcp + p_pinky_mcp) / 2.0
        # The anatomical palm center lies along the line connecting Wrist to MCP midpoint,
        # approximately 52% from the wrist (below the knuckle crease, strictly in the palm basin).
        center_px = p_wrist + 0.52 * (p_mcp_mid - p_wrist)

        # Distance between Index MCP and Pinky MCP defines the palm breadth
        palm_width = float(np.linalg.norm(p_index_mcp - p_pinky_mcp))
        # Distance from Wrist to MCP midpoint defines the palm length
        palm_length = float(np.linalg.norm(p_mcp_mid - p_wrist))

        # Size of ROI square: 65% of the minimum dimension
        # This rigorously excludes fingers (which start distal to MCPs) and the wrist crease.
        roi_side = min(palm_width, palm_length) * 0.65

        # 9. Affine Transformation: Rotate to upright and crop ROI square
        # A rightward tilt needs a counterclockwise rotation in OpenCV.
        rot_mat = cv2.getRotationMatrix2D(tuple(center_px), angle_deg, 1.0)
        # Adjust rotation matrix to map center_px to center of a new (roi_side, roi_side) canvas
        rot_mat[0, 2] += (DEFAULT_ROI_SIZE / 2.0) - center_px[0]
        rot_mat[1, 2] += (DEFAULT_ROI_SIZE / 2.0) - center_px[1]

        # Scale factor from roi_side to standardized DEFAULT_ROI_SIZE
        scale = DEFAULT_ROI_SIZE / max(1.0, roi_side)
        rot_mat[0, :] *= scale
        rot_mat[1, :] *= scale
        rot_mat[0, 2] += (1.0 - scale) * (DEFAULT_ROI_SIZE / 2.0)
        rot_mat[1, 2] += (1.0 - scale) * (DEFAULT_ROI_SIZE / 2.0)

        roi_aligned = cv2.warpAffine(
            image,
            rot_mat,
            (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Canonical handedness normalization:
        # If Left hand, flip horizontally so thenar eminence (thumb base) is always on the left
        if handedness == "Left":
            roi_aligned = cv2.flip(roi_aligned, 1)

        # 10. Generate CLAHE Enhanced Grayscale Image for Texture Analysis
        roi_gray = cv2.cvtColor(roi_aligned, cv2.COLOR_BGR2GRAY)
        clahe_enhanced = self._clahe.apply(roi_gray)

        # 11. Compute ROI Box Polygon Coordinates on Original Image
        # Define 4 corners in the canonical upright square frame centered at (0, 0)
        half_side = roi_side / 2.0
        local_corners = np.array([
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ], dtype=np.float32)

        # Rotate local corners by angle_deg and shift to center_px
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        r_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = np.dot(local_corners, r_matrix.T) + center_px
        roi_box_corners = rotated_corners.astype(np.int32)

        # 12. Quality Assessment
        rejection_reasons = []

        # Sharpness check via Laplacian variance
        laplacian_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        if not math.isfinite(laplacian_var):
            rejection_reasons.append("Invalid Laplacian variance.")
        elif laplacian_var < min_sharpness:
            rejection_reasons.append(f"Image blurred: Laplacian variance ({laplacian_var:.1f}) < {min_sharpness}")

        # Illumination check
        mean_lum = float(np.mean(roi_gray))
        if mean_lum < MIN_MEAN_ILLUMINATION:
            rejection_reasons.append(f"Insufficient lighting: Mean brightness ({mean_lum:.1f}) < {MIN_MEAN_ILLUMINATION}")
        elif mean_lum > MAX_MEAN_ILLUMINATION:
            rejection_reasons.append(f"Overexposed/Glared: Mean brightness ({mean_lum:.1f}) > {MAX_MEAN_ILLUMINATION}")

        # Minimum physical palm size check
        if palm_width < MIN_PALM_WIDTH_PX:
            rejection_reasons.append(f"Hand too far: Palm width ({palm_width:.1f}px) < {MIN_PALM_WIDTH_PX}px")

        # Palmar surface check
        if not is_palmar:
            rejection_reasons.append("Back of hand (dorsal) presented. Please present palm facing the camera.")

        quality = PalmQualityAssessment(
            is_acceptable=(len(rejection_reasons) == 0),
            sharpness=laplacian_var,
            mean_illumination=mean_lum,
            palm_width_px=palm_width,
            is_palmar=is_palmar,
            rejection_reasons=rejection_reasons
        )

        return PalmROIResult(
            has_hand=True,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            handedness=handedness,
            handedness_score=handedness_score,
            is_palmar=is_palmar,
            roi_image=roi_aligned,
            clahe_image=clahe_enhanced,
            center_px=(float(center_px[0]), float(center_px[1])),
            roi_box_corners=roi_box_corners,
            rotation_angle_deg=angle_deg,
            quality=quality
        )

    def detect_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Locate and crop standard aligned Palm ROI from raw camera frame.
        
        Returns:
            Optional[np.ndarray]: 128x128 BGR ROI if detected and quality is acceptable, else None.
        """
        res = self.detect_hand_and_roi(image)
        if res.has_hand and res.quality and res.quality.is_acceptable:
            return res.roi_image
        return None

    def extract_embedding(self, roi_image: np.ndarray) -> np.ndarray:
        """Return a unit-length float32 descriptor from a raw aligned BGR ROI."""
        if (
            not isinstance(roi_image, np.ndarray)
            or roi_image.shape != (DEFAULT_ROI_SIZE, DEFAULT_ROI_SIZE, 3)
            or roi_image.dtype != np.uint8
        ):
            raise ValueError("Expected a 128x128 uint8 BGR aligned palm ROI.")

        gray = self._clahe.apply(cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY))
        center = gray[1:-1, 1:-1]
        codes = np.zeros(center.shape, dtype=np.uint8)
        offsets = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
        height, width = center.shape
        for bit, (y, x) in enumerate(offsets):
            neighbor = gray[y:y + height, x:x + width]
            codes |= (neighbor >= center).astype(np.uint8) << bit

        histograms = []
        for row in np.array_split(codes, LBP_GRID_SIZE, axis=0):
            for cell in np.array_split(row, LBP_GRID_SIZE, axis=1):
                counts = np.bincount(cell.ravel(), minlength=LBP_BINS)
                histograms.append(counts / cell.size)

        embedding = np.sqrt(np.concatenate(histograms).astype(np.float32))
        return embedding / np.linalg.norm(embedding)

    def verify(self, probe_image: np.ndarray, enrolled_embedding: np.ndarray) -> BiometricMatchResult:
        """
        Compare captured image against registered user embedding.
        
        Pending Milestone D (SQLite integration and matching).
        """
        raise NotImplementedError("Palmprint verification will be implemented in Milestone D.")
