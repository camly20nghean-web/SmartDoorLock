"""Face detection, alignment, feature extraction, and matching using OpenCV YuNet + SFace.

Implements official OpenCV Zoo models:
- Detection: YuNet (CNN lightweight face detector with 5 landmarks)
- Recognition: SFace (SphereFace 128-dimensional L2-normalized embedding)
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .base import BaseBiometricExtractor, BiometricMatchResult

logger = logging.getLogger(__name__)

# Model metadata constants
YUNET_DEFAULT_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_DEFAULT_NAME = "face_recognition_sface_2021dec.onnx"
DEFAULT_MODEL_VERSION = "opencv_sface_2021dec"
DEFAULT_FEATURE_DIMENSION = 128


@dataclass(frozen=True)
class FaceQualityAssessment:
    """Quality metrics for face image frames."""
    brightness: float
    sharpness: float
    width: int
    height: int
    is_acceptable: bool
    rejection_reason: str = ""


class FaceRecognitionEngine(BaseBiometricExtractor):
    """
    OpenCV DNN based Face Recognition Engine.
    
    Implements YuNet for robust 5-landmark face detection and SFace for 128-D embedding extraction.
    Decoupled from access control policy decisions.
    """

    def __init__(
        self,
        model_yunet_path: Optional[str] = None,
        model_sface_path: Optional[str] = None,
        similarity_threshold: float = 0.60,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.30
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

        # Resolve model paths
        project_root = Path(__file__).resolve().parent.parent.parent
        weights_dir = project_root / "weights"

        self.yunet_path = Path(model_yunet_path) if model_yunet_path else (weights_dir / YUNET_DEFAULT_NAME)
        self.sface_path = Path(model_sface_path) if model_sface_path else (weights_dir / SFACE_DEFAULT_NAME)

        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._current_input_size: Tuple[int, int] = (320, 320)

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load ONNX networks into OpenCV DNN."""
        if not self.yunet_path.exists():
            logger.warning("YuNet model weights not found at '%s'. Run scripts/download_models.py.", self.yunet_path)
            return

        if not self.sface_path.exists():
            logger.warning("SFace model weights not found at '%s'. Run scripts/download_models.py.", self.sface_path)
            return

        try:
            # Read ONNX models into C-contiguous NumPy uint8 buffers to avoid Windows UTF-8 path issues
            with open(self.yunet_path, "rb") as f:
                yunet_buffer = np.frombuffer(f.read(), dtype=np.uint8)

            with open(self.sface_path, "rb") as f:
                sface_buffer = np.frombuffer(f.read(), dtype=np.uint8)

            self._detector = cv2.FaceDetectorYN.create(
                framework="onnx",
                bufferModel=yunet_buffer,
                bufferConfig=b"",
                input_size=self._current_input_size,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000
            )

            self._recognizer = cv2.FaceRecognizerSF.create(
                framework="onnx",
                bufferModel=sface_buffer,
                bufferConfig=b""
            )
            logger.info("FaceRecognitionEngine initialized successfully with YuNet and SFace buffers.")
        except Exception as e:
            logger.error("Failed to initialize OpenCV DNN models: %s", e)
            self._detector = None
            self._recognizer = None

    @property
    def is_ready(self) -> bool:
        """True if detector and recognizer DNN nets are loaded."""
        return self._detector is not None and self._recognizer is not None

    def detect_all_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Detect all faces in an input BGR frame.
        
        Returns:
            np.ndarray: 2D array of shape (N, 15), or empty array if no faces found.
        """
        if not self.is_ready or self._detector is None:
            raise RuntimeError("FaceRecognitionEngine models are not loaded. Run scripts/download_models.py.")

        if image is None or image.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        h, w = image.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)

        return faces

    def detect_single_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Detect and enforce exactly ONE face in the frame.
        
        Returns:
            Tuple[Optional[np.ndarray], str]: (face_data_array, status_reason)
        """
        faces = self.detect_all_faces(image)
        count = len(faces)

        if count == 0:
            return None, "No face detected in frame."
        if count > 1:
            return None, f"Security violation: Multiple faces detected ({count} faces). Exactly 1 face required."

        return faces[0], "OK"

    def assess_face_quality(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        min_face_size: int = 60,
        min_brightness: float = 35.0,
        max_brightness: float = 230.0,
        min_sharpness: float = 40.0
    ) -> FaceQualityAssessment:
        """
        Evaluate facial crop quality metrics: illumination, sharpness (Laplacian variance), and resolution.
        """
        x, y, w, h = [int(v) for v in face_box[:4]]
        img_h, img_w = image.shape[:2]

        # Bound coordinates safely inside frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return FaceQualityAssessment(0.0, 0.0, 0, 0, False, "Corrupted bounding box coordinates.")

        face_roi = image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # 1. Illumination check
        brightness = float(np.mean(gray_roi))
        if brightness < min_brightness:
            return FaceQualityAssessment(brightness, 0.0, w, h, False, f"Lighting too dark (brightness: {brightness:.1f} < {min_brightness}).")
        if brightness > max_brightness:
            return FaceQualityAssessment(brightness, 0.0, w, h, False, f"Lighting overexposed (brightness: {brightness:.1f} > {max_brightness}).")

        # 2. Sharpness / Blur check via Laplacian variance
        sharpness = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        if sharpness < min_sharpness:
            return FaceQualityAssessment(brightness, sharpness, w, h, False, f"Image is blurry (sharpness: {sharpness:.1f} < {min_sharpness}).")

        # 3. Minimum resolution check
        if w < min_face_size or h < min_face_size:
            return FaceQualityAssessment(brightness, sharpness, w, h, False, f"Face size too small ({w}x{h} < {min_face_size}px). Move closer to camera.")

        return FaceQualityAssessment(brightness, sharpness, w, h, True, "Quality acceptable.")

    def align_and_crop(self, image: np.ndarray, face_box: np.ndarray) -> np.ndarray:
        """
        Perform 5-point landmark geometric alignment and crop using official SFace alignCrop.
        
        Returns:
            np.ndarray: Aligned face chip normalized to 112x112 BGR uint8.
        """
        if not self.is_ready or self._recognizer is None:
            raise RuntimeError("SFace recognizer model is not loaded.")

        aligned_face = self._recognizer.alignCrop(image, face_box)
        return aligned_face

    def detect_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect single face and crop aligned 112x112 ROI.
        
        Implements BaseBiometricExtractor.detect_roi.
        """
        face_box, reason = self.detect_single_face(image)
        if face_box is None:
            logger.debug("ROI detection failed: %s", reason)
            return None

        return self.align_and_crop(image, face_box)

    def extract_embedding(self, roi_or_aligned_image: np.ndarray) -> np.ndarray:
        """
        Extract 128-dimensional L2-normalized float32 embedding vector.
        
        Implements BaseBiometricExtractor.extract_embedding.
        """
        if not self.is_ready or self._recognizer is None:
            raise RuntimeError("SFace recognizer model is not loaded.")

        if roi_or_aligned_image is None or roi_or_aligned_image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # Ensure image is exactly 112x112 for SFace input
        if roi_or_aligned_image.shape[:2] != (112, 112):
            roi_or_aligned_image = cv2.resize(roi_or_aligned_image, (112, 112), interpolation=cv2.INTER_LINEAR)

        feature = self._recognizer.feature(roi_or_aligned_image)
        vec = feature.flatten().astype(np.float32)

        # Ensure L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return vec

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two 128-D L2-normalized embedding vectors.
        
        Formula:
            Cosine Similarity = (u · v) / (||u|| * ||v||)
        Range: [-1.0, 1.0]
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        v1 = np.asarray(embedding1, dtype=np.float32).flatten()
        v2 = np.asarray(embedding2, dtype=np.float32).flatten()

        if v1.shape != (DEFAULT_FEATURE_DIMENSION,) or v2.shape != (DEFAULT_FEATURE_DIMENSION,):
            logger.warning("Dimension mismatch in similarity computation: %s vs %s", v1.shape, v2.shape)
            return 0.0

        if np.isnan(v1).any() or np.isnan(v2).any() or np.isinf(v1).any() or np.isinf(v2).any():
            logger.warning("Invalid numerical values (NaN/Inf) detected in embeddings.")
            return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(-1.0, min(1.0, cosine_sim))

    def verify(
        self,
        probe_image: np.ndarray,
        enrolled_embedding: np.ndarray
    ) -> BiometricMatchResult:
        """
        End-to-end verification comparing a camera probe image against an enrolled embedding vector.
        
        Implements BaseBiometricExtractor.verify.
        """
        face_box, detect_reason = self.detect_single_face(probe_image)
        if face_box is None:
            return BiometricMatchResult(
                is_matched=False,
                similarity_score=0.0,
                confidence_level="ZERO",
                message=detect_reason
            )

        # Quality check
        quality = self.assess_face_quality(probe_image, face_box)
        if not quality.is_acceptable:
            return BiometricMatchResult(
                is_matched=False,
                similarity_score=0.0,
                confidence_level="LOW",
                message=f"Rejected: {quality.rejection_reason}"
            )

        aligned = self.align_and_crop(probe_image, face_box)
        probe_embedding = self.extract_embedding(aligned)

        score = self.compute_similarity(probe_embedding, enrolled_embedding)
        is_matched = bool(score >= self.similarity_threshold)

        confidence = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= self.similarity_threshold else "LOW")
        msg = f"Cosine similarity: {score:.3f} (Threshold: {self.similarity_threshold:.2f})"

        return BiometricMatchResult(
            is_matched=is_matched,
            similarity_score=score,
            confidence_level=confidence,
            message=msg
        )
