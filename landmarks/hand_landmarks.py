"""
MediaPipe Hand Landmark Extractor
Processes detected hand bounding box crops and extracts 21 3D landmarks.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Skeleton connection pairs
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]


@dataclass
class LandmarkPoint:
    # Normalized coords [0..1] in full camera frame
    x: float
    y: float
    z: float
    # Absolute pixel coords in full camera frame
    px: int
    py: int


@dataclass
class HandLandmarksData:
    landmarks: List[LandmarkPoint]  # Length 21
    handedness: str                 # "Left" or "Right"
    handedness_confidence: float
    crop_bbox: Tuple[int, int, int, int]  # (crop_x1, crop_y1, crop_x2, crop_y2)

    def get_pt(self, idx: int) -> LandmarkPoint:
        return self.landmarks[idx]

    @property
    def wrist(self) -> LandmarkPoint:
        return self.landmarks[WRIST]

    @property
    def index_tip(self) -> LandmarkPoint:
        return self.landmarks[INDEX_TIP]

    @property
    def thumb_tip(self) -> LandmarkPoint:
        return self.landmarks[THUMB_TIP]

    @property
    def middle_tip(self) -> LandmarkPoint:
        return self.landmarks[MIDDLE_TIP]


class HandLandmarkerWrapper:
    def __init__(self, model_path: Path, num_hands: int = 2):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe task model not found at: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_inference_time_ms: float = 0.0

    def extract_from_crop(
        self,
        full_frame: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        margin_factor: float = 0.25,
    ) -> Optional[HandLandmarksData]:
        """
        Crops the hand region based on YOLO bounding box, runs MediaPipe HandLandmarker,
        and translates landmarks back to full frame coordinates.
        """
        if full_frame is None or full_frame.size == 0:
            return None

        h, w = full_frame.shape[:2]
        bx1, by1, bx2, by2 = hand_bbox
        bw = bx2 - bx1
        bh = by2 - by1

        # Expand crop box to ensure full fingers & wrist are within view
        pad_x = int(bw * margin_factor)
        pad_y = int(bh * margin_factor)

        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(w, bx2 + pad_x)
        cy2 = min(h, by2 + pad_y)

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        if crop_w < 10 or crop_h < 10:
            return None

        crop_bgr = full_frame[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        t0 = time.perf_counter()
        try:
            result = self.detector.detect(mp_image)
        except Exception as exc:
            print(f"[HandLandmarkerWrapper] Detection error: {exc}")
            return None
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        if not result.hand_landmarks:
            # Fallback: if crop was too tight or off-center, try detecting on full frame
            return self._detect_full_frame(full_frame)

        # Primary detected hand in crop
        raw_landmarks = result.hand_landmarks[0]
        handedness_name = "Right"
        handedness_score = 0.90
        if result.handedness and len(result.handedness) > 0:
            first_cat = result.handedness[0][0]
            handedness_name = first_cat.category_name
            handedness_score = float(first_cat.score)

        landmarks: List[LandmarkPoint] = []
        for lm in raw_landmarks:
            # Local crop coordinate -> Full frame pixel coordinate
            px = int(cx1 + lm.x * crop_w)
            py = int(cy1 + lm.y * crop_h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            # Full-frame normalized coordinates
            norm_x = px / w
            norm_y = py / h
            norm_z = float(lm.z)

            landmarks.append(LandmarkPoint(x=norm_x, y=norm_y, z=norm_z, px=px, py=py))

        return HandLandmarksData(
            landmarks=landmarks,
            handedness=handedness_name,
            handedness_confidence=handedness_score,
            crop_bbox=(cx1, cy1, cx2, cy2),
        )

    def _detect_full_frame(self, full_frame: np.ndarray) -> Optional[HandLandmarksData]:
        """Fallback to detect hand directly in full frame if crop fails."""
        h, w = full_frame.shape[:2]
        frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        try:
            result = self.detector.detect(mp_image)
        except Exception:
            return None

        if not result.hand_landmarks:
            return None

        raw_landmarks = result.hand_landmarks[0]
        handedness_name = "Right"
        handedness_score = 0.90
        if result.handedness and len(result.handedness) > 0:
            first_cat = result.handedness[0][0]
            handedness_name = first_cat.category_name
            handedness_score = float(first_cat.score)

        landmarks: List[LandmarkPoint] = []
        for lm in raw_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            landmarks.append(LandmarkPoint(x=lm.x, y=lm.y, z=float(lm.z), px=px, py=py))

        return HandLandmarksData(
            landmarks=landmarks,
            handedness=handedness_name,
            handedness_confidence=handedness_score,
            crop_bbox=(0, 0, w, h),
        )
