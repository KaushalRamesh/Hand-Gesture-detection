"""
Automated Verification and Unit Test Suite
Tests YOLO detection, MediaPipe extraction, gesture geometry rules, swipe detection, smoothing, and controllers.
"""

import math
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig, YOLO_MODEL_PATH, MEDIAPIPE_MODEL_PATH
from detection.yolo_detector import YOLOHandDetector, HandBox
from landmarks.hand_landmarks import (
    HandLandmarkerWrapper,
    HandLandmarksData,
    LandmarkPoint,
    WRIST,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)
from gestures.gesture_recognition import (
    GestureClassifier,
    GestureType,
    detect_gesture,
    get_finger_states,
    is_pinch,
    is_open_palm,
    is_fist,
    is_pointing,
    is_two_fingers,
    is_thumb_up,
    is_thumb_down,
)
from gestures.swipe_detector import SwipeDetector, SwipeDirection
from utils.smoothing import ExponentialSmoother2D
from utils.fps import FPSTracker


def create_synthetic_landmarks(
    thumb_ext=True,
    index_ext=True,
    middle_ext=True,
    ring_ext=True,
    pinky_ext=True,
    is_pinch_pose=False,
    thumb_dir="neutral",
) -> list[LandmarkPoint]:
    """Builds a geometrically accurate set of 21 landmarks for testing rules."""
    pts = [LandmarkPoint(x=0.5, y=0.8, z=0.0, px=320, py=384)]  # 0: Wrist

    # Base MCP positions (x, y)
    mcps = {
        1: (0.42, 0.72),  # Thumb CMC
        2: (0.38, 0.65),  # Thumb MCP
        5: (0.44, 0.55),  # Index MCP
        9: (0.50, 0.53),  # Middle MCP
        13: (0.56, 0.55), # Ring MCP
        17: (0.62, 0.58), # Pinky MCP
    }

    # Helper to construct 4 points per finger
    # Thumb (1, 2, 3, 4)
    if thumb_dir == "up":
        # Thumb pointing straight up (y decreases significantly)
        pts.append(LandmarkPoint(x=0.45, y=0.70, z=0, px=288, py=336))
        pts.append(LandmarkPoint(x=0.45, y=0.62, z=0, px=288, py=297))
        pts.append(LandmarkPoint(x=0.45, y=0.50, z=0, px=288, py=240))
        pts.append(LandmarkPoint(x=0.45, y=0.35, z=0, px=288, py=168))
    elif thumb_dir == "down":
        # Thumb pointing straight down (y increases significantly past wrist)
        pts.append(LandmarkPoint(x=0.45, y=0.72, z=0, px=288, py=345))
        pts.append(LandmarkPoint(x=0.45, y=0.78, z=0, px=288, py=374))
        pts.append(LandmarkPoint(x=0.45, y=0.88, z=0, px=288, py=422))
        pts.append(LandmarkPoint(x=0.45, y=0.98, z=0, px=288, py=470))
    elif is_pinch_pose:
        # Thumb touches index tip
        pts.append(LandmarkPoint(x=0.42, y=0.72, z=0, px=268, py=345))
        pts.append(LandmarkPoint(x=0.40, y=0.64, z=0, px=256, py=307))
        pts.append(LandmarkPoint(x=0.42, y=0.54, z=0, px=268, py=259))
        pts.append(LandmarkPoint(x=0.44, y=0.45, z=0, px=281, py=216))  # Thumb tip near index tip
    elif thumb_ext:
        pts.append(LandmarkPoint(x=0.40, y=0.70, z=0, px=256, py=336))
        pts.append(LandmarkPoint(x=0.34, y=0.62, z=0, px=217, py=297))
        pts.append(LandmarkPoint(x=0.28, y=0.54, z=0, px=179, py=259))
        pts.append(LandmarkPoint(x=0.22, y=0.46, z=0, px=140, py=220))  # Thumb tip extended
    else:
        # Thumb curled
        pts.append(LandmarkPoint(x=0.44, y=0.72, z=0, px=281, py=345))
        pts.append(LandmarkPoint(x=0.43, y=0.66, z=0, px=275, py=316))
        pts.append(LandmarkPoint(x=0.45, y=0.62, z=0, px=288, py=297))
        pts.append(LandmarkPoint(x=0.48, y=0.60, z=0, px=307, py=288))

    # Helper for fingers 5..20
    finger_configs = [
        (5, index_ext, 0.44, is_pinch_pose),
        (9, middle_ext, 0.50, False),
        (13, ring_ext, 0.56, False),
        (17, pinky_ext, 0.62, False),
    ]

    for mcp_idx, is_extended, base_x, is_pinch_finger in finger_configs:
        if is_pinch_finger:
            pts.append(LandmarkPoint(x=base_x, y=0.55, z=0, px=int(base_x*640), py=264)) # MCP
            pts.append(LandmarkPoint(x=base_x, y=0.51, z=0, px=int(base_x*640), py=244)) # PIP
            pts.append(LandmarkPoint(x=base_x, y=0.48, z=0, px=int(base_x*640), py=230)) # DIP
            pts.append(LandmarkPoint(x=base_x, y=0.45, z=0, px=int(base_x*640), py=216)) # TIP touches thumb tip
        elif is_extended:
            pts.append(LandmarkPoint(x=base_x, y=0.55, z=0, px=int(base_x*640), py=264)) # MCP
            pts.append(LandmarkPoint(x=base_x, y=0.45, z=0, px=int(base_x*640), py=216)) # PIP
            pts.append(LandmarkPoint(x=base_x, y=0.35, z=0, px=int(base_x*640), py=168)) # DIP
            pts.append(LandmarkPoint(x=base_x, y=0.25, z=0, px=int(base_x*640), py=120)) # TIP
        else:
            # Curled towards palm
            pts.append(LandmarkPoint(x=base_x, y=0.55, z=0, px=int(base_x*640), py=264)) # MCP
            pts.append(LandmarkPoint(x=base_x, y=0.62, z=0, px=int(base_x*640), py=297)) # PIP
            pts.append(LandmarkPoint(x=base_x, y=0.68, z=0, px=int(base_x*640), py=326)) # DIP
            pts.append(LandmarkPoint(x=base_x, y=0.65, z=0, px=int(base_x*640), py=312)) # TIP curled

    return pts


class TestGestureDetectionSystem(unittest.TestCase):
    def test_open_palm_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.OPEN_PALM)
        self.assertGreater(conf, 0.8)

    def test_fist_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.FIST)
        self.assertGreater(conf, 0.8)

    def test_pointing_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.POINTING)

    def test_two_fingers_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.TWO_FINGERS)

    def test_pinch_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, is_pinch_pose=True)
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.PINCH)

    def test_thumb_up_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_dir="up")
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.THUMB_UP)

    def test_thumb_down_gesture(self):
        lm = create_synthetic_landmarks(thumb_ext=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_dir="down")
        gest, conf = detect_gesture(lm)
        self.assertEqual(gest, GestureType.THUMB_DOWN)

    def test_swipe_detector(self):
        detector = SwipeDetector(displacement_threshold=0.15, cooldown_seconds=0.5, history_len=15)
        # Simulate moving from left (x=0.3) to right (x=0.6) across 0.25 seconds
        import time
        t0 = time.perf_counter()

        # Add initial positions
        detector.history.append((t0 - 0.25, 0.30, 0.50))
        detector.history.append((t0 - 0.15, 0.45, 0.50))

        result = detector.update(0.60, 0.50)
        self.assertTrue(result.triggered)
        self.assertEqual(result.direction, SwipeDirection.SWIPE_RIGHT)

    def test_exponential_smoother(self):
        smoother = ExponentialSmoother2D(alpha=0.5, deadzone=0.005)
        # First point
        x1, y1 = smoother.update(100.0, 100.0)
        self.assertEqual((x1, y1), (100.0, 100.0))

        # Second point should interpolate
        x2, y2 = smoother.update(200.0, 200.0)
        self.assertGreater(x2, 100.0)
        self.assertLess(x2, 200.0)

    def test_yolo_model_initialization(self):
        detector = YOLOHandDetector(YOLO_MODEL_PATH)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector.detect(dummy_frame)
        self.assertIsInstance(boxes, list)

    def test_mediapipe_wrapper_initialization(self):
        wrapper = HandLandmarkerWrapper(MEDIAPIPE_MODEL_PATH)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = wrapper.extract_from_crop(dummy_frame, (100, 100, 300, 300))
        self.assertIsNone(res)  # Blank image has no hand landmarks


if __name__ == "__main__":
    unittest.main()
