"""
Geometric Rule-Based Hand Gesture Recognition Engine
Analyzes 21 MediaPipe hand landmarks in scale-invariant normalized coordinates.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from landmarks.hand_landmarks import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandLandmarksData,
    LandmarkPoint,
)


class GestureType(str, Enum):
    NONE = "NONE"
    OPEN_PALM = "OPEN PALM"
    FIST = "FIST"
    PINCH = "PINCH"
    POINTING = "POINTING"
    TWO_FINGERS = "TWO FINGERS"
    THUMB_UP = "THUMB UP"
    THUMB_DOWN = "THUMB DOWN"


@dataclass
class GestureResult:
    raw_gesture: GestureType
    confirmed_gesture: GestureType
    should_trigger: bool
    confidence: float
    stability_ratio: float  # Progress towards confirmation (0.0 to 1.0)
    pinch_distance: float = 0.0


def _dist(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Euclidean distance in normalized (x, y) coordinates."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _get_palm_size(lm: List[LandmarkPoint]) -> float:
    """Scale reference: distance from wrist to middle finger MCP."""
    d = _dist(lm[WRIST], lm[MIDDLE_MCP])
    return max(0.04, d)


def get_finger_states(lm: List[LandmarkPoint]) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Evaluates whether (Thumb, Index, Middle, Ring, Pinky) are extended.
    Uses scale-invariant relative distance from wrist and MCP joints.
    """
    wrist = lm[WRIST]
    palm_size = _get_palm_size(lm)

    # Fingers: Index, Middle, Ring, Pinky
    # A finger is extended if tip is significantly further from wrist than PIP and MCP
    def is_ext(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]
        d_tip_wrist = _dist(tip, wrist)
        d_pip_wrist = _dist(pip, wrist)
        d_tip_mcp = _dist(tip, mcp)
        d_pip_mcp = _dist(pip, mcp)
        return (d_tip_wrist > d_pip_wrist * 1.08) and (d_tip_mcp > d_pip_mcp * 1.05) and (d_tip_mcp > palm_size * 0.45)

    index_ext = is_ext(INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_ext = is_ext(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_ext = is_ext(RING_TIP, RING_PIP, RING_MCP)
    pinky_ext = is_ext(PINKY_TIP, PINKY_PIP, PINKY_MCP)

    # Thumb extension: check distance from thumb tip to pinky MCP and wrist
    thumb_tip = lm[THUMB_TIP]
    thumb_ip = lm[THUMB_IP]
    thumb_mcp = lm[THUMB_MCP]
    pinky_mcp = lm[PINKY_MCP]
    d_thumb_pinky = _dist(thumb_tip, pinky_mcp)
    d_thumb_wrist = _dist(thumb_tip, wrist)
    d_thumb_ip_wrist = _dist(thumb_ip, wrist)

    thumb_ext = (d_thumb_pinky > palm_size * 0.85) and (d_thumb_wrist > d_thumb_ip_wrist * 1.05)

    return thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext


def is_pinch(lm: List[LandmarkPoint]) -> Tuple[bool, float]:
    """
    Pinch: Index tip and Thumb tip are close together,
    and index finger is reaching forward towards thumb (not tucked backwards into palm).
    """
    palm_size = _get_palm_size(lm)
    d_thumb_index = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
    ratio = d_thumb_index / palm_size

    # In a fist, index tip is curled down towards palm/wrist
    index_tip = lm[INDEX_TIP]
    index_pip = lm[INDEX_PIP]
    index_curled_into_palm = (index_tip.y > index_pip.y + palm_size * 0.1)

    is_pinching = (ratio < 0.28) and (not index_curled_into_palm)
    return is_pinching, ratio


def is_thumb_up(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """
    Thumb Up: Other 4 fingers curled, thumb pointing straight up in image coordinates (smaller y).
    """
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    if index_ext or middle_ext or ring_ext or pinky_ext:
        return False

    thumb_tip = lm[THUMB_TIP]
    thumb_mcp = lm[THUMB_MCP]
    wrist = lm[WRIST]
    palm_size = _get_palm_size(lm)

    # Tip must be above MCP and wrist by a clear vertical delta
    is_pointing_up = (thumb_tip.y < thumb_mcp.y - palm_size * 0.30) and (thumb_tip.y < wrist.y - palm_size * 0.40)
    # Also verify thumb tip is more vertical than horizontal
    dx = abs(thumb_tip.x - thumb_mcp.x)
    dy = abs(thumb_tip.y - thumb_mcp.y)
    return is_pointing_up and (dy > dx * 0.8)


def is_thumb_down(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """
    Thumb Down: Other 4 fingers curled, thumb pointing straight down in image coordinates (larger y).
    """
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    if index_ext or middle_ext or ring_ext or pinky_ext:
        return False

    thumb_tip = lm[THUMB_TIP]
    thumb_mcp = lm[THUMB_MCP]
    wrist = lm[WRIST]
    palm_size = _get_palm_size(lm)

    # Tip must be below MCP and wrist by a clear vertical delta
    is_pointing_down = (thumb_tip.y > thumb_mcp.y + palm_size * 0.30) and (thumb_tip.y > wrist.y + palm_size * 0.20)
    dx = abs(thumb_tip.x - thumb_mcp.x)
    dy = abs(thumb_tip.y - thumb_mcp.y)
    return is_pointing_down and (dy > dx * 0.8)


def is_open_palm(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """Open Palm: All 4 main fingers extended, thumb extended and spread out."""
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    all_extended = index_ext and middle_ext and ring_ext and pinky_ext
    # Ensure fingers are not pinched
    pinching, _ = is_pinch(lm)
    return all_extended and (thumb_ext or _dist(lm[THUMB_TIP], lm[PINKY_TIP]) > _get_palm_size(lm) * 0.9) and not pinching


def is_fist(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """Fist: All 4 main fingers curled in."""
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    if index_ext or middle_ext or ring_ext or pinky_ext:
        return False
    # Not thumb up or thumb down
    if is_thumb_up(lm, finger_states) or is_thumb_down(lm, finger_states):
        return False
    return True


def is_pointing(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """Pointing: Index extended, Middle/Ring/Pinky curled."""
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    pinching, _ = is_pinch(lm)
    return index_ext and not middle_ext and not ring_ext and not pinky_ext and not pinching


def is_two_fingers(lm: List[LandmarkPoint], finger_states: Tuple[bool, bool, bool, bool, bool]) -> bool:
    """Two Fingers (V / Peace sign): Index and Middle extended, Ring and Pinky curled."""
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    pinching, _ = is_pinch(lm)
    return index_ext and middle_ext and not ring_ext and not pinky_ext and not pinching


def detect_gesture(lm: List[LandmarkPoint]) -> Tuple[GestureType, float]:
    """
    Evaluates rule-based geometric hierarchy to determine instantaneous raw gesture.
    Returns (GestureType, confidence_score).
    """
    if not lm or len(lm) < 21:
        return GestureType.NONE, 0.0

    finger_states = get_finger_states(lm)
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_states
    all_curled = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)

    pinching, pinch_ratio = is_pinch(lm)

    # If all 4 main fingers are curled in
    if all_curled:
        if pinching:
            conf = max(0.6, min(1.0, 1.0 - pinch_ratio * 2.0))
            return GestureType.PINCH, conf
        if is_thumb_up(lm, finger_states):
            return GestureType.THUMB_UP, 0.92
        if is_thumb_down(lm, finger_states):
            return GestureType.THUMB_DOWN, 0.92
        return GestureType.FIST, 0.92

    # 1. Pinch detection (index and thumb touching while at least one finger or hand is open)
    if pinching:
        conf = max(0.6, min(1.0, 1.0 - pinch_ratio * 2.0))
        return GestureType.PINCH, conf

    # 2. Pointing
    if is_pointing(lm, finger_states):
        return GestureType.POINTING, 0.90

    # 3. Two fingers
    if is_two_fingers(lm, finger_states):
        return GestureType.TWO_FINGERS, 0.90

    # 4. Open Palm
    if is_open_palm(lm, finger_states):
        return GestureType.OPEN_PALM, 0.95

    return GestureType.NONE, 0.3


class GestureClassifier:
    """
    Stabilizes raw gestures across consecutive frames and applies cooldown timers.
    Prevents false triggers, repeat flickering, and flutter.
    """

    def __init__(self, stability_frames: int = 6, cooldown_seconds: float = 1.2):
        self.stability_frames = max(2, stability_frames)
        self.cooldown_seconds = max(0.2, cooldown_seconds)

        self.last_raw_gesture: GestureType = GestureType.NONE
        self.consecutive_count: int = 0
        self.confirmed_gesture: GestureType = GestureType.NONE
        self.last_triggered_gesture: GestureType = GestureType.NONE
        self.last_trigger_time: float = 0.0

    def update_config(self, stability_frames: int, cooldown_seconds: float) -> None:
        self.stability_frames = max(2, stability_frames)
        self.cooldown_seconds = max(0.2, cooldown_seconds)

    def reset(self) -> None:
        self.last_raw_gesture = GestureType.NONE
        self.consecutive_count = 0
        self.confirmed_gesture = GestureType.NONE
        self.last_triggered_gesture = GestureType.NONE

    def update(self, hand_data: Optional[HandLandmarksData]) -> GestureResult:
        if hand_data is None or not hand_data.landmarks:
            self.reset()
            return GestureResult(
                raw_gesture=GestureType.NONE,
                confirmed_gesture=GestureType.NONE,
                should_trigger=False,
                confidence=0.0,
                stability_ratio=0.0,
            )

        raw_gest, raw_conf = detect_gesture(hand_data.landmarks)
        _, pinch_ratio = is_pinch(hand_data.landmarks)

        # Track consecutive frame counts
        if raw_gest == self.last_raw_gesture and raw_gest != GestureType.NONE:
            self.consecutive_count += 1
        else:
            self.last_raw_gesture = raw_gest
            self.consecutive_count = 1

        stability_ratio = min(1.0, self.consecutive_count / self.stability_frames)
        should_trigger = False
        now = time.perf_counter()

        # If stabilized
        if self.consecutive_count >= self.stability_frames:
            self.confirmed_gesture = raw_gest

            # Determine if this confirmed gesture should fire an action
            time_since_trigger = now - self.last_trigger_time

            # Continuous gestures (POINTING for air mouse) don't lock out triggers
            if self.confirmed_gesture == GestureType.POINTING:
                should_trigger = True
            elif self.confirmed_gesture != self.last_triggered_gesture:
                # Gesture transitioned to a new action
                if time_since_trigger >= (self.cooldown_seconds * 0.5):
                    should_trigger = True
                    self.last_triggered_gesture = self.confirmed_gesture
                    self.last_trigger_time = now
            elif time_since_trigger >= self.cooldown_seconds:
                # Same gesture held longer than full cooldown
                # For safety, require gesture reset or release to trigger again
                should_trigger = False
        else:
            # Not yet reached stability threshold
            if self.consecutive_count < self.stability_frames // 2:
                self.confirmed_gesture = GestureType.NONE

        return GestureResult(
            raw_gesture=raw_gest,
            confirmed_gesture=self.confirmed_gesture,
            should_trigger=should_trigger,
            confidence=raw_conf,
            stability_ratio=stability_ratio,
            pinch_distance=pinch_ratio,
        )
