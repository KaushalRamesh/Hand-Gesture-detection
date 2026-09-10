"""
Multi-Frame Swipe Gesture Detector
Tracks horizontal trajectory of the hand center and wrist across recent frames.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SwipeDirection(str, Enum):
    NONE = "NONE"
    SWIPE_RIGHT = "SWIPE RIGHT"
    SWIPE_LEFT = "SWIPE LEFT"


@dataclass
class SwipeResult:
    direction: SwipeDirection
    displacement_x: float
    velocity: float
    triggered: bool
    # Visual feedback tracking
    visual_active: bool = False
    visual_direction: SwipeDirection = SwipeDirection.NONE
    visual_progress: float = 0.0  # 1.0 -> 0.0 decay for HUD animation


class SwipeDetector:
    def __init__(
        self,
        displacement_threshold: float = 0.16,
        cooldown_seconds: float = 1.0,
        history_len: int = 14,
        min_speed: float = 0.35,  # Normalized units per second
    ):
        self.displacement_threshold = displacement_threshold
        self.cooldown_seconds = cooldown_seconds
        self.history = deque(maxlen=history_len)
        self.min_speed = min_speed

        self.last_swipe_time: float = 0.0
        self.last_triggered_direction = SwipeDirection.NONE

        # Visual indicator persistence
        self.visual_banner_time: float = 0.0
        self.visual_banner_dir = SwipeDirection.NONE
        self.visual_duration: float = 1.2

    def update_config(self, threshold: float, cooldown: float) -> None:
        self.displacement_threshold = max(0.05, min(0.50, threshold))
        self.cooldown_seconds = max(0.3, cooldown)

    def reset(self) -> None:
        self.history.clear()

    def update(self, center_x: Optional[float], center_y: Optional[float]) -> SwipeResult:
        """
        Takes normalized coordinates [0..1] of hand center or wrist.
        Returns SwipeResult indicating whether a swipe was detected.
        """
        now = time.perf_counter()

        # Update visual banner animation state
        time_since_banner = now - self.visual_banner_time
        visual_active = time_since_banner < self.visual_duration
        visual_progress = max(0.0, 1.0 - (time_since_banner / self.visual_duration)) if visual_active else 0.0

        if center_x is None or center_y is None:
            self.reset()
            return SwipeResult(
                direction=SwipeDirection.NONE,
                displacement_x=0.0,
                velocity=0.0,
                triggered=False,
                visual_active=visual_active,
                visual_direction=self.visual_banner_dir if visual_active else SwipeDirection.NONE,
                visual_progress=visual_progress,
            )

        self.history.append((now, center_x, center_y))

        # Check cooldown
        if (now - self.last_swipe_time) < self.cooldown_seconds:
            return SwipeResult(
                direction=SwipeDirection.NONE,
                displacement_x=0.0,
                velocity=0.0,
                triggered=False,
                visual_active=visual_active,
                visual_direction=self.visual_banner_dir if visual_active else SwipeDirection.NONE,
                visual_progress=visual_progress,
            )

        # Look back in history for movement across ~0.15s to 0.45s
        triggered = False
        swipe_dir = SwipeDirection.NONE
        max_dx = 0.0
        calculated_velocity = 0.0

        for past_time, past_x, past_y in list(self.history)[:-2]:
            dt = now - past_time
            if 0.12 <= dt <= 0.50:
                dx = center_x - past_x
                dy = center_y - past_y
                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # Horizontal movement must exceed threshold and dominate vertical jitter
                if abs_dx >= self.displacement_threshold and abs_dx > (abs_dy * 1.35):
                    speed = abs_dx / dt
                    if speed >= self.min_speed:
                        triggered = True
                        swipe_dir = SwipeDirection.SWIPE_RIGHT if dx > 0 else SwipeDirection.SWIPE_LEFT
                        max_dx = dx
                        calculated_velocity = speed
                        break

        if triggered:
            self.last_swipe_time = now
            self.last_triggered_direction = swipe_dir
            self.visual_banner_time = now
            self.visual_banner_dir = swipe_dir
            self.history.clear()  # Clear history to avoid re-triggering from trailing frames
            return SwipeResult(
                direction=swipe_dir,
                displacement_x=max_dx,
                velocity=calculated_velocity,
                triggered=True,
                visual_active=True,
                visual_direction=swipe_dir,
                visual_progress=1.0,
            )

        return SwipeResult(
            direction=SwipeDirection.NONE,
            displacement_x=0.0,
            velocity=0.0,
            triggered=False,
            visual_active=visual_active,
            visual_direction=self.visual_banner_dir if visual_active else SwipeDirection.NONE,
            visual_progress=visual_progress,
        )
