"""
Point and Coordinate Smoothing Filters
Reduces jitter and hand tremors for smooth cursor movements.
"""

import math
from typing import Optional, Tuple


class ExponentialSmoother2D:
    """
    Exponential Moving Average (EMA) smoother for 2D screen or normalized coordinates.
    Includes a small deadzone to prevent resting tremors.
    """

    def __init__(self, alpha: float = 0.35, deadzone: float = 0.004):
        self.alpha = max(0.01, min(1.0, alpha))
        self.deadzone = deadzone
        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        if self.current_x is None or self.current_y is None:
            self.current_x = target_x
            self.current_y = target_y
            return target_x, target_y

        dist = math.hypot(target_x - self.current_x, target_y - self.current_y)
        if dist < self.deadzone:
            return self.current_x, self.current_y

        # Dynamic alpha: if moving fast, increase responsiveness; if moving slow, increase smoothing
        speed_factor = min(1.0, dist * 5.0)
        effective_alpha = self.alpha + (1.0 - self.alpha) * speed_factor * 0.5

        self.current_x = effective_alpha * target_x + (1.0 - effective_alpha) * self.current_x
        self.current_y = effective_alpha * target_y + (1.0 - effective_alpha) * self.current_y

        return self.current_x, self.current_y

    def reset(self) -> None:
        self.current_x = None
        self.current_y = None


class ValueSmoother:
    """Scalar smoother for continuous 1D values (e.g., confidence, distance)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.val: Optional[float] = None

    def update(self, new_val: float) -> float:
        if self.val is None:
            self.val = new_val
        else:
            self.val = self.alpha * new_val + (1.0 - self.alpha) * self.val
        return self.val

    def reset(self) -> None:
        self.val = None
