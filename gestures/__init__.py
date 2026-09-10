"""
Gesture recognition and motion analysis module.
"""

from .gesture_recognition import (
    GestureClassifier,
    GestureResult,
    GestureType,
)
from .swipe_detector import SwipeDetector, SwipeDirection

__all__ = [
    "GestureClassifier",
    "GestureResult",
    "GestureType",
    "SwipeDetector",
    "SwipeDirection",
]
