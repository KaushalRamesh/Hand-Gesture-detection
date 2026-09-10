"""
Air Mouse Cursor Controller
Translates fingertip landmarks to smoothed OS cursor movements, drag-and-drop, and clicks.
"""

import math
import sys
import time
from typing import Optional, Tuple

import pyautogui
from utils.smoothing import ExponentialSmoother2D

pyautogui.PAUSE = 0.005
pyautogui.FAILSAFE = False

if sys.platform == "win32":
    import winsound
else:
    winsound = None


class AirMouseController:
    def __init__(
        self,
        sensitivity: float = 1.6,
        smoothing_alpha: float = 0.35,
        margin_x: float = 0.15,
        margin_y: float = 0.15,
        sound_feedback: bool = True,
    ):
        self.sensitivity = sensitivity
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.sound_feedback = sound_feedback
        self.smoother = ExponentialSmoother2D(alpha=smoothing_alpha)

        # Screen dimensions
        try:
            self.screen_w, self.screen_h = pyautogui.size()
        except Exception:
            self.screen_w, self.screen_h = 1920, 1080

        # Click and drag state
        self.is_dragging: bool = False
        self.last_right_click_time: float = 0.0
        self.last_action_desc: str = "Mouse Ready"

    def _beep(self, freq: int = 1500, duration_ms: int = 40) -> None:
        if self.sound_feedback and winsound:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass

    def update_config(self, sensitivity: float, smoothing_alpha: float) -> None:
        self.sensitivity = max(0.5, min(3.5, sensitivity))
        self.smoother.alpha = max(0.05, min(0.95, smoothing_alpha))

    def reset(self) -> None:
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self.is_dragging = False
        self.smoother.reset()

    def map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates [0..1] to desktop screen pixel resolution
        using an inner bounding region to allow effortless reach of screen corners.
        """
        # Crop bounds with margin
        x_clamped = (norm_x - self.margin_x) / max(0.01, (1.0 - 2.0 * self.margin_x))
        y_clamped = (norm_y - self.margin_y) / max(0.01, (1.0 - 2.0 * self.margin_y))

        # Clamp to 0..1
        x_clamped = max(0.0, min(1.0, x_clamped))
        y_clamped = max(0.0, min(1.0, y_clamped))

        # Target screen pixel coordinates
        target_x = x_clamped * (self.screen_w - 1)
        target_y = y_clamped * (self.screen_h - 1)

        # Smooth coordinates
        smooth_x, smooth_y = self.smoother.update(target_x, target_y)
        return int(smooth_x), int(smooth_y)

    def move_cursor(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Smoothly move the mouse cursor to the given normalized camera position."""
        sx, sy = self.map_to_screen(norm_x, norm_y)
        try:
            pyautogui.moveTo(sx, sy)
            self.last_action_desc = f"Cursor: ({sx}, {sy})"
        except Exception as exc:
            self.last_action_desc = f"Cursor error: {exc}"
        return sx, sy

    def handle_pinch(self, norm_x: float, norm_y: float, is_pinched: bool) -> str:
        """Handles click and drag gestures."""
        sx, sy = self.map_to_screen(norm_x, norm_y)

        if is_pinched:
            if not self.is_dragging:
                # Pinch start -> Mouse Down / Click
                try:
                    pyautogui.mouseDown(button="left")
                    self.is_dragging = True
                    self._beep(1600, 35)
                    self.last_action_desc = "Mouse Left Click / Drag Start"
                except Exception as exc:
                    self.last_action_desc = f"Click error: {exc}"
            else:
                # Continue Drag
                try:
                    pyautogui.moveTo(sx, sy)
                    self.last_action_desc = f"Dragging: ({sx}, {sy})"
                except Exception:
                    pass
        else:
            if self.is_dragging:
                # Pinch released -> Mouse Up
                try:
                    pyautogui.mouseUp(button="left")
                    self.is_dragging = False
                    self._beep(1200, 35)
                    self.last_action_desc = "Mouse Drag End / Released"
                except Exception as exc:
                    self.last_action_desc = f"Release error: {exc}"

        return self.last_action_desc

    def right_click(self) -> str:
        """Triggers right-click with debounce."""
        now = time.perf_counter()
        if now - self.last_right_click_time > 0.6:
            try:
                pyautogui.rightClick()
                self._beep(1800, 50)
                self.last_right_click_time = now
                self.last_action_desc = "Right Click"
            except Exception as exc:
                self.last_action_desc = f"Right click error: {exc}"
        return self.last_action_desc
