"""
Video Playback Controller
Controls desktop media players (VLC, browser, etc.) and hooks into the built-in video player.
"""

import sys
from typing import Callable, Optional
import pyautogui

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = False

if sys.platform == "win32":
    import winsound
else:
    winsound = None


class VideoController:
    def __init__(self, sound_feedback: bool = True):
        self.sound_feedback = sound_feedback
        self.last_action_desc: str = "Ready"
        # Optional callback to directly control the integrated PySide6 player
        self.internal_player_callback: Optional[Callable[[str, any], None]] = None

    def set_internal_player_callback(self, cb: Optional[Callable[[str, any], None]]) -> None:
        self.internal_player_callback = cb

    def _beep(self, freq: int = 1200, duration_ms: int = 60) -> None:
        if self.sound_feedback and winsound:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass

    def play(self) -> str:
        """Trigger Play / Resume."""
        if self.internal_player_callback:
            self.internal_player_callback("play", None)
        else:
            try:
                pyautogui.press("space")
            except Exception:
                pass
        self._beep(1500, 60)
        self.last_action_desc = "Play / Resume (Space)"
        return self.last_action_desc

    def pause(self) -> str:
        """Trigger Pause."""
        if self.internal_player_callback:
            self.internal_player_callback("pause", None)
        else:
            try:
                pyautogui.press("space")
            except Exception:
                pass
        self._beep(900, 60)
        self.last_action_desc = "Pause (Space)"
        return self.last_action_desc

    def seek_forward(self, seconds: int = 5) -> str:
        """Seek forward (+5s or Next video)."""
        if self.internal_player_callback:
            self.internal_player_callback("seek_relative", seconds * 1000)
        else:
            try:
                pyautogui.press("right")
            except Exception:
                pass
        self._beep(1400, 50)
        self.last_action_desc = f"Seek Forward +{seconds}s (Right Arrow)"
        return self.last_action_desc

    def seek_backward(self, seconds: int = 5) -> str:
        """Seek backward (-5s or Prev video)."""
        if self.internal_player_callback:
            self.internal_player_callback("seek_relative", -seconds * 1000)
        else:
            try:
                pyautogui.press("left")
            except Exception:
                pass
        self._beep(1100, 50)
        self.last_action_desc = f"Seek Backward -{seconds}s (Left Arrow)"
        return self.last_action_desc

    def click(self) -> str:
        """Mouse click for playhead scrubbing or button selection."""
        try:
            pyautogui.click()
            self._beep(1600, 40)
            self.last_action_desc = "Pinch Click"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"
