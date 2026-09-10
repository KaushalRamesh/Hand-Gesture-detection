"""
PowerPoint Presentation Controller
Dispatches touchless presentation shortcuts via PyAutoGUI.
"""

import sys
import pyautogui

# Set low latency for responsive feel without accidental aborts
pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = False

if sys.platform == "win32":
    import winsound
else:
    winsound = None


class PowerPointController:
    def __init__(self, sound_feedback: bool = True):
        self.sound_feedback = sound_feedback
        self.last_action_desc: str = "Ready"

    def _beep(self, freq: int = 1200, duration_ms: int = 60) -> None:
        if self.sound_feedback and winsound:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass

    def next_slide(self) -> str:
        """Advance to next slide (Right Arrow / Page Down)."""
        try:
            pyautogui.press("right")
            self._beep(1400, 50)
            self.last_action_desc = "Next Slide (Right Arrow)"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"

    def previous_slide(self) -> str:
        """Return to previous slide (Left Arrow / Page Up)."""
        try:
            pyautogui.press("left")
            self._beep(1000, 50)
            self.last_action_desc = "Previous Slide (Left Arrow)"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"

    def start_slideshow(self) -> str:
        """Start or resume slideshow from current slide (Shift+F5)."""
        try:
            pyautogui.hotkey("shift", "f5")
            self._beep(1600, 80)
            self.last_action_desc = "Start / Resume Slideshow (Shift+F5)"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"

    def pause_presentation(self) -> str:
        """Toggle blank/black screen or pause slideshow (B key)."""
        try:
            pyautogui.press("b")
            self._beep(800, 70)
            self.last_action_desc = "Blank / Pause Presentation (B)"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"

    def exit_slideshow(self) -> str:
        """Exit slideshow presentation (Esc)."""
        try:
            pyautogui.press("esc")
            self._beep(600, 90)
            self.last_action_desc = "Exit Slideshow (Esc)"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"

    def select_or_click(self) -> str:
        """Select item or click on current screen location."""
        try:
            pyautogui.click()
            self._beep(1500, 40)
            self.last_action_desc = "Pinch Click"
            return self.last_action_desc
        except Exception as exc:
            return f"Error: {exc}"
