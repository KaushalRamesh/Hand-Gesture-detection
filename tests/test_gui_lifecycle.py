"""
GUI Lifecycle and Component Test
Verifies that all GUI components, modes, tabs, video player, and settings dialogs
can be instantiated and exercised without errors.
"""

import sys
from pathlib import Path
import unittest

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from ui.dashboard import GestureDashboardWindow
from ui.settings_dialog import SettingsDialog
from ui.video_player import VideoPlayerWidget


class TestGUILifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Re-use existing QApplication or create one
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_settings_dialog(self):
        config = AppConfig()
        dlg = SettingsDialog(config)
        self.assertIsNotNone(dlg)
        self.assertEqual(dlg.spin_cam_idx.value(), config.camera_index)
        self.assertEqual(dlg.spin_stability.value(), config.gesture_stability_frames)
        dlg.close()

    def test_video_player_widget(self):
        player = VideoPlayerWidget()
        self.assertIsNotNone(player)
        self.assertIsNotNone(player.player)
        self.assertIsNotNone(player.video_widget)
        # Test demo clip loading
        player.load_default_demo()
        player.close()

    def test_dashboard_window_components(self):
        win = GestureDashboardWindow()
        self.assertIsNotNone(win)
        self.assertEqual(len(win.mode_group.buttons()), 3)

        # Exercise mode switching
        win.btn_mode_video.click()
        self.assertEqual(win.config.active_mode, "Video")

        win.btn_mode_ppt.click()
        self.assertEqual(win.config.active_mode, "PowerPoint")

        win.btn_mode_mouse.click()
        self.assertEqual(win.config.active_mode, "AirMouse")

        # Close and stop
        win._stop_camera()
        win.close()


if __name__ == "__main__":
    unittest.main()
