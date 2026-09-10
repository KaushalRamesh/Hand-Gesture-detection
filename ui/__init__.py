"""
User Interface module using PySide6.
"""

from .dashboard import GestureDashboardWindow
from .settings_dialog import SettingsDialog
from .video_player import VideoPlayerWidget

__all__ = ["GestureDashboardWindow", "SettingsDialog", "VideoPlayerWidget"]
