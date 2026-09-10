"""
System control actuators: PowerPoint, Video Playback, and Air Mouse.
"""

from .mouse_controller import AirMouseController
from .powerpoint_controller import PowerPointController
from .video_controller import VideoController

__all__ = ["AirMouseController", "PowerPointController", "VideoController"]
