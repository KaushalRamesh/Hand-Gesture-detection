"""
High-Precision FPS Tracker
Calculates real-time moving average FPS and inference frame times.
"""

import time
from collections import deque


class FPSTracker:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        """Call on every frame render/capture cycle. Returns smoothed FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)

        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self._fps = (len(self.timestamps) - 1) / elapsed
            else:
                self._fps = 0.0
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_time_ms(self) -> float:
        if self._fps > 0:
            return 1000.0 / self._fps
        return 0.0
