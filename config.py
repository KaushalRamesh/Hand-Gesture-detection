"""
Application Configuration and Settings Management
Handles paths, model downloading, runtime parameters, and persistent user settings.
"""

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
SETTINGS_PATH = BASE_DIR / "settings.json"

YOLO_MODEL_PATH = MODELS_DIR / "hand_yolov8n.pt"
MEDIAPIPE_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

YOLO_MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_models_exist(progress_callback=None) -> tuple[bool, str]:
    """
    Verifies that YOLO and MediaPipe models are downloaded.
    Downloads them if missing.
    Returns (success, message).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    def _download(url: str, dest_path: Path, model_name: str):
        if dest_path.exists() and dest_path.stat().st_size > 100_000:
            return
        if progress_callback:
            progress_callback(f"Downloading {model_name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 HandGestureControl/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 65536
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if progress_callback and total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    progress_callback(f"Downloading {model_name}: {pct}%")

    try:
        _download(YOLO_MODEL_URL, YOLO_MODEL_PATH, "YOLOv8 Hand Detector")
        _download(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH, "MediaPipe Hand Landmarker")
        return True, "Models verified."
    except Exception as exc:
        return False, f"Model download error: {exc}"


@dataclass
class AppConfig:
    # Camera settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    mirror_camera: bool = True

    # YOLO object detection
    yolo_confidence_threshold: float = 0.50
    yolo_iou_threshold: float = 0.45

    # Landmark and Hand preference
    hand_selection: str = "Auto"  # "Auto", "Right", "Left"
    max_num_hands: int = 2

    # Gesture confirmation and cooldown
    gesture_stability_frames: int = 6
    gesture_cooldown_seconds: float = 1.2

    # Swipe parameters
    swipe_displacement_threshold: float = 0.16  # Normalized fraction of frame width
    swipe_cooldown_seconds: float = 1.0
    swipe_history_len: int = 12

    # Air Mouse parameters
    air_mouse_sensitivity: float = 1.6
    air_mouse_smoothing: float = 0.35
    air_mouse_enabled: bool = True

    # Feedback and General
    sound_feedback: bool = True
    active_mode: str = "PowerPoint"  # "PowerPoint", "Video", "AirMouse"

    def save(self, filepath: Optional[Path] = None) -> None:
        """Persist settings to json file."""
        target = filepath or SETTINGS_PATH
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as exc:
            print(f"[Config] Warning: Failed to save config to {target}: {exc}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "AppConfig":
        """Load settings from json file with fallback to defaults."""
        target = filepath or SETTINGS_PATH
        if not target.exists():
            config = cls()
            config.save(target)
            return config

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_keys = set(cls.__dataclass_fields__.keys())
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception as exc:
            print(f"[Config] Warning: Failed to load config from {target}, using defaults: {exc}")
            return cls()
