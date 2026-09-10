"""
YOLOv8 Hand Object Detector
Detects hand regions and confidence scores using Ultralytics YOLO.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class HandBox:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel space
    confidence: float
    class_id: int
    class_name: str = "HAND"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2


class YOLOHandDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_inference_time_ms: float = 0.0

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found at: {self.model_path}")

        print(f"[YOLOHandDetector] Loading model from {self.model_path} onto {self.device}...")
        self.model = YOLO(str(self.model_path))
        print(f"[YOLOHandDetector] Model loaded successfully. Classes: {self.model.names}")

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = max(0.05, min(0.99, threshold))

    def detect(self, frame: np.ndarray) -> List[HandBox]:
        """
        Run YOLO hand detection on a BGR or RGB OpenCV image.
        Returns a list of detected HandBox objects sorted by confidence (descending).
        """
        if frame is None or frame.size == 0:
            return []

        t0 = time.perf_counter()
        try:
            # Run inference with ultralytics
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            print(f"[YOLOHandDetector] Inference error: {exc}")
            return []

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[HandBox] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # Clamp bounding box coordinates to frame boundaries
                x1 = max(0, min(w - 1, int(coords[0])))
                y1 = max(0, min(h - 1, int(coords[1])))
                x2 = max(0, min(w - 1, int(coords[2])))
                y2 = max(0, min(h - 1, int(coords[3])))

                if x2 > x1 and y2 > y1:
                    detections.append(
                        HandBox(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name="HAND",
                        )
                    )

        # Sort descending by confidence
        detections.sort(key=lambda b: b.confidence, reverse=True)
        return detections
