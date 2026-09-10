# Pretrained Vision Models

This directory contains the computer vision model weights used by the application:

1. **`hand_yolov8n.pt`**:
   - Lightweight YOLOv8 Nano model fine-tuned for hand detection.
   - Origin: Hugging Face repository `Bingsu/adetailer` (derived from Ultralytics YOLOv8).
   - Task: Object detection, identifying bounding box coordinates `(x1, y1, x2, y2)` and confidence for hands in real-time.

2. **`hand_landmarker.task`**:
   - Official Google MediaPipe Hand Landmarker model bundle (float16).
   - Origin: Google MediaPipe Models repository.
   - Task: Extracts 21 3D hand landmarks and left/right handedness classification from the cropped hand region.

The application automatically verifies and downloads these models upon first startup via `config.py:ensure_models_exist()`.
