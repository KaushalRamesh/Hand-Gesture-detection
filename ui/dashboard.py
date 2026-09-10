"""
AI Hand Gesture Control Dashboard
Modern cyber-dark workstation interface using PySide6.
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QPoint, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import (
    ASSETS_DIR,
    MEDIAPIPE_MODEL_PATH,
    YOLO_MODEL_PATH,
    AppConfig,
    ensure_models_exist,
)
from controls.mouse_controller import AirMouseController
from controls.powerpoint_controller import PowerPointController
from controls.video_controller import VideoController
from detection.yolo_detector import HandBox, YOLOHandDetector
from gestures.gesture_recognition import (
    GestureClassifier,
    GestureResult,
    GestureType,
)
from gestures.swipe_detector import SwipeDetector, SwipeDirection, SwipeResult
from landmarks.hand_landmarks import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    WRIST,
    HandLandmarkerWrapper,
    HandLandmarksData,
)
from ui.settings_dialog import SettingsDialog
from ui.video_player import VideoPlayerWidget
from utils.fps import FPSTracker


class VideoProcessingWorker(QThread):
    """
    Background worker thread for webcam capture, YOLO detection,
    MediaPipe landmark analysis, gesture recognition, and HUD drawing.
    """
    frame_processed = Signal(QPixmap, dict, str)
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.running = False
        self.mutex = QMutex()

        # Controllers
        self.ppt_controller = PowerPointController(sound_feedback=config.sound_feedback)
        self.video_controller = VideoController(sound_feedback=config.sound_feedback)
        self.mouse_controller = AirMouseController(
            sensitivity=config.air_mouse_sensitivity,
            smoothing_alpha=config.air_mouse_smoothing,
            sound_feedback=config.sound_feedback,
        )

        # Vision engines
        self.yolo_detector: Optional[YOLOHandDetector] = None
        self.mp_landmarker: Optional[HandLandmarkerWrapper] = None
        self.gesture_classifier = GestureClassifier(
            stability_frames=config.gesture_stability_frames,
            cooldown_seconds=config.gesture_cooldown_seconds,
        )
        self.swipe_detector = SwipeDetector(
            displacement_threshold=config.swipe_displacement_threshold,
            cooldown_seconds=config.swipe_cooldown_seconds,
            history_len=config.swipe_history_len,
        )
        self.fps_tracker = FPSTracker()

        self.last_action_text = "System Ready"

    def update_config(self, new_config: AppConfig):
        self.mutex.lock()
        try:
            self.config = new_config
            self.ppt_controller.sound_feedback = new_config.sound_feedback
            self.video_controller.sound_feedback = new_config.sound_feedback
            self.mouse_controller.sound_feedback = new_config.sound_feedback
            self.mouse_controller.update_config(new_config.air_mouse_sensitivity, new_config.air_mouse_smoothing)
            if self.yolo_detector:
                self.yolo_detector.set_confidence_threshold(new_config.yolo_confidence_threshold)
            self.gesture_classifier.update_config(new_config.gesture_stability_frames, new_config.gesture_cooldown_seconds)
            self.swipe_detector.update_config(new_config.swipe_displacement_threshold, new_config.swipe_cooldown_seconds)
        finally:
            self.mutex.unlock()

    def stop(self):
        self.running = False
        self.wait(1500)

    def run(self):
        self.status_message.emit("Initializing Vision AI Pipeline...")

        # Initialize models
        try:
            self.yolo_detector = YOLOHandDetector(
                YOLO_MODEL_PATH,
                confidence_threshold=self.config.yolo_confidence_threshold,
            )
            self.mp_landmarker = HandLandmarkerWrapper(MEDIAPIPE_MODEL_PATH)
        except Exception as exc:
            self.error_occurred.emit(f"Initialization Failed: {exc}")
            return

        cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            # Try default backend fallback
            cap = cv2.VideoCapture(self.config.camera_index)

        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open camera device {self.config.camera_index}. Check connection or settings.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        self.running = True
        self.status_message.emit("Camera feed active.")

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # Mirror camera for intuitive user interaction
            if self.config.mirror_camera:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            fps = self.fps_tracker.tick()

            # 1. YOLO Hand Object Detection
            hand_boxes = self.yolo_detector.detect(frame)
            yolo_ms = self.yolo_detector.last_inference_time_ms

            # Select primary hand
            primary_box: Optional[HandBox] = None
            if hand_boxes:
                primary_box = hand_boxes[0]

            # 2. MediaPipe Hand Landmark Detection
            hand_data: Optional[HandLandmarksData] = None
            mp_ms = 0.0
            if primary_box is not None:
                hand_data = self.mp_landmarker.extract_from_crop(frame, primary_box.bbox)
                mp_ms = self.mp_landmarker.last_inference_time_ms

            # 3. Gesture Recognition & Swipe Detection
            gest_result = self.gesture_classifier.update(hand_data)

            # Swipe tracking point: wrist or hand center
            center_x, center_y = None, None
            if hand_data is not None:
                center_x, center_y = hand_data.wrist.x, hand_data.wrist.y
            elif primary_box is not None:
                cx, cy = primary_box.center
                center_x, center_y = cx / w, cy / h

            swipe_result = self.swipe_detector.update(center_x, center_y)

            # 4. Action Dispatch
            action_desc = self._dispatch_actions(gest_result, swipe_result, hand_data)
            if action_desc:
                self.last_action_text = action_desc

            # 5. Draw Visual Overlays & HUD on Frame
            self._draw_hud_overlays(frame, primary_box, hand_boxes, hand_data, gest_result, swipe_result, fps, yolo_ms, mp_ms)

            # Convert to QPixmap for PySide6 GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Package telemetry stats
            telemetry = {
                "fps": fps,
                "yolo_detected": primary_box is not None,
                "yolo_conf": primary_box.confidence if primary_box else 0.0,
                "yolo_ms": yolo_ms,
                "mp_detected": hand_data is not None,
                "mp_ms": mp_ms,
                "handedness": hand_data.handedness if hand_data else "None",
                "handedness_score": hand_data.handedness_confidence if hand_data else 0.0,
                "gesture": gest_result.confirmed_gesture.value,
                "raw_gesture": gest_result.raw_gesture.value,
                "stability": gest_result.stability_ratio,
                "action": self.last_action_text,
                "mode": self.config.active_mode,
            }

            self.frame_processed.emit(pixmap, telemetry, self.last_action_text)

        cap.release()
        self.mouse_controller.reset()
        self.status_message.emit("Camera stopped.")

    def _dispatch_actions(
        self,
        gest: GestureResult,
        swipe: SwipeResult,
        hand: Optional[HandLandmarksData],
    ) -> Optional[str]:
        mode = self.config.active_mode

        # Priority A: Swipe Actions (PowerPoint next/prev, Video seek/next)
        if swipe.triggered:
            if mode == "PowerPoint":
                if swipe.direction == SwipeDirection.SWIPE_RIGHT:
                    return self.ppt_controller.next_slide()
                elif swipe.direction == SwipeDirection.SWIPE_LEFT:
                    return self.ppt_controller.previous_slide()
            elif mode == "Video":
                if swipe.direction == SwipeDirection.SWIPE_RIGHT:
                    return self.video_controller.seek_forward(5)
                elif swipe.direction == SwipeDirection.SWIPE_LEFT:
                    return self.video_controller.seek_backward(5)

        # Priority B: Air Mouse Continuous Tracking
        if mode == "AirMouse" and hand is not None:
            index_pt = hand.index_tip
            thumb_pt = hand.thumb_tip

            if gest.confirmed_gesture == GestureType.POINTING or gest.raw_gesture == GestureType.POINTING:
                # Continuous cursor movement
                self.mouse_controller.move_cursor(index_pt.x, index_pt.y)
                return "Air Mouse: Moving Cursor"
            elif gest.confirmed_gesture == GestureType.PINCH or gest.raw_gesture == GestureType.PINCH:
                # Pinch click and drag
                return self.mouse_controller.handle_pinch(index_pt.x, index_pt.y, is_pinched=True)
            elif gest.confirmed_gesture == GestureType.TWO_FINGERS and gest.should_trigger:
                return self.mouse_controller.right_click()
            elif gest.confirmed_gesture == GestureType.OPEN_PALM:
                self.mouse_controller.handle_pinch(index_pt.x, index_pt.y, is_pinched=False)
                return "Air Mouse: Release / Idle"
            else:
                # Release pinch if held
                self.mouse_controller.handle_pinch(index_pt.x, index_pt.y, is_pinched=False)

        # Priority C: Discrete Gestures in PowerPoint and Video Modes
        if gest.should_trigger:
            cg = gest.confirmed_gesture
            if mode == "PowerPoint":
                if cg == GestureType.OPEN_PALM:
                    return self.ppt_controller.start_slideshow()
                elif cg == GestureType.FIST:
                    return self.ppt_controller.pause_presentation()
                elif cg == GestureType.PINCH:
                    return self.ppt_controller.select_or_click()
                elif cg == GestureType.POINTING:
                    # Switch to cursor control
                    return "Air Cursor Mode Activated"

            elif mode == "Video":
                if cg in (GestureType.OPEN_PALM, GestureType.THUMB_UP):
                    return self.video_controller.play()
                elif cg in (GestureType.FIST, GestureType.THUMB_DOWN):
                    return self.video_controller.pause()
                elif cg == GestureType.PINCH:
                    return self.video_controller.click()

        return None

    def _draw_hud_overlays(
        self,
        frame: np.ndarray,
        primary_box: Optional[HandBox],
        all_boxes: List[HandBox],
        hand_data: Optional[HandLandmarksData],
        gest: GestureResult,
        swipe: SwipeResult,
        fps: float,
        yolo_ms: float,
        mp_ms: float,
    ):
        """Draws bounding boxes, skeleton landmarks, HUD telemetry badges, and animations."""
        h, w = frame.shape[:2]

        # 1. Draw YOLO Bounding Boxes
        for box in all_boxes:
            is_primary = (box == primary_box)
            color = (254, 242, 0) if is_primary else (150, 150, 150)  # Cyan for primary (BGR: 254, 242, 0)
            x1, y1, x2, y2 = box.bbox

            # Corner brackets for futuristic look
            corner_len = min(24, (x2 - x1) // 4, (y2 - y1) // 4)
            thickness = 2 if is_primary else 1

            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

            # Thin bounding box border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # YOLO Badge Tag
            label = f"HAND {box.confidence:.1%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 12, y1), (18, 22, 31), -1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 12, y1), color, 1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw MediaPipe 21 Hand Landmarks & Skeleton
        if hand_data is not None:
            pts = [(lm.px, lm.py) for lm in hand_data.landmarks]

            # Draw bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                pt1 = pts[p1_idx]
                pt2 = pts[p2_idx]
                cv2.line(frame, pt1, pt2, (120, 220, 255), 2, cv2.LINE_AA)

            # Draw nodes
            for i, (px, py) in enumerate(pts):
                # Highlight index and thumb tip with vibrant nodes
                if i in (INDEX_TIP, 4):  # 4 is thumb tip
                    cv2.circle(frame, (px, py), 7, (0, 242, 254), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 4, (255, 150, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top HUD: Mode & FPS Badges
        # Top Left: Mode Badge
        mode_str = f"MODE: {self.config.active_mode.upper()}"
        cv2.rectangle(frame, (14, 14), (200, 48), (15, 20, 30), -1)
        cv2.rectangle(frame, (14, 14), (200, 48), (0, 242, 254), 1)
        cv2.putText(frame, mode_str, (24, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2, cv2.LINE_AA)

        # Top Right: FPS Badge
        fps_str = f"FPS: {fps:.0f}"
        (fw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (w - fw - 34, 14), (w - 14, 48), (15, 20, 30), -1)
        cv2.rectangle(frame, (w - fw - 34, 14), (w - 14, 48), (0, 230, 118), 1)
        cv2.putText(frame, fps_str, (w - fw - 24, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 2, cv2.LINE_AA)

        # 4. Center Swipe Banner
        if swipe.visual_active and swipe.visual_direction != SwipeDirection.NONE:
            banner_w = 260
            banner_h = 56
            bx = (w - banner_w) // 2
            by = h // 2 - 40
            alpha = swipe.visual_progress

            # Pulsing color
            c_val = int(255 * alpha)
            cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (20, 26, 38), -1)
            cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, c_val, 255), 2)

            swipe_txt = ">>> SWIPE RIGHT >>>" if swipe.visual_direction == SwipeDirection.SWIPE_RIGHT else "<<< SWIPE LEFT <<<"
            cv2.putText(
                frame,
                swipe_txt,
                (bx + 18, by + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, c_val, 255),
                2,
                cv2.LINE_AA,
            )

        # 5. Bottom HUD: Confirmed Gesture & Action Pill
        gest_name = gest.confirmed_gesture.value if gest.confirmed_gesture != GestureType.NONE else "SEARCHING HAND..."
        action_text = self.last_action_text

        hud_w = min(480, w - 28)
        hud_h = 54
        hx = (w - hud_w) // 2
        hy = h - hud_h - 14

        cv2.rectangle(frame, (hx, hy), (hx + hud_w, hy + hud_h), (12, 16, 24), -1)
        cv2.rectangle(frame, (hx, hy), (hx + hud_w, hy + hud_h), (36, 44, 61), 1)

        # Gesture Title
        cv2.putText(frame, f"GESTURE: {gest_name}", (hx + 16, hy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Action Title
        cv2.putText(frame, f"ACTION: {action_text}", (hx + 16, hy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 254), 1, cv2.LINE_AA)

        # Stability progress mini-bar
        if gest.raw_gesture != GestureType.NONE and gest.stability_ratio > 0.1:
            p_len = int(100 * gest.stability_ratio)
            cv2.rectangle(frame, (hx + hud_w - 116, hy + 22), (hx + hud_w - 16, hy + 32), (30, 38, 50), -1)
            cv2.rectangle(frame, (hx + hud_w - 116, hy + 22), (hx + hud_w - 116 + p_len, hy + 32), (0, 230, 118), -1)


class GestureDashboardWindow(QMainWindow):
    """
    Main application dashboard window featuring camera view, telemetry panels,
    mode switchers, gesture quick-reference, and integrated video player.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Hand Gesture Control System | PowerPoint & Video")
        self.resize(1200, 760)
        self.setMinimumSize(960, 600)

        self.config = AppConfig.load()
        self.worker: Optional[VideoProcessingWorker] = None

        self._setup_stylesheet()
        self._setup_ui()
        self._connect_signals()

        # Initialize camera once Qt event loop begins
        QTimer.singleShot(250, self._check_and_start_camera)

    def _setup_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0c0f17;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QGroupBox {
                background-color: #121722;
                border: 1px solid #1f2737;
                border-radius: 8px;
                margin-top: 14px;
                padding: 12px;
                font-weight: 600;
                color: #00f2fe;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #1a2233;
                color: #f1f5f9;
                border: 1px solid #2d3748;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #243049;
                border-color: #00f2fe;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0f1523;
            }
            QPushButton:checked {
                background-color: #008080;
                border-color: #00f2fe;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #1f2737;
                border-radius: 8px;
                background-color: #121722;
            }
            QTabBar::tab {
                background-color: #161c2b;
                color: #94a3b8;
                border: 1px solid #1f2737;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #121722;
                color: #00f2fe;
                border-color: #00f2fe;
            }
            QLabel {
                font-size: 12px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # 1. Header Bar: Title, Mode Selector, Settings Button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)

        title_label = QLabel("✨ AI GESTURE CONTROL")
        title_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: 1px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Mode Selection Buttons
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.btn_mode_ppt = QPushButton("📊 PowerPoint Mode")
        self.btn_mode_ppt.setCheckable(True)

        self.btn_mode_video = QPushButton("🎬 Video Mode")
        self.btn_mode_video.setCheckable(True)

        self.btn_mode_mouse = QPushButton("🖱 Air Mouse Mode")
        self.btn_mode_mouse.setCheckable(True)

        # Set default active mode from config
        if self.config.active_mode == "Video":
            self.btn_mode_video.setChecked(True)
        elif self.config.active_mode == "AirMouse":
            self.btn_mode_mouse.setChecked(True)
        else:
            self.btn_mode_ppt.setChecked(True)

        self.mode_group.addButton(self.btn_mode_ppt)
        self.mode_group.addButton(self.btn_mode_video)
        self.mode_group.addButton(self.btn_mode_mouse)

        header_layout.addWidget(self.btn_mode_ppt)
        header_layout.addWidget(self.btn_mode_video)
        header_layout.addWidget(self.btn_mode_mouse)

        self.btn_settings = QPushButton("⚙ Settings")
        header_layout.addWidget(self.btn_settings)

        main_layout.addLayout(header_layout)

        # 2. Main Body Splitter: Camera Feed (Left) & Telemetry / Player Tabs (Right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: Camera View
        cam_container = QFrame()
        cam_container.setStyleSheet("""
            QFrame {
                background-color: #080b11;
                border: 2px solid #1a2333;
                border-radius: 12px;
            }
        """)
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(8, 8, 8, 8)

        self.camera_label = QLabel("Initializing Camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("color: #64748b; font-size: 14px; font-weight: 500;")
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam_layout.addWidget(self.camera_label)

        # Camera Control Bar under video
        cam_ctrl_layout = QHBoxLayout()
        self.btn_start_cam = QPushButton("▶ Start Camera")
        self.btn_stop_cam = QPushButton("⏹ Stop Camera")
        self.lbl_cam_status = QLabel("Camera: Standby")
        self.lbl_cam_status.setStyleSheet("color: #94a3b8; font-size: 11px;")

        cam_ctrl_layout.addWidget(self.btn_start_cam)
        cam_ctrl_layout.addWidget(self.btn_stop_cam)
        cam_ctrl_layout.addStretch()
        cam_ctrl_layout.addWidget(self.lbl_cam_status)
        cam_layout.addLayout(cam_ctrl_layout)

        splitter.addWidget(cam_container)

        # Right: Tabs (Telemetry & Reference, Integrated Video Player)
        self.tabs = QTabWidget()

        # Tab 1: Telemetry & Gesture Reference
        tab_telemetry = QWidget()
        tab_tel_layout = QVBoxLayout(tab_telemetry)
        tab_tel_layout.setContentsMargins(10, 10, 10, 10)
        tab_tel_layout.setSpacing(10)

        # Group A: Detection Pipeline Telemetry
        grp_pipeline = QGroupBox("Vision Pipeline Telemetry")
        grid_pipe = QGridLayout(grp_pipeline)
        grid_pipe.setSpacing(8)

        grid_pipe.addWidget(QLabel("YOLO Detection:"), 0, 0)
        self.val_yolo_status = QLabel("Searching...")
        self.val_yolo_status.setStyleSheet("color: #00f2fe; font-weight: bold;")
        grid_pipe.addWidget(self.val_yolo_status, 0, 1)

        grid_pipe.addWidget(QLabel("YOLO Confidence:"), 1, 0)
        self.val_yolo_conf = QLabel("0.0%")
        grid_pipe.addWidget(self.val_yolo_conf, 1, 1)

        grid_pipe.addWidget(QLabel("MediaPipe Hands:"), 2, 0)
        self.val_mp_status = QLabel("Standby")
        grid_pipe.addWidget(self.val_mp_status, 2, 1)

        grid_pipe.addWidget(QLabel("Active Hand:"), 3, 0)
        self.val_hand = QLabel("Auto")
        grid_pipe.addWidget(self.val_hand, 3, 1)

        grid_pipe.addWidget(QLabel("Pipeline FPS:"), 4, 0)
        self.val_fps = QLabel("0.0")
        self.val_fps.setStyleSheet("color: #00e676; font-weight: bold;")
        grid_pipe.addWidget(self.val_fps, 4, 1)

        tab_tel_layout.addWidget(grp_pipeline)

        # Group B: Active Gesture & Triggered Action
        grp_action = QGroupBox("Gesture & System Action")
        grid_act = QGridLayout(grp_action)
        grid_act.setSpacing(8)

        grid_act.addWidget(QLabel("Current Gesture:"), 0, 0)
        self.val_gesture = QLabel("NONE")
        self.val_gesture.setStyleSheet("color: #ffaa00; font-size: 13px; font-weight: bold;")
        grid_act.addWidget(self.val_gesture, 0, 1)

        grid_act.addWidget(QLabel("Last Action:"), 1, 0)
        self.val_action = QLabel("Ready")
        self.val_action.setStyleSheet("color: #00f2fe; font-size: 13px; font-weight: bold;")
        grid_act.addWidget(self.val_action, 1, 1)

        tab_tel_layout.addWidget(grp_action)

        # Group C: Gesture Quick Reference Cards (Scrollable)
        grp_ref = QGroupBox("Gesture Reference Guide")
        ref_layout = QVBoxLayout(grp_ref)
        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_widget = QWidget()
        ref_widget_layout = QVBoxLayout(ref_widget)
        ref_widget_layout.setSpacing(6)

        reference_items = [
            ("✋ Open Palm", "Play / Resume Video | Start Slideshow"),
            ("✊ Fist", "Pause Video | Pause / Blank Slideshow"),
            ("👉 Swipe Right", "Next Slide | Seek +5s / Next Video"),
            ("👈 Swipe Left", "Prev Slide | Seek -5s / Prev Video"),
            ("🤏 Pinch", "Mouse Left Click / Drag / Select"),
            ("☝ Point Up", "Air Mouse Cursor Control Mode"),
            ("✌ Two Fingers", "Mouse Right Click"),
            ("👍 Thumb Up", "Play / Resume Video"),
            ("👎 Thumb Down", "Pause Video"),
        ]

        for icon_gest, desc in reference_items:
            row = QFrame()
            row.setStyleSheet("background-color: #161c28; border-radius: 6px; padding: 4px 8px;")
            r_layout = QHBoxLayout(row)
            r_layout.setContentsMargins(6, 4, 6, 4)
            lbl_gest = QLabel(icon_gest)
            lbl_gest.setStyleSheet("font-weight: bold; color: #ffffff; min-width: 110px;")
            lbl_desc = QLabel(desc)
            lbl_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
            r_layout.addWidget(lbl_gest)
            r_layout.addWidget(lbl_desc, 1)
            ref_widget_layout.addWidget(row)

        ref_scroll.setWidget(ref_widget)
        ref_layout.addWidget(ref_scroll)
        tab_tel_layout.addWidget(grp_ref, 1)

        self.tabs.addTab(tab_telemetry, "📊 Telemetry & Guide")

        # Tab 2: Integrated Video Player
        self.video_player = VideoPlayerWidget(self)
        self.tabs.addTab(self.video_player, "🎬 Built-in Video Player")

        splitter.addWidget(self.tabs)
        splitter.setSizes([650, 450])

        main_layout.addWidget(splitter, 1)

    def _connect_signals(self):
        self.btn_mode_ppt.clicked.connect(lambda: self._set_mode("PowerPoint"))
        self.btn_mode_video.clicked.connect(lambda: self._set_mode("Video"))
        self.btn_mode_mouse.clicked.connect(lambda: self._set_mode("AirMouse"))
        self.btn_settings.clicked.connect(self._open_settings)

        self.btn_start_cam.clicked.connect(self._start_camera)
        self.btn_stop_cam.clicked.connect(self._stop_camera)

    def _set_mode(self, mode: str):
        self.config.active_mode = mode
        self.config.save()
        if self.worker:
            self.worker.update_config(self.config)
        # If user switches to Video Mode, switch to the Video Player tab automatically for instant demonstration
        if mode == "Video":
            self.tabs.setCurrentIndex(1)
            # Ensure demo video is ready in player
            self.video_player.load_default_demo()

    def _open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.saved_config
            if self.worker:
                self.worker.update_config(self.config)

    def _check_and_start_camera(self):
        # Auto-download models if missing
        ok, msg = ensure_models_exist(self._on_download_progress)
        if not ok:
            QMessageBox.critical(self, "Model Error", msg)
            return

        # Connect internal player callback to video controller
        self._start_camera()

    def _on_download_progress(self, message: str):
        self.camera_label.setText(message)

    def _start_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = VideoProcessingWorker(self.config)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.status_message.connect(self._on_status_message)
        self.worker.error_occurred.connect(self._on_error_occurred)

        # Connect video controller internal player hook
        self.worker.video_controller.set_internal_player_callback(self._on_video_player_command)

        self.worker.start()
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)

    def _stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(2000)
            self.worker = None

        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.camera_label.setText("Camera Stopped.")

    def _on_frame_processed(self, pixmap: QPixmap, telemetry: dict, action_text: str):
        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(scaled)

        # Update telemetry labels
        fps = telemetry["fps"]
        self.val_fps.setText(f"{fps:.1f} FPS")

        if telemetry["yolo_detected"]:
            self.val_yolo_status.setText("HAND DETECTED")
            self.val_yolo_status.setStyleSheet("color: #00f2fe; font-weight: bold;")
            self.val_yolo_conf.setText(f"{telemetry['yolo_conf']:.1%} ({telemetry['yolo_ms']:.1f} ms)")
        else:
            self.val_yolo_status.setText("No Hand")
            self.val_yolo_status.setStyleSheet("color: #64748b; font-weight: normal;")
            self.val_yolo_conf.setText("0.0%")

        if telemetry["mp_detected"]:
            self.val_mp_status.setText(f"21 Points ({telemetry['mp_ms']:.1f} ms)")
            self.val_mp_status.setStyleSheet("color: #00e676;")
            self.val_hand.setText(f"{telemetry['handedness']} ({telemetry['handedness_score']:.0%})")
        else:
            self.val_mp_status.setText("Standby")
            self.val_mp_status.setStyleSheet("color: #64748b;")
            self.val_hand.setText("None")

        gest = telemetry["gesture"]
        self.val_gesture.setText(gest if gest != "NONE" else "---")
        self.val_action.setText(action_text)

    def _on_status_message(self, msg: str):
        self.lbl_cam_status.setText(f"Status: {msg}")

    def _on_error_occurred(self, err: str):
        self.lbl_cam_status.setText(f"Status: {err}")
        self.camera_label.setText(f"⚠ {err}\n\nPlease verify camera connection or select another Camera Index in Settings.")
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)

    def _on_video_player_command(self, cmd: str, arg: any):
        """Dispatches touchless video gestures directly into the built-in video player."""
        if cmd == "play":
            self.video_player.play()
        elif cmd == "pause":
            self.video_player.pause()
        elif cmd == "seek_relative":
            self.video_player.seek_relative(arg)

    def closeEvent(self, event):
        self._stop_camera()
        event.accept()
