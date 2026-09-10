"""
Settings and Configuration Modal Dialog
Allows interactive tuning of camera, thresholds, sensitivities, and audio feedback.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)
from config import AppConfig


class SettingsDialog(QDialog):
    def __init__(self, current_config: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Settings & Gesture Tuning")
        self.resize(480, 560)
        self.config = current_config
        self.saved_config = None

        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0f131a;
                color: #e2e8f0;
            }
            QGroupBox {
                background-color: #151a24;
                border: 1px solid #242c3d;
                border-radius: 8px;
                margin-top: 14px;
                padding: 14px;
                font-weight: bold;
                color: #00f2fe;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e2636;
                color: #ffffff;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 80px;
            }
            QCheckBox {
                color: #cbd5e1;
                font-size: 11px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #1e2636;
            }
            QCheckBox::indicator:checked {
                background-color: #00f2fe;
                border-color: #00f2fe;
            }
            QPushButton {
                background-color: #1e2636;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2a354c;
                border-color: #00f2fe;
            }
            QPushButton#saveBtn {
                background-color: #008080;
                border-color: #00f2fe;
                color: #ffffff;
            }
            QPushButton#saveBtn:hover {
                background-color: #00a3a3;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Vision & Detection Group
        grp_vision = QGroupBox("Camera & Object Detection")
        form_vision = QFormLayout(grp_vision)
        form_vision.setSpacing(10)

        self.spin_cam_idx = QSpinBox()
        self.spin_cam_idx.setRange(0, 10)
        form_vision.addRow("Camera Index:", self.spin_cam_idx)

        self.spin_yolo_conf = QDoubleSpinBox()
        self.spin_yolo_conf.setRange(0.10, 0.95)
        self.spin_yolo_conf.setSingleStep(0.05)
        self.spin_yolo_conf.setDecimals(2)
        form_vision.addRow("YOLO Confidence Threshold:", self.spin_yolo_conf)

        self.combo_hand = QComboBox()
        self.combo_hand.addItems(["Auto", "Right", "Left"])
        form_vision.addRow("Hand Selection:", self.combo_hand)

        self.check_mirror = QCheckBox("Mirror Camera Preview (Horizontal Flip)")
        form_vision.addRow("", self.check_mirror)

        layout.addWidget(grp_vision)

        # 2. Gesture Stability & Anti-Flutter
        grp_gest = QGroupBox("Gesture Confirmation & Cooldowns")
        form_gest = QFormLayout(grp_gest)
        form_gest.setSpacing(10)

        self.spin_stability = QSpinBox()
        self.spin_stability.setRange(2, 20)
        form_gest.addRow("Stability Frames (Confirmation Delay):", self.spin_stability)

        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.2, 5.0)
        self.spin_cooldown.setSingleStep(0.2)
        self.spin_cooldown.setDecimals(1)
        form_gest.addRow("Gesture Action Cooldown (seconds):", self.spin_cooldown)

        self.spin_swipe_thresh = QDoubleSpinBox()
        self.spin_swipe_thresh.setRange(0.05, 0.40)
        self.spin_swipe_thresh.setSingleStep(0.02)
        self.spin_swipe_thresh.setDecimals(2)
        form_gest.addRow("Swipe Displacement Threshold:", self.spin_swipe_thresh)

        self.spin_swipe_cd = QDoubleSpinBox()
        self.spin_swipe_cd.setRange(0.3, 3.0)
        self.spin_swipe_cd.setSingleStep(0.2)
        self.spin_swipe_cd.setDecimals(1)
        form_gest.addRow("Swipe Cooldown (seconds):", self.spin_swipe_cd)

        layout.addWidget(grp_gest)

        # 3. Air Mouse & Feedback
        grp_mouse = QGroupBox("Air Mouse & Audio Feedback")
        form_mouse = QFormLayout(grp_mouse)
        form_mouse.setSpacing(10)

        self.spin_mouse_sens = QDoubleSpinBox()
        self.spin_mouse_sens.setRange(0.5, 4.0)
        self.spin_mouse_sens.setSingleStep(0.2)
        self.spin_mouse_sens.setDecimals(1)
        form_mouse.addRow("Cursor Sensitivity:", self.spin_mouse_sens)

        self.spin_mouse_smooth = QDoubleSpinBox()
        self.spin_mouse_smooth.setRange(0.05, 0.90)
        self.spin_mouse_smooth.setSingleStep(0.05)
        self.spin_mouse_smooth.setDecimals(2)
        form_mouse.addRow("Smoothing Alpha (Lower = Smoother):", self.spin_mouse_smooth)

        self.check_sound = QCheckBox("Enable Audio Sound Feedback (Beep on Action)")
        form_mouse.addRow("", self.check_sound)

        layout.addWidget(grp_mouse)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_defaults = QPushButton("Reset Defaults")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save & Apply")
        self.btn_save.setObjectName("saveBtn")

        self.btn_defaults.clicked.connect(self._reset_defaults)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save_and_accept)

        btn_layout.addWidget(self.btn_defaults)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

    def _load_from_config(self):
        c = self.config
        self.spin_cam_idx.setValue(c.camera_index)
        self.spin_yolo_conf.setValue(c.yolo_confidence_threshold)
        idx = self.combo_hand.findText(c.hand_selection)
        if idx >= 0:
            self.combo_hand.setCurrentIndex(idx)
        self.check_mirror.setChecked(c.mirror_camera)

        self.spin_stability.setValue(c.gesture_stability_frames)
        self.spin_cooldown.setValue(c.gesture_cooldown_seconds)
        self.spin_swipe_thresh.setValue(c.swipe_displacement_threshold)
        self.spin_swipe_cd.setValue(c.swipe_cooldown_seconds)

        self.spin_mouse_sens.setValue(c.air_mouse_sensitivity)
        self.spin_mouse_smooth.setValue(c.air_mouse_smoothing)
        self.check_sound.setChecked(c.sound_feedback)

    def _reset_defaults(self):
        self.config = AppConfig()
        self._load_from_config()

    def _save_and_accept(self):
        c = self.config
        c.camera_index = self.spin_cam_idx.value()
        c.yolo_confidence_threshold = self.spin_yolo_conf.value()
        c.hand_selection = self.combo_hand.currentText()
        c.mirror_camera = self.check_mirror.isChecked()

        c.gesture_stability_frames = self.spin_stability.value()
        c.gesture_cooldown_seconds = self.spin_cooldown.value()
        c.swipe_displacement_threshold = self.spin_swipe_thresh.value()
        c.swipe_cooldown_seconds = self.spin_swipe_cd.value()

        c.air_mouse_sensitivity = self.spin_mouse_sens.value()
        c.air_mouse_smoothing = self.spin_mouse_smooth.value()
        c.sound_feedback = self.check_sound.isChecked()

        c.save()
        self.saved_config = c
        self.accept()
