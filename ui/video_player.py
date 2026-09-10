"""
Integrated PySide6 Video Player Widget
Allows demonstration of touchless gesture control directly inside the application.
"""

from pathlib import Path
from PySide6.QtCore import QTime, QUrl, Qt
from PySide6.QtGui import QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Video display
        self.video_widget.setMinimumHeight(240)
        self.video_widget.setStyleSheet("background-color: #0b0e14; border-radius: 8px; border: 1px solid #242c3d;")
        layout.addWidget(self.video_widget, 1)

        # Timeline and time label
        time_layout = QHBoxLayout()
        time_layout.setSpacing(10)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1c2230;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00f2fe;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #00f2fe;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #a0aec0; font-family: 'Consolas', monospace; font-size: 11px;")

        time_layout.addWidget(self.slider, 1)
        time_layout.addWidget(self.time_label)
        layout.addLayout(time_layout)

        # Controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.btn_play = QPushButton("▶ Play")
        self.btn_prev = QPushButton("⏪ -5s")
        self.btn_next = QPushButton("⏩ +5s")
        self.btn_load_demo = QPushButton("🎬 Demo Clip")
        self.btn_open = QPushButton("📂 Open File...")

        button_style = """
            QPushButton {
                background-color: #1a2233;
                color: #e2e8f0;
                border: 1px solid #2d3748;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #243049;
                border-color: #00f2fe;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0d1524;
            }
        """
        for btn in (self.btn_play, self.btn_prev, self.btn_next, self.btn_load_demo, self.btn_open):
            btn.setStyleSheet(button_style)

        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #008080;
                color: #ffffff;
                border: 1px solid #00f2fe;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00a3a3;
            }
        """)

        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_load_demo)
        controls_layout.addWidget(self.btn_open)

        layout.addLayout(controls_layout)

    def _connect_signals(self):
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.btn_prev.clicked.connect(lambda: self.seek_relative(-5000))
        self.btn_next.clicked.connect(lambda: self.seek_relative(5000))
        self.btn_load_demo.clicked.connect(self.load_default_demo)
        self.btn_open.clicked.connect(self.open_file_dialog)

        self.slider.sliderMoved.connect(self.set_position)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.playbackStateChanged.connect(self.playback_state_changed)

    def load_video(self, file_path: Path):
        path = Path(file_path)
        if path.exists():
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.player.play()

    def load_default_demo(self):
        demo_path = Path(__file__).resolve().parent.parent / "assets" / "sample_video.mp4"
        if demo_path.exists():
            self.load_video(demo_path)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        if file_path:
            self.load_video(Path(file_path))

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def toggle_play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.pause()
        else:
            self.play()

    def seek_relative(self, delta_ms: int):
        current = self.player.position()
        target = max(0, min(self.player.duration(), current + delta_ms))
        self.player.setPosition(target)

    def position_changed(self, position: int):
        if not self.slider.isSliderDown() and self.player.duration() > 0:
            self.slider.setValue(int(position * 1000 / self.player.duration()))
        self._update_time_label()

    def duration_changed(self, duration: int):
        self._update_time_label()

    def set_position(self, value: int):
        if self.player.duration() > 0:
            pos = int(value * self.player.duration() / 1000)
            self.player.setPosition(pos)

    def playback_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setText("⏸ Pause")
        else:
            self.btn_play.setText("▶ Play")

    def _update_time_label(self):
        pos_sec = self.player.position() // 1000
        dur_sec = self.player.duration() // 1000
        t_pos = QTime(0, pos_sec // 60, pos_sec % 60).toString("mm:ss")
        t_dur = QTime(0, dur_sec // 60, dur_sec % 60).toString("mm:ss")
        self.time_label.setText(f"{t_pos} / {t_dur}")
