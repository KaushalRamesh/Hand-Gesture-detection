"""
Main Application Entrypoint
Touchless PowerPoint and Video Control via YOLO & MediaPipe Hand Gesture Detection.
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from ui.dashboard import GestureDashboardWindow


def main():
    # Configure high-DPI display attributes
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AI Hand Gesture Control System")
    app.setOrganizationName("AIVision")

    window = GestureDashboardWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
