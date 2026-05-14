from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Sound Morpher")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SoundMorpher")

    icon_path = Path(__file__).parent / "resources" / "icons" / "app.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = Path(__file__).parent / "resources" / "theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
