import sys
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    """Session-scoped QApplication for all tests that need Qt widgets."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
