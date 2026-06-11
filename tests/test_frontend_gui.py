import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from src.frontend.gui_client import CameraClientWindow


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


def test_frontend_gui_defaults():
    _app()
    window = CameraClientWindow()

    assert window.host_edit.text() == "127.0.0.1"
    assert window.port_spin.value() == 5001
    assert window.camera_spin.value() == 0
    assert window.width_spin.value() == 640
    assert window.fps_spin.value() == 8
    assert window.quality_spin.value() == 80
    assert "未连接" in window.status_label.text()
    assert window.stop_button.isEnabled() is False
