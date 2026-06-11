import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from src.backend.app import DEFAULT_MODEL_PATH
from src.backend.gui_app import BackendMonitorWindow


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


def test_backend_gui_defaults():
    _app()
    window = BackendMonitorWindow()
    try:
        assert window.host_edit.text() == "0.0.0.0"
        assert window.port_spin.value() == 5001
        assert window.model_edit.text() == DEFAULT_MODEL_PATH
        assert window.alarm_spin.value() == 3.0
        assert "未监听" in window.status_label.text()
        assert window.stop_button.isEnabled() is False
    finally:
        window.close()
        window.deleteLater()


class FakeWorker:
    def __init__(self, running=True):
        self.running = running
        self.stopped = False

    def stop(self):
        self.stopped = True

    def isRunning(self):
        return self.running


class FakeCloseEvent:
    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True


def test_stop_backend_keeps_start_disabled_while_worker_is_stopping():
    _app()
    window = BackendMonitorWindow()
    try:
        worker = FakeWorker(running=True)
        window.worker = worker

        window.stop_backend()

        assert worker.stopped is True
        assert window.start_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
        assert "停止中" in window.status_label.text()
    finally:
        window.worker = None
        window.close()
        window.deleteLater()


def test_stop_backend_returns_to_idle_when_worker_is_not_running():
    _app()
    window = BackendMonitorWindow()
    try:
        worker = FakeWorker(running=False)
        window.worker = worker

        window.stop_backend()

        assert worker.stopped is True
        assert window.start_button.isEnabled() is True
        assert window.stop_button.isEnabled() is False
    finally:
        window.worker = None
        window.close()
        window.deleteLater()


def test_close_event_ignores_close_while_worker_is_stopping():
    _app()
    window = BackendMonitorWindow()
    try:
        worker = FakeWorker(running=True)
        window.worker = worker
        event = FakeCloseEvent()

        window.closeEvent(event)

        assert worker.stopped is True
        assert event.ignored is True
        assert window.start_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
        assert "停止中" in window.status_label.text()
    finally:
        window.worker = None
        window.close()
        window.deleteLater()
