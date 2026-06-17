import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from src.frontend.gui_client import CameraClientWindow, CameraSenderWorker


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


def test_frontend_gui_defaults():
    _app()
    window = CameraClientWindow()
    try:
        assert window.host_edit.text() == "127.0.0.1"
        assert window.port_spin.value() == 5001
        assert window.camera_spin.value() == 0
        assert window.width_spin.value() == 640
        assert window.fps_spin.value() == 8
        assert window.quality_spin.value() == 80
        assert "未连接" in window.status_label.text()
        assert window.stop_button.isEnabled() is False
        assert window.centralWidget().objectName() == "qtDashboardShell"
        assert window.page_scroll.objectName() == "dashboardScroll"
        assert window.page_scroll.widgetResizable() is True
        assert window.findChildren(QtWidgets.QFrame, "dashboardSidebar") == []
        assert "NSGD" in window.protocol_badge.text()
        assert "TCP" in window.protocol_badge.text()
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


def test_stop_sender_keeps_start_disabled_while_worker_is_stopping():
    _app()
    window = CameraClientWindow()
    try:
        worker = FakeWorker(running=True)
        window.worker = worker

        window.stop_sender()

        assert worker.stopped is True
        assert window.start_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
        assert "停止中" in window.status_label.text()
    finally:
        window.worker = None
        window.close()
        window.deleteLater()


def test_stop_sender_returns_to_idle_when_worker_is_not_running():
    _app()
    window = CameraClientWindow()
    try:
        worker = FakeWorker(running=False)
        window.worker = worker

        window.stop_sender()

        assert worker.stopped is True
        assert window.start_button.isEnabled() is True
        assert window.stop_button.isEnabled() is False
    finally:
        window.worker = None
        window.close()
        window.deleteLater()


def test_close_event_ignores_close_while_worker_is_stopping():
    _app()
    window = CameraClientWindow()
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


class FakeClosedCapture:
    def __init__(self, camera):
        self.camera = camera
        self.released = False

    def isOpened(self):
        return False

    def release(self):
        self.released = True


def test_worker_camera_open_error_is_not_overwritten_by_disconnected_status(monkeypatch):
    _app()
    capture = FakeClosedCapture(camera=0)
    monkeypatch.setattr("src.frontend.gui_client.cv2.VideoCapture", lambda camera: capture)
    worker = CameraSenderWorker(
        host="127.0.0.1",
        port=5001,
        camera=0,
        width=640,
        fps=8,
        quality=80,
    )
    errors = []
    statuses = []
    worker.error_occurred.connect(errors.append)
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert errors == ["无法打开摄像头 0"]
    assert "已断开" not in statuses
    assert capture.released is True
