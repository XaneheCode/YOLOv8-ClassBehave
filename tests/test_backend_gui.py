import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtWidgets

from src.backend.app import DEFAULT_MODEL_PATH
from src.backend.gui_app import BackendMonitorWindow, BackendReceiverWorker
from src.common.protocol import FramePacket
from src.common.types import AlarmState, Detection, DetectionAssessment


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
        assert window.output_edit.text() == "output/alarms"
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


class FakeDetector:
    def __init__(self, model_path):
        self.model_path = model_path

    def detect(self, frame):
        return []


class FakeSocket:
    def __init__(self):
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def shutdown(self, how):
        pass

    def close(self):
        self.closed = True


class FakeServerSocket(FakeSocket):
    def __init__(self, conn=None):
        super().__init__()
        self.conn = conn or FakeSocket()

    def setsockopt(self, level, optname, value):
        pass

    def bind(self, address):
        self.address = address

    def listen(self, backlog):
        self.backlog = backlog

    def accept(self):
        return self.conn, ("127.0.0.1", 61234)


def test_worker_reports_protocol_value_error_without_idle_status(monkeypatch):
    _app()
    server = FakeServerSocket()
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", FakeDetector)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr(
        "src.backend.gui_app.recv_packet",
        lambda conn: (_ for _ in ()).throw(ValueError("Invalid frame magic")),
    )
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )
    errors = []
    statuses = []
    worker.error_occurred.connect(errors.append)
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert any("Invalid frame magic" in error for error in errors)
    assert statuses[-1] != "未监听"
    assert server.closed is True
    assert server.conn.closed is True


class FakeAlarmAnalyzer:
    def __init__(self, threshold_seconds):
        self.threshold_seconds = threshold_seconds

    def update(self, detections, now_seconds):
        assessment = DetectionAssessment(
            detection=Detection(label="Sleeping", confidence=0.9, bbox=(0, 0, 10, 10)),
            status="abnormal",
            is_abnormal=True,
            is_alarm=True,
            reason="Sleeping",
            duration_seconds=3.0,
        )
        alarm = AlarmState(
            is_alarm=True,
            suspicious=True,
            duration_seconds=3.0,
            reason="multi_behaviour_abnormal",
            abnormal_count=1,
            abnormal_labels=("Sleeping",),
        )
        return [assessment], alarm


def test_worker_does_not_log_or_append_alarm_when_screenshot_save_fails(monkeypatch, tmp_path):
    _app()
    server = FakeServerSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir=tmp_path,
    )

    def fake_recv_packet(conn):
        worker._running = False
        return FramePacket(frame_id=7, timestamp_ms=123456, image_bytes=b"jpeg")

    appended = []
    logs = []
    errors = []
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", FakeDetector)
    monkeypatch.setattr("src.backend.gui_app.BehaviourAnalyzer", FakeAlarmAnalyzer)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr("src.backend.gui_app.recv_packet", fake_recv_packet)
    monkeypatch.setattr(
        "src.backend.gui_app.decode_jpeg",
        lambda data: np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr("src.backend.gui_app.draw_overlay", lambda frame, assessments, alarm, fps, latency_ms: frame)
    monkeypatch.setattr("src.backend.gui_app.cv2.imwrite", lambda path, image: False)
    monkeypatch.setattr("src.backend.gui_app.append_alarm", lambda *args, **kwargs: appended.append(args))
    worker.error_occurred.connect(errors.append)
    worker.log_ready.connect(logs.append)

    worker.run()

    assert any("报警截图保存失败" in error for error in errors)
    assert appended == []
    assert logs == []
