import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtWidgets

from src.backend.app import DEFAULT_MODEL_PATH
from src.backend.gui_app import BackendMonitorWindow, BackendReceiverWorker, LocalMediaWorker
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
        assert window.qwen_window.windowTitle() == "千问分析结果"
        assert window.image_test_button.text() == "选择图片测试"
        assert window.video_test_button.text() == "选择视频测试"
        assert window.stop_test_button.text() == "停止测试"
        assert window.stop_test_button.isEnabled() is False
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


class ControlledServerSocket(FakeSocket):
    def __init__(self, accepts):
        super().__init__()
        self.accepts = list(accepts)
        self.accept_calls = 0

    def setsockopt(self, level, optname, value):
        pass

    def bind(self, address):
        self.address = address

    def listen(self, backlog):
        self.backlog = backlog

    def accept(self):
        self.accept_calls += 1
        item = self.accepts.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item, ("127.0.0.1", 61234)


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


def test_worker_skips_packet_when_stopped_during_receive(monkeypatch):
    _app()
    server = FakeServerSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )

    def fake_recv_packet(conn):
        worker._running = False
        return FramePacket(frame_id=7, timestamp_ms=123456, image_bytes=b"jpeg")

    decode_calls = []
    detect_calls = []

    class RecordingDetector:
        def __init__(self, model_path):
            self.model_path = model_path

        def detect(self, frame):
            detect_calls.append(frame)
            return []

    monkeypatch.setattr("src.backend.gui_app.YoloDetector", RecordingDetector)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr("src.backend.gui_app.recv_packet", fake_recv_packet)
    monkeypatch.setattr(
        "src.backend.gui_app.decode_jpeg",
        lambda data: decode_calls.append(data) or np.zeros((20, 20, 3), dtype=np.uint8),
    )

    worker.run()

    assert decode_calls == []
    assert detect_calls == []


def test_worker_clean_disconnect_returns_to_waiting_without_error(monkeypatch):
    _app()
    first_conn = FakeSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )

    def stop_accepting():
        worker._running = False
        return OSError("server stopped")

    server = ControlledServerSocket([first_conn, stop_accepting])
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", FakeDetector)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr(
        "src.backend.gui_app.recv_packet",
        lambda conn: (_ for _ in ()).throw(ConnectionError("Socket closed before enough bytes were received")),
    )
    errors = []
    statuses = []
    worker.error_occurred.connect(errors.append)
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert errors == []
    assert any("前端已断开" in status for status in statuses)
    assert server.accept_calls == 2
    assert first_conn.closed is True


def test_worker_skips_bad_frame_and_keeps_receiving(monkeypatch):
    _app()
    server = FakeServerSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )
    recv_calls = []

    def fake_recv_packet(conn):
        recv_calls.append(conn)
        if len(recv_calls) == 1:
            return FramePacket(frame_id=1, timestamp_ms=123456, image_bytes=b"bad")
        worker._running = False
        raise ConnectionError("Socket closed before enough bytes were received")

    logs = []
    errors = []
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", FakeDetector)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr("src.backend.gui_app.recv_packet", fake_recv_packet)
    monkeypatch.setattr(
        "src.backend.gui_app.decode_jpeg",
        lambda data: (_ for _ in ()).throw(ValueError("failed to decode JPEG frame")),
    )
    worker.log_ready.connect(logs.append)
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert len(recv_calls) == 2
    assert any("帧处理失败" in log and "failed to decode JPEG frame" in log for log in logs)
    assert not any("处理画面失败" in error for error in errors)


def test_worker_emits_raw_frame_for_qwen_after_detection(monkeypatch):
    _app()
    server = FakeServerSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )
    recv_count = 0
    detections = [
        Detection(label="Reading", confidence=0.9, bbox=(1, 2, 3, 4)),
        Detection(label="Sleeping", confidence=0.8, bbox=(5, 6, 7, 8)),
    ]

    def fake_recv_packet(conn):
        nonlocal recv_count
        recv_count += 1
        if recv_count == 1:
            return FramePacket(frame_id=7, timestamp_ms=123456, image_bytes=b"jpeg")
        worker._running = False
        raise ConnectionError("Socket closed before enough bytes were received")

    class DetectionDetector:
        def __init__(self, model_path):
            self.model_path = model_path

        def detect(self, frame):
            return detections

    qwen_frames = []
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", DetectionDetector)
    monkeypatch.setattr("src.backend.gui_app.socket.socket", lambda *args, **kwargs: server)
    monkeypatch.setattr("src.backend.gui_app.recv_packet", fake_recv_packet)
    monkeypatch.setattr(
        "src.backend.gui_app.decode_jpeg",
        lambda data: np.zeros((20, 30, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "src.backend.gui_app.draw_overlay",
        lambda frame, assessments, alarm, fps, latency_ms: frame,
    )
    worker.qwen_frame_ready.connect(lambda frame, target_count: qwen_frames.append((frame, target_count)))

    worker.run()

    assert len(qwen_frames) == 1
    assert qwen_frames[0][0].shape == (20, 30, 3)
    assert qwen_frames[0][1] == 2


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


def test_local_image_worker_processes_one_frame_and_emits_display_updates(monkeypatch):
    _app()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    detections = [Detection(label="Reading", confidence=0.9, bbox=(1, 2, 10, 12))]

    class OneDetectionDetector:
        def __init__(self, model_path):
            self.model_path = model_path

        def detect(self, image):
            return detections

    monkeypatch.setattr("src.backend.gui_app.YoloDetector", OneDetectionDetector)
    monkeypatch.setattr("src.backend.gui_app.cv2.imread", lambda path: frame.copy())
    monkeypatch.setattr("src.backend.gui_app.draw_overlay", lambda image, assessments, alarm, fps, latency_ms: image)

    worker = LocalMediaWorker(
        media_path="demo.jpg",
        media_type="image",
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )
    frames = []
    qwen_frames = []
    metrics = []
    counts = []
    statuses = []
    alarms = []
    worker.frame_ready.connect(frames.append)
    worker.qwen_frame_ready.connect(lambda image, target_count: qwen_frames.append((image, target_count)))
    worker.metrics_ready.connect(lambda fps, resolution, frame_count, latency_ms: metrics.append((resolution, frame_count)))
    worker.counts_ready.connect(counts.append)
    worker.status_changed.connect(statuses.append)
    worker.alarm_changed.connect(lambda status, is_alarm: alarms.append((status, is_alarm)))

    worker.run()

    assert len(frames) == 1
    assert len(qwen_frames) == 1
    assert qwen_frames[0][0].shape == frame.shape
    assert qwen_frames[0][1] == 1
    assert metrics == [("32x24", 1)]
    assert counts[-1]["看书"] == 1
    assert statuses[0] == "加载模型"
    assert "测试完成" in statuses[-1]
    assert alarms[-1][1] is False


def test_local_image_worker_reports_unreadable_file(monkeypatch):
    _app()
    monkeypatch.setattr("src.backend.gui_app.YoloDetector", FakeDetector)
    monkeypatch.setattr("src.backend.gui_app.cv2.imread", lambda path: None)
    worker = LocalMediaWorker(
        media_path="missing.jpg",
        media_type="image",
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir="output/alarms",
    )
    errors = []
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert errors == ["无法读取图片：missing.jpg"]


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
    recv_count = 0

    def fake_recv_packet(conn):
        nonlocal recv_count
        recv_count += 1
        if recv_count == 1:
            return FramePacket(frame_id=7, timestamp_ms=123456, image_bytes=b"jpeg")
        worker._running = False
        raise ConnectionError("Socket closed before enough bytes were received")

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


def test_worker_removes_saved_screenshot_when_alarm_csv_append_fails(monkeypatch, tmp_path):
    _app()
    server = FakeServerSocket()
    worker = BackendReceiverWorker(
        host="0.0.0.0",
        port=5001,
        model=DEFAULT_MODEL_PATH,
        alarm_seconds=3.0,
        output_dir=tmp_path,
    )
    recv_count = 0

    def fake_recv_packet(conn):
        nonlocal recv_count
        recv_count += 1
        if recv_count == 1:
            return FramePacket(frame_id=7, timestamp_ms=123456, image_bytes=b"jpeg")
        worker._running = False
        raise ConnectionError("Socket closed before enough bytes were received")

    def fake_imwrite(path, image):
        Path(path).write_bytes(b"image")
        return True

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
    monkeypatch.setattr("src.backend.gui_app.cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(
        "src.backend.gui_app.append_alarm",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    worker.error_occurred.connect(errors.append)
    worker.log_ready.connect(logs.append)

    worker.run()

    assert list(tmp_path.glob("*.jpg")) == []
    assert any("报警记录写入失败" in error and "disk full" in error for error in errors)
    assert logs == []
