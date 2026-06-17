import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtWidgets

from src.backend.app import DEFAULT_MODEL_PATH
from src.backend.gui_app import (
    BackendMonitorWindow,
    BackendReceiverWorker,
    LocalMediaWorker,
    QwenWorker,
    DEFAULT_PERSON_MODEL_PATH,
    INFERENCE_MODE_PERSON_VLM,
    format_qwen_result_details,
    qwen_person_color,
)
from src.backend.qwen_analysis import QwenAnalysisResult, QwenPerson, QwenSettings
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
        assert window.mode_combo.currentData() == INFERENCE_MODE_PERSON_VLM
        assert DEFAULT_PERSON_MODEL_PATH == "yolov8s.pt"
        assert window.model_edit.text() == DEFAULT_PERSON_MODEL_PATH
        assert window.alarm_spin.value() == 3.0
        assert window.output_edit.text() == "output/alarms"
        assert "未监听" in window.status_label.text()
        assert window.stop_button.isEnabled() is False
        assert window.qwen_window.windowTitle() == "大模型分析结果"
        assert window.qwen_window.result_group.title() == "分析结果"
        assert window.qwen_window.status_group.title() == "大模型状态"
        assert window.qwen_window.text_browser is window.qwen_window.result_browser
        assert window.image_test_button.text() == "选择图片测试"
        assert window.video_test_button.text() == "选择视频测试"
        assert window.stop_test_button.text() == "停止测试"
        assert window.stop_test_button.isEnabled() is False
        assert window.centralWidget().objectName() == "qtDashboardShell"
        assert window.page_scroll.objectName() == "dashboardScroll"
        assert window.page_scroll.widgetResizable() is True
        assert window.findChildren(QtWidgets.QFrame, "dashboardSidebar") == []
        assert "NSGD" in window.protocol_badge.text()
        assert "TCP" in window.protocol_badge.text()
        assert window.log_text.maximumHeight() <= 96
        assert window.video_label.minimumHeight() <= 260
    finally:
        window.close()
        window.deleteLater()


def test_qwen_result_details_include_numbered_six_class_labels():
    result = QwenAnalysisResult(
        people=[
            QwenPerson(bbox=[10, 20, 60, 110], label="Writing", status="坐在座位上，正在写字", confidence="high"),
            QwenPerson(bbox=[100, 30, 160, 130], label="Head-down", status="低头看向桌面", confidence="medium"),
        ],
        summary="多名学生在教室内学习。",
        raw_text="{}",
    )

    details = format_qwen_result_details(result)

    assert "大模型概括：多名学生在教室内学习。" in details
    assert "1. 类别：学习，坐标 [10, 20, 60, 110]，状态：坐在座位上，正在写字，置信度：high" in details
    assert "2. 类别：低头，坐标 [100, 30, 160, 130]，状态：低头看向桌面，置信度：medium" in details


def test_backend_qwen_window_updates_result_and_status_separately():
    _app()
    window = BackendMonitorWindow()
    try:
        qwen_window = window.qwen_window
        qwen_window.show = lambda: None

        qwen_window.show_pending()

        assert "正在上传" in qwen_window.status_browser.toPlainText()
        assert "等待分析结果" in qwen_window.result_browser.toPlainText()

        result = QwenAnalysisResult(
            people=[QwenPerson(bbox=[1, 2, 20, 30], label="Reading", status="正在看书", confidence="high")],
            summary="一名学生正在看书。",
            raw_text="{}",
        )
        qwen_window.show_result(np.zeros((40, 50, 3), dtype=np.uint8), result)

        assert "类别：学习" in qwen_window.result_browser.toPlainText()
        assert "分析完成" in qwen_window.status_browser.toPlainText()
    finally:
        window.close()
        window.deleteLater()


def test_mode_selector_switches_between_person_and_six_class_models():
    _app()
    window = BackendMonitorWindow()
    try:
        window.mode_combo.setCurrentIndex(1)
        assert window.model_edit.text() == DEFAULT_MODEL_PATH

        window.mode_combo.setCurrentIndex(0)
        assert window.model_edit.text() == DEFAULT_PERSON_MODEL_PATH
    finally:
        window.close()
        window.deleteLater()


def test_backend_gui_selects_large_model_and_target_limit(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-5.5")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_VL_MODEL", "qwen3.6-flash")
    monkeypatch.setenv("QWEN_MAX_YOLO_TARGETS", "30")

    _app()
    window = BackendMonitorWindow()
    try:
        assert window.vision_model_combo.currentData() == "openai"
        assert "GPT-5.5" in window.vision_model_combo.currentText()
        assert window.vision_target_spin.value() == 30

        qwen_index = window.vision_model_combo.findData("dashscope")
        window.vision_model_combo.setCurrentIndex(qwen_index)
        window.vision_target_spin.setValue(42)

        settings = window._selected_qwen_settings()

        assert settings.provider == "dashscope"
        assert settings.api_key == "qwen-key"
        assert settings.model == "qwen3.6-flash"
        assert settings.max_yolo_targets == 42
    finally:
        window.close()
        window.deleteLater()


def test_qwen_worker_person_mode_classifies_numbered_crop_grid(monkeypatch):
    _app()
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    detections = [Detection(label="person", confidence=0.91, bbox=(10, 10, 40, 60))]
    settings = QwenSettings(api_key="key", model="gpt-5.5", interval_seconds=10, provider="openai")
    captured = {}

    def fake_call_person_crop_vision(grid_image, source_by_id, settings):
        captured["grid_shape"] = grid_image.shape
        captured["source_by_id"] = source_by_id
        return QwenAnalysisResult(
            people=[QwenPerson(bbox=[10, 10, 40, 60], label="Writing", status="正在写字", confidence="high")],
            summary="ok",
            raw_text="{}",
        )

    monkeypatch.setattr("src.backend.gui_app.call_person_crop_vision", fake_call_person_crop_vision)

    worker = QwenWorker(frame, settings, detections=detections, mode=INFERENCE_MODE_PERSON_VLM)
    results = []
    worker.succeeded.connect(lambda display_frame, result: results.append((display_frame, result)))
    worker.run()

    assert results[0][0].shape == frame.shape
    assert results[0][1].people[0].label == "Writing"
    assert captured["source_by_id"][1].bbox == (10, 10, 40, 60)
    assert captured["grid_shape"][0] > 0


def test_qwen_person_color_uses_six_class_palette():
    assert qwen_person_color("Hand-raise").getRgb()[:3] == (37, 99, 235)
    assert qwen_person_color("Reading").getRgb()[:3] == (22, 163, 74)
    assert qwen_person_color("Writing").getRgb()[:3] == (22, 163, 74)
    assert qwen_person_color("Useing-Phone").getRgb()[:3] == (220, 38, 38)
    assert qwen_person_color("Head-down").getRgb()[:3] == (234, 88, 12)
    assert qwen_person_color("Sleeping").getRgb()[:3] == (147, 51, 234)


def test_qwen_in_flight_is_released_only_after_result_or_error():
    _app()
    window = BackendMonitorWindow()
    try:
        window._qwen_in_flight = True
        window.qwen_worker = object()

        window._on_qwen_worker_finished()

        assert window.qwen_worker is None
        assert window._qwen_in_flight is True

        window._show_qwen_error(None, "timeout")

        assert window._qwen_in_flight is False
    finally:
        window.close()
        window.deleteLater()


def test_qwen_error_starts_cooldown_before_next_upload():
    _app()
    window = BackendMonitorWindow()
    try:
        window._qwen_in_flight = True
        window._show_qwen_error(None, "network")

        assert window._qwen_in_flight is False
        assert window._qwen_cooldown_until > 0

        window._handle_qwen_frame(np.zeros((20, 20, 3), dtype=np.uint8), target_count=0)

        assert window.qwen_worker is None
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
    worker.qwen_frame_ready.connect(lambda frame, detections: qwen_frames.append((frame, detections)))

    worker.run()

    assert len(qwen_frames) == 1
    assert qwen_frames[0][0].shape == (20, 30, 3)
    assert len(qwen_frames[0][1]) == 2
    assert qwen_frames[0][1][0].label == "Reading"


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
    worker.qwen_frame_ready.connect(lambda image, detections: qwen_frames.append((image, detections)))
    worker.metrics_ready.connect(lambda fps, resolution, frame_count, latency_ms: metrics.append((resolution, frame_count)))
    worker.counts_ready.connect(counts.append)
    worker.status_changed.connect(statuses.append)
    worker.alarm_changed.connect(lambda status, is_alarm: alarms.append((status, is_alarm)))

    worker.run()

    assert len(frames) == 1
    assert len(qwen_frames) == 1
    assert qwen_frames[0][0].shape == frame.shape
    assert len(qwen_frames[0][1]) == 1
    assert qwen_frames[0][1][0].label == "Reading"
    assert metrics == [("32x24", 1)]
    assert counts[-1]["学习"] == 1
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
