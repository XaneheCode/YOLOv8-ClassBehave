from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from src.backend.app import (
    DEFAULT_MODEL_PATH,
    append_alarm,
    behaviour_counts,
    draw_overlay,
    frame_status_text,
)
from src.backend.behaviour_analyzer import BehaviourAnalyzer
from src.backend.detector import YoloDetector
from src.backend.qwen_analysis import (
    QwenAnalysisError,
    call_qwen_vision,
    load_qwen_settings,
    prepare_frame_for_qwen,
    should_upload_frame,
    should_use_qwen_for_scene,
)
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet


_APP: QtWidgets.QApplication | None = None


def _ensure_app() -> None:
    global _APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        _APP = QtWidgets.QApplication([])
    else:
        _APP = app


def cv_frame_to_qimage(frame) -> QtGui.QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QtGui.QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QtGui.QImage.Format.Format_RGB888,
    ).copy()


class QwenAnalysisWindow(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("千问分析结果")
        self.resize(900, 700)

        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(860, 520)
        self.image_label.setStyleSheet("background: #111; color: #ddd;")
        layout.addWidget(self.image_label)

        self.text_browser = QtWidgets.QTextBrowser(self)
        self.text_browser.setMinimumHeight(120)
        self.text_browser.setText("等待后端画面")
        layout.addWidget(self.text_browser)

    def show_pending(self) -> None:
        self.text_browser.setText("正在上传当前画面到千问分析...")
        self.show()

    def show_error(self, frame, message: str) -> None:
        if frame is not None:
            self.image_label.setPixmap(self._scaled_pixmap(self._frame_to_pixmap(frame)))
        self.text_browser.setText(f"千问分析失败：{message}")
        self.show()

    def show_skipped(self, frame, message: str) -> None:
        if frame is not None:
            self.image_label.setPixmap(self._scaled_pixmap(self._frame_to_pixmap(frame)))
        self.text_browser.setText(message)
        self.show()

    def show_result(self, frame, result) -> None:
        pixmap = self._frame_to_pixmap(frame)
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 255), 3)
        painter.setPen(pen)
        painter.setFont(QtGui.QFont("Microsoft YaHei", 14))

        for person in result.people:
            x1, y1, x2, y2 = person.bbox
            label = f"{person.status} ({person.confidence})"
            painter.drawRect(QtCore.QRect(x1, y1, x2 - x1, y2 - y1))
            label_rect = QtCore.QRect(x1, max(0, y1 - 30), max(220, x2 - x1), 28)
            painter.fillRect(label_rect, QtGui.QColor(255, 0, 255, 180))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            painter.drawText(label_rect.adjusted(4, 0, -4, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, label)
            painter.setPen(pen)
        painter.end()

        info = f"千问概括：{result.summary or '无'}\n\n"
        for index, person in enumerate(result.people, start=1):
            info += f"{index}. 坐标 {person.bbox}，状态：{person.status}，置信度：{person.confidence}\n"
        if not result.people:
            info += "未识别到人物框。\n"
        self.image_label.setPixmap(self._scaled_pixmap(pixmap))
        self.text_browser.setText(info)
        self.show()

    def _frame_to_pixmap(self, frame) -> QtGui.QPixmap:
        qimg = QtGui.QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format.Format_BGR888,
        )
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _scaled_pixmap(self, pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        return pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )


class QwenWorker(QtCore.QThread):
    succeeded = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(object, str)

    def __init__(self, frame, settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.frame = prepare_frame_for_qwen(frame, settings)

    def run(self) -> None:
        try:
            result = call_qwen_vision(self.frame, self.settings)
        except QwenAnalysisError as exc:
            self.failed.emit(self.frame, str(exc))
        except Exception as exc:
            self.failed.emit(self.frame, f"未知错误：{exc}")
        else:
            self.succeeded.emit(self.frame, result)


class BackendReceiverWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    qwen_frame_ready = QtCore.pyqtSignal(object, int)
    metrics_ready = QtCore.pyqtSignal(float, str, int, int)
    counts_ready = QtCore.pyqtSignal(dict)
    alarm_changed = QtCore.pyqtSignal(str, bool)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    log_ready = QtCore.pyqtSignal(str)

    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        alarm_seconds: float,
        output_dir: str | Path,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.alarm_seconds = alarm_seconds
        self.output_dir = Path(output_dir)
        self._running = False
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._socket_lock = threading.Lock()

    def stop(self) -> None:
        self._running = False
        with self._socket_lock:
            conn = self._conn
            server = self._server
        self._close_socket(conn)
        self._close_socket(server)

    def run(self) -> None:
        self._running = True
        had_error = False
        server: socket.socket | None = None
        conn: socket.socket | None = None

        try:
            self.status_changed.emit("加载模型")
            try:
                detector = YoloDetector(model_path=self.model)
            except Exception as exc:
                had_error = True
                self.error_occurred.emit(f"模型加载失败：{exc}")
                return

            analyzer = BehaviourAnalyzer(threshold_seconds=self.alarm_seconds)
            csv_path = self.output_dir / "alarms.csv"
            last_alarm_frame = -1
            frame_count = 0
            fps_started = time.time()

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(0.5)
            with self._socket_lock:
                self._server = server
            server.bind((self.host, self.port))
            server.listen(1)
            self.status_changed.emit(f"监听中 {self.host}:{self.port}")

            while self._running:
                try:
                    conn, addr = self._accept_connection(server)
                    if conn is None:
                        break
                    conn.settimeout(None)
                    with self._socket_lock:
                        self._conn = conn
                    self.status_changed.emit(f"已连接 {addr[0]}:{addr[1]}")

                    while self._running:
                        try:
                            packet = recv_packet(conn)
                        except ValueError as exc:
                            if self._running:
                                had_error = True
                                self.error_occurred.emit(f"数据包错误：{exc}")
                                self._running = False
                            break
                        except ConnectionError:
                            if self._running:
                                self.status_changed.emit("前端已断开，等待连接")
                            break
                        except OSError as exc:
                            if self._running:
                                had_error = True
                                self.error_occurred.emit(f"接收画面失败：{exc}")
                                self._running = False
                            break

                        if not self._running:
                            break

                        try:
                            frame = decode_jpeg(packet.image_bytes)
                            now = time.time()
                            latency_ms = int(now * 1000) - packet.timestamp_ms
                            detections = detector.detect(frame)
                            self.qwen_frame_ready.emit(frame.copy(), len(detections))
                            assessments, alarm = analyzer.update(detections, now_seconds=now)

                            frame_count += 1
                            elapsed = max(0.001, now - fps_started)
                            fps = frame_count / elapsed
                            overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=latency_ms)

                            if alarm.is_alarm and packet.frame_id != last_alarm_frame:
                                image_path = self.output_dir / f"alarm_{packet.frame_id}_{packet.timestamp_ms}.jpg"
                                if self._save_alarm(
                                    csv_path,
                                    packet.frame_id,
                                    packet.timestamp_ms,
                                    alarm.reason,
                                    alarm.duration_seconds,
                                    alarm.abnormal_count,
                                    alarm.abnormal_labels,
                                    image_path,
                                    overlay,
                                ):
                                    last_alarm_frame = packet.frame_id
                                else:
                                    had_error = True

                            height, width = frame.shape[:2]
                            self.frame_ready.emit(cv_frame_to_qimage(overlay))
                            self.metrics_ready.emit(fps, f"{width}x{height}", frame_count, latency_ms)
                            self.counts_ready.emit(behaviour_counts(assessments))
                            self.alarm_changed.emit(frame_status_text(alarm), alarm.is_alarm)
                        except Exception as exc:
                            self.log_ready.emit(f"帧处理失败：{exc}")
                            continue
                finally:
                    self._close_socket(conn)
                    with self._socket_lock:
                        if self._conn is conn:
                            self._conn = None
                    conn = None
        except OSError as exc:
            if self._running:
                had_error = True
                self.error_occurred.emit(f"监听失败：{exc}")
        finally:
            self._running = False
            self._close_socket(conn)
            self._close_socket(server)
            with self._socket_lock:
                if self._conn is conn:
                    self._conn = None
                if self._server is server:
                    self._server = None
            if not had_error:
                self.status_changed.emit("未监听")

    def _save_alarm(
        self,
        csv_path: Path,
        frame_id: int,
        timestamp_ms: int,
        reason: str,
        duration_seconds: float,
        abnormal_count: int,
        abnormal_labels: tuple[str, ...],
        image_path: Path,
        image,
    ) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(image_path), image)
        if not saved:
            self.error_occurred.emit(f"报警截图保存失败：{image_path}")
            return False

        try:
            append_alarm(
                csv_path,
                frame_id,
                timestamp_ms,
                reason,
                duration_seconds,
                abnormal_count,
                abnormal_labels,
                image_path,
            )
        except Exception as exc:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.error_occurred.emit(f"报警记录写入失败：{exc}")
            return False

        self.log_ready.emit(f"已保存报警：{image_path}")
        return True

    def _accept_connection(self, server: socket.socket) -> tuple[socket.socket, tuple[str, int]] | tuple[None, None]:
        while self._running:
            try:
                return server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    raise
                break
        return None, None

    @staticmethod
    def _close_socket(sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


class LocalMediaWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    qwen_frame_ready = QtCore.pyqtSignal(object, int)
    metrics_ready = QtCore.pyqtSignal(float, str, int, int)
    counts_ready = QtCore.pyqtSignal(dict)
    alarm_changed = QtCore.pyqtSignal(str, bool)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    log_ready = QtCore.pyqtSignal(str)

    def __init__(
        self,
        media_path: str | Path,
        media_type: str,
        model: str,
        alarm_seconds: float,
        output_dir: str | Path,
    ) -> None:
        super().__init__()
        self.media_path = str(media_path)
        self.media_type = media_type
        self.model = model
        self.alarm_seconds = alarm_seconds
        self.output_dir = Path(output_dir)
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        had_error = False
        try:
            self.status_changed.emit("加载模型")
            try:
                detector = YoloDetector(model_path=self.model)
            except Exception as exc:
                had_error = True
                self.error_occurred.emit(f"模型加载失败：{exc}")
                return

            analyzer = BehaviourAnalyzer(threshold_seconds=self.alarm_seconds)
            if self.media_type == "image":
                had_error = not self._run_image(detector, analyzer)
            elif self.media_type == "video":
                had_error = not self._run_video(detector, analyzer)
            else:
                had_error = True
                self.error_occurred.emit(f"不支持的测试类型：{self.media_type}")
        finally:
            self._running = False
            if not had_error:
                self.status_changed.emit("测试完成")

    def _run_image(self, detector: YoloDetector, analyzer: BehaviourAnalyzer) -> bool:
        self.status_changed.emit(f"图片测试：{Path(self.media_path).name}")
        frame = cv2.imread(self.media_path)
        if frame is None:
            self.error_occurred.emit(f"无法读取图片：{self.media_path}")
            return False
        self._process_frame(frame, detector, analyzer, frame_id=1, fps=0.0)
        return True

    def _run_video(self, detector: YoloDetector, analyzer: BehaviourAnalyzer) -> bool:
        self.status_changed.emit(f"视频测试：{Path(self.media_path).name}")
        cap = cv2.VideoCapture(self.media_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开视频：{self.media_path}")
            return False

        frame_id = 0
        started = time.time()
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / source_fps if source_fps and source_fps > 0 else 0.0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                elapsed = max(0.001, time.time() - started)
                fps = frame_id / elapsed
                self._process_frame(frame, detector, analyzer, frame_id=frame_id, fps=fps)
                if frame_delay > 0:
                    time.sleep(min(frame_delay, 0.05))
        finally:
            cap.release()
        return True

    def _process_frame(
        self,
        frame,
        detector: YoloDetector,
        analyzer: BehaviourAnalyzer,
        frame_id: int,
        fps: float,
    ) -> None:
        now = time.time()
        timestamp_ms = int(now * 1000)
        detections = detector.detect(frame)
        self.qwen_frame_ready.emit(frame.copy(), len(detections))
        assessments, alarm = analyzer.update(detections, now_seconds=now)
        overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=0)

        height, width = frame.shape[:2]
        self.frame_ready.emit(cv_frame_to_qimage(overlay))
        self.metrics_ready.emit(fps, f"{width}x{height}", frame_id, 0)
        self.counts_ready.emit(behaviour_counts(assessments))
        self.alarm_changed.emit(frame_status_text(alarm), alarm.is_alarm)


class BackendMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        _ensure_app()
        super().__init__()
        self.worker: BackendReceiverWorker | None = None
        self.local_worker: LocalMediaWorker | None = None
        self.qwen_settings = load_qwen_settings()
        self.qwen_worker: QwenWorker | None = None
        self.qwen_window = QwenAnalysisWindow(self)
        self._qwen_last_upload_at: float | None = None
        self._qwen_in_flight = False
        self._closing_after_worker = False
        self._had_error = False
        self.setWindowTitle("课堂行为监测 - 后端分析端")
        self.resize(980, 720)
        self._build_ui()
        self._set_running(False)
        self._set_testing(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QGridLayout()
        self.host_edit = QtWidgets.QLineEdit("0.0.0.0")

        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5001)

        self.model_edit = QtWidgets.QLineEdit(DEFAULT_MODEL_PATH)

        self.alarm_spin = QtWidgets.QDoubleSpinBox()
        self.alarm_spin.setRange(0.5, 30.0)
        self.alarm_spin.setDecimals(1)
        self.alarm_spin.setValue(3.0)

        self.output_edit = QtWidgets.QLineEdit("output/alarms")

        form.addWidget(QtWidgets.QLabel("监听地址"), 0, 0)
        form.addWidget(self.host_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("端口"), 0, 2)
        form.addWidget(self.port_spin, 0, 3)
        form.addWidget(QtWidgets.QLabel("模型路径"), 1, 0)
        form.addWidget(self.model_edit, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("报警秒数"), 2, 0)
        form.addWidget(self.alarm_spin, 2, 1)
        form.addWidget(QtWidgets.QLabel("输出目录"), 2, 2)
        form.addWidget(self.output_edit, 2, 3)
        root.addLayout(form)

        self.video_label = QtWidgets.QLabel("等待前端画面")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")
        root.addWidget(self.video_label, stretch=1)

        metrics = QtWidgets.QGridLayout()
        self.status_label = QtWidgets.QLabel("状态：未监听")
        self.alarm_label = QtWidgets.QLabel("报警：normal")
        self.fps_label = QtWidgets.QLabel("FPS：0.0")
        self.resolution_label = QtWidgets.QLabel("分辨率：-")
        self.frame_count_label = QtWidgets.QLabel("帧数：0")
        self.latency_label = QtWidgets.QLabel("延迟：0 ms")

        metrics.addWidget(self.status_label, 0, 0)
        metrics.addWidget(self.alarm_label, 0, 1)
        metrics.addWidget(self.fps_label, 0, 2)
        metrics.addWidget(self.resolution_label, 1, 0)
        metrics.addWidget(self.frame_count_label, 1, 1)
        metrics.addWidget(self.latency_label, 1, 2)
        root.addLayout(metrics)

        details = QtWidgets.QHBoxLayout()
        self.counts_text = QtWidgets.QTextEdit()
        self.counts_text.setReadOnly(True)
        self.counts_text.setPlaceholderText("行为计数")
        self.counts_text.setPlainText("暂无数据")

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("报警日志")

        details.addWidget(self.counts_text)
        details.addWidget(self.log_text)
        root.addLayout(details)

        buttons = QtWidgets.QHBoxLayout()
        self.image_test_button = QtWidgets.QPushButton("选择图片测试")
        self.video_test_button = QtWidgets.QPushButton("选择视频测试")
        self.stop_test_button = QtWidgets.QPushButton("停止测试")
        self.image_test_button.clicked.connect(self.open_image_test)
        self.video_test_button.clicked.connect(self.open_video_test)
        self.stop_test_button.clicked.connect(self.stop_local_test)
        buttons.addWidget(self.image_test_button)
        buttons.addWidget(self.video_test_button)
        buttons.addWidget(self.stop_test_button)
        buttons.addStretch(1)
        self.start_button = QtWidgets.QPushButton("开始监听")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.start_button.clicked.connect(self.start_backend)
        self.stop_button.clicked.connect(self.stop_backend)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)

        self.setCentralWidget(central)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.image_test_button.setEnabled(not running)
        self.video_test_button.setEnabled(not running)

    def _set_testing(self, testing: bool) -> None:
        self.image_test_button.setEnabled(not testing)
        self.video_test_button.setEnabled(not testing)
        self.stop_test_button.setEnabled(testing)
        self.start_button.setEnabled(not testing)

    def _set_stopping(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态：停止中")

    def start_backend(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.local_worker is not None and self.local_worker.isRunning():
            self.stop_local_test()
            return

        self._had_error = False
        self.worker = BackendReceiverWorker(
            host=self.host_edit.text().strip() or "0.0.0.0",
            port=self.port_spin.value(),
            model=self.model_edit.text().strip() or DEFAULT_MODEL_PATH,
            alarm_seconds=self.alarm_spin.value(),
            output_dir=self.output_edit.text().strip() or "output/alarms",
        )
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.qwen_frame_ready.connect(self._handle_qwen_frame)
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.counts_ready.connect(self._update_counts)
        self.worker.alarm_changed.connect(self._update_alarm)
        self.worker.status_changed.connect(self._update_status)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.log_ready.connect(self._append_log)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        self.qwen_settings = load_qwen_settings()
        self._qwen_last_upload_at = None
        self._qwen_in_flight = False
        self.worker.start()

    def stop_backend(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            if self.worker.isRunning():
                self._set_stopping()
                return
            self.worker = None
        self._set_running(False)

    def open_image_test(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择测试图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if file_name:
            self.start_local_test(file_name, "image")

    def open_video_test(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择测试视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if file_name:
            self.start_local_test(file_name, "video")

    def start_local_test(self, media_path: str, media_type: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.stop_backend()
            if self.worker is not None and self.worker.isRunning():
                return
        if self.local_worker is not None and self.local_worker.isRunning():
            return

        self._had_error = False
        self.qwen_settings = load_qwen_settings()
        self._qwen_last_upload_at = None
        self._qwen_in_flight = False
        self.local_worker = LocalMediaWorker(
            media_path=media_path,
            media_type=media_type,
            model=self.model_edit.text().strip() or DEFAULT_MODEL_PATH,
            alarm_seconds=self.alarm_spin.value(),
            output_dir=self.output_edit.text().strip() or "output/alarms",
        )
        self.local_worker.frame_ready.connect(self._update_frame)
        self.local_worker.qwen_frame_ready.connect(self._handle_qwen_frame)
        self.local_worker.metrics_ready.connect(self._update_metrics)
        self.local_worker.counts_ready.connect(self._update_counts)
        self.local_worker.alarm_changed.connect(self._update_alarm)
        self.local_worker.status_changed.connect(self._update_status)
        self.local_worker.error_occurred.connect(self._show_error)
        self.local_worker.log_ready.connect(self._append_log)
        self.local_worker.finished.connect(self._on_local_worker_finished)
        self._set_testing(True)
        self.local_worker.start()

    def stop_local_test(self) -> None:
        if self.local_worker is not None:
            self.local_worker.stop()
            if self.local_worker.isRunning():
                self.stop_test_button.setEnabled(False)
                self.status_label.setText("状态：停止测试中")
                return
            self.local_worker = None
        self._set_testing(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._closing_after_worker = True
            self.stop_backend()
            event.ignore()
            return
        if self.local_worker is not None and self.local_worker.isRunning():
            self._closing_after_worker = True
            self.stop_local_test()
            event.ignore()
            return
        self._wait_for_qwen_worker()
        super().closeEvent(event)

    def _handle_qwen_frame(self, frame, target_count: int) -> None:
        now = time.monotonic()
        if not should_upload_frame(
            now=now,
            last_upload_at=self._qwen_last_upload_at,
            interval_seconds=self.qwen_settings.interval_seconds,
            in_flight=self._qwen_in_flight,
        ):
            return

        self._qwen_last_upload_at = now
        if not should_use_qwen_for_scene(target_count, self.qwen_settings.max_yolo_targets):
            prepared = prepare_frame_for_qwen(frame, self.qwen_settings)
            message = (
                f"当前 YOLO 检测到 {target_count} 个目标，超过千问少人场景阈值 "
                f"{self.qwen_settings.max_yolo_targets}。\n"
                "已跳过千问坐标检测，请以主窗口 YOLO 标注为准。\n"
                "如需调整阈值，可设置环境变量 QWEN_MAX_YOLO_TARGETS。"
            )
            self.qwen_window.show_skipped(prepared, message)
            self._append_log("千问分析已跳过：当前目标数量较多")
            return

        if self.qwen_worker is not None and self.qwen_worker.isRunning():
            return

        self._qwen_in_flight = True
        self.qwen_window.show_pending()
        self.qwen_worker = QwenWorker(frame, self.qwen_settings, self)
        self.qwen_worker.succeeded.connect(self._show_qwen_result)
        self.qwen_worker.failed.connect(self._show_qwen_error)
        self.qwen_worker.finished.connect(self._on_qwen_worker_finished)
        self.qwen_worker.start()

    def _show_qwen_result(self, frame, result) -> None:
        self._qwen_in_flight = False
        self.qwen_window.show_result(frame, result)
        self._append_log("千问分析完成")

    def _show_qwen_error(self, frame, message: str) -> None:
        self._qwen_in_flight = False
        self.qwen_window.show_error(frame, message)
        self._append_log(f"千问分析失败：{message}")

    def _on_qwen_worker_finished(self) -> None:
        self._qwen_in_flight = False
        self.qwen_worker = None

    def _wait_for_qwen_worker(self) -> None:
        if self.qwen_worker is not None and self.qwen_worker.isRunning():
            self.qwen_worker.wait(1000)

    def _update_frame(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _update_metrics(self, fps: float, resolution: str, frame_count: int, latency_ms: int) -> None:
        self.fps_label.setText(f"FPS：{fps:.1f}")
        self.resolution_label.setText(f"分辨率：{resolution}")
        self.frame_count_label.setText(f"帧数：{frame_count}")
        self.latency_label.setText(f"延迟：{latency_ms} ms")

    def _update_counts(self, counts: dict[str, int]) -> None:
        self.counts_text.setPlainText("\n".join(f"{label}：{count}" for label, count in counts.items()))

    def _update_alarm(self, status: str, is_alarm: bool) -> None:
        self.alarm_label.setText(f"报警：{status}")
        if is_alarm:
            self.alarm_label.setStyleSheet("color: #b00020; font-weight: bold;")
        else:
            self.alarm_label.setStyleSheet("color: #166534;")

    def _update_status(self, status: str) -> None:
        self.status_label.setText(f"状态：{status}")

    def _show_error(self, message: str) -> None:
        self._had_error = True
        self.status_label.setText(f"状态：错误 - {message}")
        self._append_log(f"错误：{message}")

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _on_worker_finished(self) -> None:
        self._set_running(False)
        self.worker = None
        if not self._had_error and "停止中" in self.status_label.text():
            self.status_label.setText("状态：未监听")
        if self._closing_after_worker:
            self._closing_after_worker = False
            self.close()

    def _on_local_worker_finished(self) -> None:
        self._set_testing(False)
        self.local_worker = None
        if self._closing_after_worker:
            self._closing_after_worker = False
            self.close()


def main() -> None:
    app = QtWidgets.QApplication([])
    window = BackendMonitorWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
