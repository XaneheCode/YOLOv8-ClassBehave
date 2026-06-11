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


class BackendReceiverWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
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

            conn, addr = self._accept_connection(server)
            if conn is None:
                return
            conn.settimeout(None)
            with self._socket_lock:
                self._conn = conn
            self.status_changed.emit(f"已连接 {addr[0]}:{addr[1]}")

            while self._running:
                try:
                    packet = recv_packet(conn)
                except (ConnectionError, OSError) as exc:
                    if self._running:
                        had_error = True
                        self.error_occurred.emit(f"接收画面失败：{exc}")
                    break

                try:
                    frame = decode_jpeg(packet.image_bytes)
                    now = time.time()
                    latency_ms = int(now * 1000) - packet.timestamp_ms
                    detections = detector.detect(frame)
                    assessments, alarm = analyzer.update(detections, now_seconds=now)

                    frame_count += 1
                    elapsed = max(0.001, now - fps_started)
                    fps = frame_count / elapsed
                    overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=latency_ms)

                    if alarm.is_alarm and packet.frame_id != last_alarm_frame:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        image_path = self.output_dir / f"alarm_{packet.frame_id}_{packet.timestamp_ms}.jpg"
                        cv2.imwrite(str(image_path), overlay)
                        append_alarm(
                            csv_path,
                            packet.frame_id,
                            packet.timestamp_ms,
                            alarm.reason,
                            alarm.duration_seconds,
                            alarm.abnormal_count,
                            alarm.abnormal_labels,
                            image_path,
                        )
                        last_alarm_frame = packet.frame_id
                        self.log_ready.emit(f"已保存报警：{image_path}")

                    height, width = frame.shape[:2]
                    self.frame_ready.emit(cv_frame_to_qimage(overlay))
                    self.metrics_ready.emit(fps, f"{width}x{height}", frame_count, latency_ms)
                    self.counts_ready.emit(behaviour_counts(assessments))
                    self.alarm_changed.emit(frame_status_text(alarm), alarm.is_alarm)
                except Exception as exc:
                    had_error = True
                    self.error_occurred.emit(f"处理画面失败：{exc}")
                    break
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


class BackendMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        _ensure_app()
        super().__init__()
        self.worker: BackendReceiverWorker | None = None
        self._closing_after_worker = False
        self._had_error = False
        self.setWindowTitle("课堂行为监测 - 后端分析端")
        self.resize(980, 720)
        self._build_ui()
        self._set_running(False)

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

    def _set_stopping(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态：停止中")

    def start_backend(self) -> None:
        if self.worker is not None and self.worker.isRunning():
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
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.counts_ready.connect(self._update_counts)
        self.worker.alarm_changed.connect(self._update_alarm)
        self.worker.status_changed.connect(self._update_status)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.log_ready.connect(self._append_log)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        self.worker.start()

    def stop_backend(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            if self.worker.isRunning():
                self._set_stopping()
                return
            self.worker = None
        self._set_running(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._closing_after_worker = True
            self.stop_backend()
            event.ignore()
            return
        super().closeEvent(event)

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


def main() -> None:
    app = QtWidgets.QApplication([])
    window = BackendMonitorWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
