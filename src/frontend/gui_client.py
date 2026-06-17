from __future__ import annotations

import socket
import time

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from src.common.image_codec import encode_jpeg, resize_to_width
from src.common.protocol import send_packet
from src.common.qt_dashboard_theme import (
    apply_dashboard_style,
    make_metric_card,
    make_panel,
    make_sidebar,
    set_button_role,
)


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


class CameraSenderWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    metrics_ready = QtCore.pyqtSignal(float, str, int, int)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        host: str,
        port: int,
        camera: int,
        width: int,
        fps: float,
        quality: int,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.camera = camera
        self.width = width
        self.fps = fps
        self.quality = quality
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        had_error = False
        delay_seconds = 1.0 / max(0.1, self.fps)
        cap = cv2.VideoCapture(self.camera)
        try:
            if not cap.isOpened():
                had_error = True
                self.error_occurred.emit(f"无法打开摄像头 {self.camera}")
                return

            frame_id = 0
            started_at = time.time()
            self.status_changed.emit("连接中")
            with socket.create_connection((self.host, self.port), timeout=10) as conn:
                self.status_changed.emit("已连接")
                while self._running:
                    loop_started = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        had_error = True
                        self.error_occurred.emit("读取摄像头画面失败")
                        break

                    frame = resize_to_width(frame, self.width)
                    image_bytes = encode_jpeg(frame, quality=self.quality)
                    timestamp_ms = int(time.time() * 1000)
                    try:
                        send_packet(
                            conn,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                            image_bytes=image_bytes,
                        )
                    except OSError as exc:
                        had_error = True
                        self.error_occurred.emit(f"发送画面失败：{exc}")
                        break

                    self.frame_ready.emit(cv_frame_to_qimage(frame))

                    elapsed = max(0.001, time.time() - started_at)
                    current_fps = (frame_id + 1) / elapsed
                    height, width = frame.shape[:2]
                    self.metrics_ready.emit(
                        current_fps,
                        f"{width}x{height}",
                        frame_id + 1,
                        len(image_bytes),
                    )
                    frame_id += 1

                    sleep_for = delay_seconds - (time.time() - loop_started)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        except OSError as exc:
            had_error = True
            self.error_occurred.emit(f"连接失败：{exc}")
        except Exception as exc:
            had_error = True
            self.error_occurred.emit(f"发送端错误：{exc}")
        finally:
            cap.release()
            self._running = False
            if not had_error:
                self.status_changed.emit("已断开")


class CameraClientWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        _ensure_app()
        super().__init__()
        self.worker: CameraSenderWorker | None = None
        self._closing_after_worker = False
        self.setWindowTitle("课堂行为监测 - 前端发送端")
        self.resize(900, 620)
        self._build_ui()
        self._set_running(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        central.setObjectName("qtDashboardShell")
        shell = QtWidgets.QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.sidebar = make_sidebar(
            title="ClassBehave",
            subtitle="TCP Sender",
            active_label="▧ 发送端",
            secondary_labels=["▣ 实时分析", "◇ 大模型", "◎ 日志设置"],
            note="运行链路\n摄像头采集 -> NSGD TCP 帧包 -> 后端监听",
        )
        shell.addWidget(self.sidebar)

        main = QtWidgets.QWidget(central)
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)
        shell.addWidget(main, stretch=1)

        topbar = QtWidgets.QHBoxLayout()
        title_block = QtWidgets.QVBoxLayout()
        eyebrow = QtWidgets.QLabel("双机课堂行为远程监测")
        eyebrow.setObjectName("mutedText")
        page_title = QtWidgets.QLabel("前端发送端")
        page_title.setObjectName("pageTitle")
        title_block.addWidget(eyebrow)
        title_block.addWidget(page_title)
        topbar.addLayout(title_block)
        topbar.addStretch(1)
        self.protocol_badge = QtWidgets.QLabel("NSGD TCP 帧包发送端")
        self.protocol_badge.setObjectName("protocolBadge")
        topbar.addWidget(self.protocol_badge)
        main_layout.addLayout(topbar)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(18)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")

        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5001)

        self.camera_spin = QtWidgets.QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(0)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(160, 1920)
        self.width_spin.setValue(640)

        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(8)

        self.quality_spin = QtWidgets.QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(80)

        form.addWidget(QtWidgets.QLabel("后端 IP"), 0, 0)
        form.addWidget(self.host_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("端口"), 0, 2)
        form.addWidget(self.port_spin, 0, 3)
        form.addWidget(QtWidgets.QLabel("摄像头"), 1, 0)
        form.addWidget(self.camera_spin, 1, 1)
        form.addWidget(QtWidgets.QLabel("宽度"), 1, 2)
        form.addWidget(self.width_spin, 1, 3)
        form.addWidget(QtWidgets.QLabel("FPS"), 2, 0)
        form.addWidget(self.fps_spin, 2, 1)
        form.addWidget(QtWidgets.QLabel("JPEG 质量"), 2, 2)
        form.addWidget(self.quality_spin, 2, 3)
        control_panel, control_layout = make_panel(
            "发送控制",
            "选择摄像头并设置 JPEG 帧参数，按自定义 TCP 协议发送到后端。",
        )
        control_layout.addLayout(form)

        self.start_button = QtWidgets.QPushButton("开始发送")
        self.stop_button = QtWidgets.QPushButton("停止")
        set_button_role(self.start_button, "primary")
        set_button_role(self.stop_button, "danger")
        self.start_button.clicked.connect(self.start_sender)
        self.stop_button.clicked.connect(self.stop_sender)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        control_layout.addLayout(buttons)
        control_layout.addStretch(1)

        preview_panel, preview_layout = make_panel(
            "画面源",
            "本地摄像头预览。实际传输仍使用 OpenCV 编码和 NSGD TCP 帧包。",
        )
        stage = QtWidgets.QFrame()
        stage.setObjectName("darkStage")
        stage_layout = QtWidgets.QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QtWidgets.QLabel("等待摄像头画面")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background: transparent; color: #d4d7d2;")
        stage_layout.addWidget(self.preview_label)
        preview_layout.addWidget(stage, stretch=1)

        content.addWidget(preview_panel, stretch=1)
        content.addWidget(control_panel)
        main_layout.addLayout(content, stretch=1)

        self.status_label = QtWidgets.QLabel("状态：未连接")
        self.fps_label = QtWidgets.QLabel("FPS：0.0")
        self.resolution_label = QtWidgets.QLabel("分辨率：-")
        self.frame_count_label = QtWidgets.QLabel("帧数：0")
        self.jpeg_size_label = QtWidgets.QLabel("JPEG：0 KB")

        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(10)
        metrics.addWidget(make_metric_card("连接状态", self.status_label))
        metrics.addWidget(make_metric_card("发送 FPS", self.fps_label))
        metrics.addWidget(make_metric_card("分辨率", self.resolution_label))
        metrics.addWidget(make_metric_card("帧数", self.frame_count_label))
        metrics.addWidget(make_metric_card("JPEG 大小", self.jpeg_size_label))
        main_layout.addLayout(metrics)

        self.setCentralWidget(central)
        apply_dashboard_style(self)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _set_stopping(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态：停止中")

    def start_sender(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = CameraSenderWorker(
            host=self.host_edit.text().strip() or "127.0.0.1",
            port=self.port_spin.value(),
            camera=self.camera_spin.value(),
            width=self.width_spin.value(),
            fps=self.fps_spin.value(),
            quality=self.quality_spin.value(),
        )
        self.worker.frame_ready.connect(self._update_preview)
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.status_changed.connect(self._update_status)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        self.worker.start()

    def stop_sender(self) -> None:
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
            self.stop_sender()
            event.ignore()
            return
        super().closeEvent(event)

    def _update_preview(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _update_metrics(
        self,
        fps: float,
        resolution: str,
        frame_count: int,
        jpeg_size: int,
    ) -> None:
        self.fps_label.setText(f"FPS：{fps:.1f}")
        self.resolution_label.setText(f"分辨率：{resolution}")
        self.frame_count_label.setText(f"帧数：{frame_count}")
        self.jpeg_size_label.setText(f"JPEG：{jpeg_size / 1024:.1f} KB")

    def _update_status(self, status: str) -> None:
        self.status_label.setText(f"状态：{status}")

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"状态：错误 - {message}")

    def _on_worker_finished(self) -> None:
        self._set_running(False)
        self.worker = None
        if self._closing_after_worker:
            self._closing_after_worker = False
            self.close()


def main() -> None:
    app = QtWidgets.QApplication([])
    window = CameraClientWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
