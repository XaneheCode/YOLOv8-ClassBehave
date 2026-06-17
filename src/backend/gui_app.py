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
from src.backend.behaviour_analyzer import display_label
from src.backend.detector import YoloDetector
from src.backend.person_crop_grid import build_person_crop_grid
from src.backend.qwen_analysis import (
    QwenAnalysisError,
    call_person_crop_vision,
    call_qwen_vision,
    load_qwen_settings,
    prepare_frame_for_qwen,
    should_upload_frame,
    should_use_qwen_for_scene,
)
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet
from src.common.qt_dashboard_theme import (
    apply_dashboard_style,
    make_metric_card,
    make_panel,
    set_button_role,
)
from src.common.types import Detection


_APP: QtWidgets.QApplication | None = None
INFERENCE_MODE_PERSON_VLM = "person_vlm"
INFERENCE_MODE_BEHAVIOUR_YOLO = "behaviour_yolo"
DEFAULT_PERSON_MODEL_PATH = "yolov8s.pt"
QWEN_ERROR_COOLDOWN_SECONDS = 30
QWEN_PERSON_COLORS = {
    "Hand-raise": (37, 99, 235),
    "Reading": (22, 163, 74),
    "Writing": (22, 163, 74),
    "Useing-Phone": (220, 38, 38),
    "Head-down": (234, 88, 12),
    "Sleeping": (147, 51, 234),
}


def qwen_person_color(label: str) -> QtGui.QColor:
    red, green, blue = QWEN_PERSON_COLORS.get(label, (255, 0, 255))
    return QtGui.QColor(red, green, blue)


def format_qwen_result_details(result) -> str:
    info = f"大模型概括：{result.summary or '无'}\n\n"
    for index, person in enumerate(result.people, start=1):
        info += (
            f"{index}. 类别：{display_label(person.label)}，坐标 {person.bbox}，"
            f"状态：{person.status}，置信度：{person.confidence}\n"
        )
    if not result.people:
        info += "未识别到人物框。\n"
    return info


def _qwen_confidence_score(value: str) -> float:
    return {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.45,
        "unknown": 0.4,
    }.get(str(value or "").strip().lower(), 0.4)


def qwen_result_to_detections(result) -> list[Detection]:
    detections = []
    for person in result.people:
        if len(person.bbox) != 4:
            continue
        detections.append(
            Detection(
                label=person.label,
                confidence=_qwen_confidence_score(person.confidence),
                bbox=tuple(person.bbox),
            )
        )
    return detections


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
        self.setWindowTitle("大模型分析结果")
        self.resize(980, 760)

        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(920, 500)
        self.image_label.setStyleSheet("background: #111; color: #ddd;")
        layout.addWidget(self.image_label, stretch=4)

        bottom_layout = QtWidgets.QHBoxLayout()

        self.result_group = QtWidgets.QGroupBox("分析结果", self)
        result_layout = QtWidgets.QVBoxLayout(self.result_group)
        self.result_browser = QtWidgets.QTextBrowser(self.result_group)
        self.result_browser.setMinimumHeight(180)
        self.result_browser.setText("等待分析结果")
        result_layout.addWidget(self.result_browser)
        bottom_layout.addWidget(self.result_group, stretch=3)

        self.status_group = QtWidgets.QGroupBox("大模型状态", self)
        status_layout = QtWidgets.QVBoxLayout(self.status_group)
        self.status_browser = QtWidgets.QTextBrowser(self.status_group)
        self.status_browser.setMinimumHeight(180)
        self.status_browser.setText("等待后端画面")
        status_layout.addWidget(self.status_browser)
        bottom_layout.addWidget(self.status_group, stretch=2)

        self.text_browser = self.result_browser
        layout.addLayout(bottom_layout, stretch=1)

    def show_pending(self) -> None:
        self.status_browser.setText("正在上传当前画面到大模型分析...")
        if not self.result_browser.toPlainText().strip():
            self.result_browser.setText("等待分析结果")
        self.show()

    def show_error(self, frame, message: str) -> None:
        if frame is not None:
            self.image_label.setPixmap(self._scaled_pixmap(self._frame_to_pixmap(frame)))
        self.status_browser.setText(f"大模型分析失败：{message}")
        self.show()

    def show_skipped(self, frame, message: str) -> None:
        if frame is not None:
            self.image_label.setPixmap(self._scaled_pixmap(self._frame_to_pixmap(frame)))
        self.status_browser.setText(message)
        self.show()

    def show_result(self, frame, result) -> None:
        pixmap = self._frame_to_pixmap(frame)
        painter = QtGui.QPainter(pixmap)
        painter.setFont(QtGui.QFont("Microsoft YaHei", 13, QtGui.QFont.Weight.Bold))

        for index, person in enumerate(result.people, start=1):
            x1, y1, x2, y2 = person.bbox
            color = qwen_person_color(person.label)
            rect = QtCore.QRect(x1, y1, x2 - x1, y2 - y1)
            painter.setPen(QtGui.QPen(QtGui.QColor(10, 10, 10), 7))
            painter.drawRect(rect)
            pen = QtGui.QPen(color, 5)
            painter.setPen(pen)
            painter.drawRect(rect)
            label = str(index)
            label_width = max(34, 16 + len(label) * 12)
            label_rect = QtCore.QRect(x1, max(0, y1 - 30), label_width, 28)
            badge_color = QtGui.QColor(color)
            badge_color.setAlpha(255)
            painter.fillRect(label_rect, badge_color)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            painter.drawText(
                label_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                label,
            )
            painter.setPen(pen)
        painter.end()

        self.image_label.setPixmap(self._scaled_pixmap(pixmap))
        self.result_browser.setText(format_qwen_result_details(result))
        self.status_browser.setText(f"分析完成：识别到 {len(result.people)} 个大模型目标")
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

    def __init__(
        self,
        frame,
        settings,
        parent=None,
        detections: list[Detection] | None = None,
        mode: str = INFERENCE_MODE_BEHAVIOUR_YOLO,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.display_frame = frame.copy()
        self.detections = detections or []
        self.mode = mode
        if mode == INFERENCE_MODE_PERSON_VLM:
            self.crop_grid = build_person_crop_grid(
                frame,
                self.detections,
                max_people=settings.max_yolo_targets,
            )
            self.analysis_frame = self.crop_grid.image
        else:
            self.crop_grid = None
            self.analysis_frame = prepare_frame_for_qwen(frame, settings)
            self.display_frame = self.analysis_frame

    def run(self) -> None:
        try:
            if self.mode == INFERENCE_MODE_PERSON_VLM and self.crop_grid is not None:
                result = call_person_crop_vision(self.analysis_frame, self.crop_grid.source_by_id, self.settings)
            else:
                result = call_qwen_vision(self.analysis_frame, self.settings)
        except QwenAnalysisError as exc:
            self.failed.emit(self.display_frame, str(exc))
        except Exception as exc:
            self.failed.emit(self.display_frame, f"未知错误：{exc}")
        else:
            self.succeeded.emit(self.display_frame, result)


class BackendReceiverWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    qwen_frame_ready = QtCore.pyqtSignal(object, object)
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
        inference_mode: str = INFERENCE_MODE_BEHAVIOUR_YOLO,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.alarm_seconds = alarm_seconds
        self.output_dir = Path(output_dir)
        self.inference_mode = inference_mode
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
                if self.inference_mode == INFERENCE_MODE_PERSON_VLM:
                    detector = YoloDetector(model_path=self.model, allowed_labels={"person"})
                else:
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
                            self.qwen_frame_ready.emit(frame.copy(), detections)
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
    qwen_frame_ready = QtCore.pyqtSignal(object, object)
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
        inference_mode: str = INFERENCE_MODE_BEHAVIOUR_YOLO,
    ) -> None:
        super().__init__()
        self.media_path = str(media_path)
        self.media_type = media_type
        self.model = model
        self.alarm_seconds = alarm_seconds
        self.output_dir = Path(output_dir)
        self.inference_mode = inference_mode
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        had_error = False
        try:
            self.status_changed.emit("加载模型")
            try:
                if self.inference_mode == INFERENCE_MODE_PERSON_VLM:
                    detector = YoloDetector(model_path=self.model, allowed_labels={"person"})
                else:
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
        self.qwen_frame_ready.emit(frame.copy(), detections)
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
        self._qwen_cooldown_until = 0.0
        self._closing_after_worker = False
        self._had_error = False
        self.setWindowTitle("课堂行为监测 - 后端分析端")
        self.resize(980, 720)
        self._build_ui()
        self._set_running(False)
        self._set_testing(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        central.setObjectName("qtDashboardShell")
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = QtWidgets.QScrollArea(central)
        self.page_scroll.setObjectName("dashboardScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root_layout.addWidget(self.page_scroll)

        page = QtWidgets.QWidget(self.page_scroll)
        main_layout = QtWidgets.QVBoxLayout(page)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        topbar = QtWidgets.QHBoxLayout()
        topbar.setSpacing(12)
        brand_mark = QtWidgets.QLabel("CB")
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(42, 42)
        brand_mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        topbar.addWidget(brand_mark)
        title_block = QtWidgets.QVBoxLayout()
        title_block.setSpacing(1)
        eyebrow = QtWidgets.QLabel("双机课堂行为远程监测")
        eyebrow.setObjectName("mutedText")
        page_title = QtWidgets.QLabel("后端分析端")
        page_title.setObjectName("pageTitle")
        title_block.addWidget(eyebrow)
        title_block.addWidget(page_title)
        topbar.addLayout(title_block)
        topbar.addStretch(1)
        self.protocol_badge = QtWidgets.QLabel("NSGD TCP 监听端")
        self.protocol_badge.setObjectName("protocolBadge")
        topbar.addWidget(self.protocol_badge)
        main_layout.addLayout(topbar)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        for column in (1, 3, 5):
            form.setColumnStretch(column, 1)
        self.host_edit = QtWidgets.QLineEdit("0.0.0.0")

        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5001)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("人体YOLO+大模型", INFERENCE_MODE_PERSON_VLM)
        self.mode_combo.addItem("六类YOLO", INFERENCE_MODE_BEHAVIOUR_YOLO)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.model_edit = QtWidgets.QLineEdit(DEFAULT_PERSON_MODEL_PATH)

        self.vision_model_combo = QtWidgets.QComboBox()
        openai_settings = load_qwen_settings(provider="openai")
        dashscope_settings = load_qwen_settings(provider="dashscope")
        openai_label = "GPT-5.5（OpenAI兼容）" if openai_settings.model == "gpt-5.5" else f"GPT（{openai_settings.model}）"
        self.vision_model_combo.addItem(openai_label, "openai")
        self.vision_model_combo.addItem(f"千问（{dashscope_settings.model}）", "dashscope")
        selected_provider_index = self.vision_model_combo.findData(self.qwen_settings.provider)
        if selected_provider_index >= 0:
            self.vision_model_combo.setCurrentIndex(selected_provider_index)

        self.vision_target_spin = QtWidgets.QSpinBox()
        self.vision_target_spin.setRange(1, 200)
        self.vision_target_spin.setValue(self.qwen_settings.max_yolo_targets)

        self.alarm_spin = QtWidgets.QDoubleSpinBox()
        self.alarm_spin.setRange(0.5, 30.0)
        self.alarm_spin.setDecimals(1)
        self.alarm_spin.setValue(3.0)

        self.output_edit = QtWidgets.QLineEdit("output/alarms")

        form.addWidget(QtWidgets.QLabel("监听地址"), 0, 0)
        form.addWidget(self.host_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("端口"), 0, 2)
        form.addWidget(self.port_spin, 0, 3)
        form.addWidget(QtWidgets.QLabel("识别模式"), 0, 4)
        form.addWidget(self.mode_combo, 0, 5)
        form.addWidget(QtWidgets.QLabel("模型路径"), 1, 0)
        form.addWidget(self.model_edit, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("输出目录"), 1, 4)
        form.addWidget(self.output_edit, 1, 5)
        form.addWidget(QtWidgets.QLabel("大模型"), 2, 0)
        form.addWidget(self.vision_model_combo, 2, 1)
        form.addWidget(QtWidgets.QLabel("目标上限"), 2, 2)
        form.addWidget(self.vision_target_spin, 2, 3)
        form.addWidget(QtWidgets.QLabel("报警秒数"), 2, 4)
        form.addWidget(self.alarm_spin, 2, 5)

        control_panel, control_layout = make_panel(
            "监听与模型",
            "后端保持 TCP socket 监听，按自定义 NSGD 帧包接收前端 JPEG 画面。",
        )
        control_panel.setMaximumHeight(245)
        control_layout.addLayout(form)

        action_buttons = QtWidgets.QHBoxLayout()
        action_buttons.setSpacing(8)
        self.image_test_button = QtWidgets.QPushButton("选择图片测试")
        self.video_test_button = QtWidgets.QPushButton("选择视频测试")
        self.stop_test_button = QtWidgets.QPushButton("停止测试")
        set_button_role(self.image_test_button, "soft")
        set_button_role(self.video_test_button, "soft")
        set_button_role(self.stop_test_button, "danger")
        self.image_test_button.clicked.connect(self.open_image_test)
        self.video_test_button.clicked.connect(self.open_video_test)
        self.stop_test_button.clicked.connect(self.stop_local_test)
        self.start_button = QtWidgets.QPushButton("开始监听")
        self.stop_button = QtWidgets.QPushButton("停止")
        set_button_role(self.start_button, "primary")
        set_button_role(self.stop_button, "danger")
        self.start_button.clicked.connect(self.start_backend)
        self.stop_button.clicked.connect(self.stop_backend)
        action_buttons.addWidget(self.image_test_button)
        action_buttons.addWidget(self.video_test_button)
        action_buttons.addWidget(self.stop_test_button)
        action_buttons.addStretch(1)
        action_buttons.addWidget(self.start_button)
        action_buttons.addWidget(self.stop_button)
        control_layout.addLayout(action_buttons)
        main_layout.addWidget(control_panel)

        monitor_row = QtWidgets.QHBoxLayout()
        monitor_row.setSpacing(10)

        monitor_panel, monitor_layout = make_panel(
            "分析画面",
            "前端 TCP 画面、YOLO 人体框、大模型回填分类结果会在这里展示。",
        )
        stage = QtWidgets.QFrame()
        stage.setObjectName("darkStage")
        stage_layout = QtWidgets.QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QtWidgets.QLabel("等待前端画面")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 240)
        self.video_label.setStyleSheet("background: transparent; color: #d4d7d2;")
        stage_layout.addWidget(self.video_label)
        monitor_layout.addWidget(stage, stretch=1)
        monitor_row.addWidget(monitor_panel, stretch=1)

        insight_panel, insight_layout = make_panel(
            "运行状态",
            "双机连接、检测性能、报警和行为计数。",
        )

        self.status_label = QtWidgets.QLabel("状态：未监听")
        self.alarm_label = QtWidgets.QLabel("报警：normal")
        self.fps_label = QtWidgets.QLabel("FPS：0.0")
        self.resolution_label = QtWidgets.QLabel("分辨率：-")
        self.frame_count_label = QtWidgets.QLabel("帧数：0")
        self.latency_label = QtWidgets.QLabel("延迟：0 ms")

        metric_grid = QtWidgets.QGridLayout()
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)
        metric_grid.addWidget(make_metric_card("连接状态", self.status_label), 0, 0)
        metric_grid.addWidget(make_metric_card("报警状态", self.alarm_label), 0, 1)
        metric_grid.addWidget(make_metric_card("FPS", self.fps_label), 1, 0)
        metric_grid.addWidget(make_metric_card("延迟", self.latency_label), 1, 1)
        metric_grid.addWidget(make_metric_card("分辨率", self.resolution_label), 2, 0)
        metric_grid.addWidget(make_metric_card("帧数", self.frame_count_label), 2, 1)
        insight_layout.addLayout(metric_grid)

        self.counts_text = QtWidgets.QTextEdit()
        self.counts_text.setReadOnly(True)
        self.counts_text.setPlaceholderText("行为计数")
        self.counts_text.setPlainText("暂无数据")
        insight_layout.addWidget(self.counts_text)
        monitor_row.addWidget(insight_panel)
        main_layout.addLayout(monitor_row, stretch=1)

        log_panel, log_layout = make_panel(
            "日志设置",
            "显示连接、检测、大模型和报警保存状态。",
        )
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("报警日志")
        self.log_text.setMaximumHeight(86)
        log_layout.addWidget(self.log_text)
        log_panel.setMaximumHeight(140)
        main_layout.addWidget(log_panel)

        self.page_scroll.setWidget(page)
        self.setCentralWidget(central)
        apply_dashboard_style(self)

    def _current_inference_mode(self) -> str:
        return str(self.mode_combo.currentData() or INFERENCE_MODE_PERSON_VLM)

    def _selected_qwen_settings(self):
        provider = str(self.vision_model_combo.currentData() or self.qwen_settings.provider)
        settings = load_qwen_settings(provider=provider)
        settings.max_yolo_targets = self.vision_target_spin.value()
        return settings

    def _on_mode_changed(self) -> None:
        if self._current_inference_mode() == INFERENCE_MODE_PERSON_VLM:
            self.model_edit.setText(DEFAULT_PERSON_MODEL_PATH)
        else:
            self.model_edit.setText(DEFAULT_MODEL_PATH)

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
            inference_mode=self._current_inference_mode(),
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
        self.qwen_settings = self._selected_qwen_settings()
        self._qwen_last_upload_at = None
        self._qwen_in_flight = False
        self._qwen_cooldown_until = 0.0
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
        self.qwen_settings = self._selected_qwen_settings()
        self._qwen_last_upload_at = None
        self._qwen_in_flight = False
        self._qwen_cooldown_until = 0.0
        self.local_worker = LocalMediaWorker(
            media_path=media_path,
            media_type=media_type,
            model=self.model_edit.text().strip() or DEFAULT_MODEL_PATH,
            alarm_seconds=self.alarm_spin.value(),
            output_dir=self.output_edit.text().strip() or "output/alarms",
            inference_mode=self._current_inference_mode(),
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

    def _handle_qwen_frame(self, frame, detections_or_count=None, *, target_count: int | None = None) -> None:
        now = time.monotonic()
        if now < self._qwen_cooldown_until:
            return

        if target_count is not None:
            detections = []
        elif isinstance(detections_or_count, int):
            detections: list[Detection] = []
            target_count = detections_or_count
        else:
            detections = list(detections_or_count or [])
            target_count = len(detections)

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
                f"当前 YOLO 检测到 {target_count} 个目标，超过大模型目标上限 "
                f"{self.qwen_settings.max_yolo_targets}。\n"
                "已跳过大模型分析，请以主窗口 YOLO 标注为准。\n"
                "如需调整阈值，可修改后端界面的“目标上限”。"
            )
            self.qwen_window.show_skipped(prepared, message)
            self._append_log("大模型分析已跳过：当前目标数量较多")
            return

        if self.qwen_worker is not None and self.qwen_worker.isRunning():
            return

        self._qwen_in_flight = True
        self.qwen_window.show_pending()
        self.qwen_worker = QwenWorker(
            frame,
            self.qwen_settings,
            self,
            detections=detections,
            mode=self._current_inference_mode(),
        )
        self.qwen_worker.succeeded.connect(self._show_qwen_result)
        self.qwen_worker.failed.connect(self._show_qwen_error)
        self.qwen_worker.finished.connect(self._on_qwen_worker_finished)
        self.qwen_worker.start()

    def _show_qwen_result(self, frame, result) -> None:
        self._qwen_in_flight = False
        self._qwen_cooldown_until = 0.0
        self.qwen_window.show_result(frame, result)
        if self._current_inference_mode() == INFERENCE_MODE_PERSON_VLM:
            detections = qwen_result_to_detections(result)
            analyzer = BehaviourAnalyzer(threshold_seconds=self.alarm_spin.value())
            assessments, alarm = analyzer.update(detections, now_seconds=time.time())
            overlay = draw_overlay(frame, assessments, alarm, fps=0.0, latency_ms=0)
            self._update_frame(cv_frame_to_qimage(overlay))
            self._update_counts(behaviour_counts(assessments))
            self._update_alarm(frame_status_text(alarm), alarm.is_alarm)
        self._append_log("大模型分析完成")

    def _show_qwen_error(self, frame, message: str) -> None:
        self._qwen_in_flight = False
        self._qwen_cooldown_until = time.monotonic() + QWEN_ERROR_COOLDOWN_SECONDS
        self.qwen_window.show_error(frame, message)
        self._append_log(f"千问分析失败：{message}")

    def _on_qwen_worker_finished(self) -> None:
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
