# YOLOv8 PyQt Dual GUI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-laptop PyQt6 classroom behaviour monitoring system that uses the `D:\Documents\YOLOv8` six-class model while preserving the existing TCP image transfer and target-level abnormal highlighting.

**Architecture:** Keep the existing protocol, image codec, detector adapter, and alarm CSV flow. Add focused GUI entrypoints for the frontend camera sender and backend monitor, and update behaviour analysis to classify the six YOLOv8 labels independently per detection.

**Tech Stack:** Python 3.12, PyQt6, OpenCV, Ultralytics YOLOv8, TCP sockets, pytest, PowerShell.

---

## File Structure

- `models/classroom_behaviour_6cls.pt`: local runtime copy of `D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt`. This file is ignored by git because `*.pt` is ignored.
- `src/backend/behaviour_analyzer.py`: six-class normal/abnormal label rules, Chinese display names, per-target assessment, alarm duration.
- `src/backend/app.py`: command-line backend defaults, reusable overlay/status helpers, alarm CSV writer.
- `src/backend/gui_app.py`: PyQt6 backend monitor window and receiver worker.
- `src/frontend/gui_client.py`: PyQt6 frontend sender window and camera sender worker.
- `requirements.txt`: add PyQt6 for both GUI entrypoints.
- `START_BACKEND_GUI.ps1`: launch backend monitor GUI.
- `START_FRONTEND_GUI.ps1`: launch frontend sender GUI.
- `scripts/package_backend.ps1`: package backend GUI, CLI backend, common code, dependencies, and six-class model.
- `README.md`: make GUI workflow the primary course demonstration path.
- `docs/course-evidence/checklist.md`: update evidence checklist for dual GUI and target-level red boxes.
- `docs/course-evidence/yolov8-6cls-offline-test.md`: record six-class model offline test command and outputs.
- `tests/test_behaviour_analyzer.py`: six-class label mapping and target-level assessment tests.
- `tests/test_backend_app.py`: default model path, overlay color, behaviour counts/status helper tests.
- `tests/test_frontend_gui.py`: frontend GUI default fields and initial state.
- `tests/test_backend_gui.py`: backend GUI default fields and initial state.

## Task 1: Six-Class Model Asset, Dependency, and Behaviour Rules

**Files:**
- Modify: `requirements.txt`
- Modify: `src/backend/behaviour_analyzer.py`
- Modify: `src/backend/app.py`
- Modify: `tests/test_behaviour_analyzer.py`
- Modify: `tests/test_backend_app.py`
- Runtime asset: `models/classroom_behaviour_6cls.pt`

- [ ] **Step 1: Copy the YOLOv8 six-class model into this project**

Run:

```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
Copy-Item -LiteralPath "D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt" -Destination "models\classroom_behaviour_6cls.pt" -Force
```

Expected: `models\classroom_behaviour_6cls.pt` exists. It remains untracked because `.gitignore` ignores `*.pt`.

- [ ] **Step 2: Verify copied model labels**

Run:

```powershell
@'
from ultralytics import YOLO
model = YOLO("models/classroom_behaviour_6cls.pt")
print(model.names)
'@ | .\.venv\Scripts\python.exe -
```

Expected:

```text
{0: 'Hand-raise', 1: 'Reading', 2: 'Writing', 3: 'Useing-Phone', 4: 'Head-down', 5: 'Sleeping'}
```

- [ ] **Step 3: Add PyQt6 to runtime requirements**

Update `requirements.txt` so it contains:

```text
numpy>=1.26.0
opencv-python>=4.9.0
pytest>=8.0.0
ultralytics>=8.2.0
requests>=2.23.0
PyQt6>=6.6.0
```

- [ ] **Step 4: Install the updated requirements**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: PyQt6 installs successfully and `python -c "import PyQt6"` exits with code 0.

- [ ] **Step 5: Write failing tests for six-class rules and default model path**

Replace the final label-set test in `tests/test_behaviour_analyzer.py` and add the new target-level test:

```python
def test_declared_label_sets_match_yolov8_six_class_model():
    assert ABNORMAL_LABELS == {"Useing-Phone", "Head-down", "Sleeping"}
    assert NORMAL_LABELS == {"Hand-raise", "Reading", "Writing"}


def test_abnormal_and_normal_yolov8_detections_are_assessed_independently():
    analyzer = BehaviourAnalyzer(threshold_seconds=3.0, min_confidence=0.35)
    detections = [
        Detection(label="Sleeping", confidence=0.91, bbox=(10, 10, 80, 60)),
        Detection(label="Writing", confidence=0.88, bbox=(90, 10, 140, 120)),
    ]

    assessments, alarm = analyzer.update(detections, now_seconds=10.0)

    assert assessments[0].is_abnormal is True
    assert assessments[0].status == "abnormal"
    assert assessments[0].reason == "Sleeping"
    assert assessments[1].is_abnormal is False
    assert assessments[1].status == "normal"
    assert assessments[1].reason == "Writing"
    assert alarm.suspicious is True
    assert alarm.is_alarm is False
    assert alarm.abnormal_count == 1
    assert alarm.abnormal_labels == ("Sleeping",)
```

Update `tests/test_backend_app.py` default-model test:

```python
def test_backend_parser_defaults_to_yolov8_six_class_model():
    args = build_arg_parser().parse_args([])

    assert args.model == "models/classroom_behaviour_6cls.pt"
```

- [ ] **Step 6: Run tests and verify they fail for the expected reason**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_behaviour_analyzer.py tests\test_backend_app.py -q
```

Expected: tests fail because code still uses the old 12-class labels and old default model path.

- [ ] **Step 7: Implement six-class rules**

Update the top of `src/backend/behaviour_analyzer.py` to:

```python
ABNORMAL_LABELS = {"Useing-Phone", "Head-down", "Sleeping"}
NORMAL_LABELS = {"Hand-raise", "Reading", "Writing"}
LABEL_DISPLAY_NAMES = {
    "Hand-raise": "举手",
    "Reading": "看书",
    "Writing": "写字",
    "Useing-Phone": "使用手机",
    "Head-down": "低头",
    "Sleeping": "睡觉",
}

_ABNORMAL_BY_LOWER = {label.lower(): label for label in ABNORMAL_LABELS}
_NORMAL_BY_LOWER = {label.lower(): label for label in NORMAL_LABELS}
_DISPLAY_BY_LOWER = {label.lower(): display for label, display in LABEL_DISPLAY_NAMES.items()}


def display_label(label: str) -> str:
    return _DISPLAY_BY_LOWER.get(label.lower(), label)
```

Keep `BehaviourAnalyzer.update()` and `_assess_detection()` logic unchanged except for using the new constants.

- [ ] **Step 8: Implement default model path constant**

In `src/backend/app.py`, add a module constant near imports:

```python
DEFAULT_MODEL_PATH = "models/classroom_behaviour_6cls.pt"
```

Update `build_arg_parser()`:

```python
parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
```

- [ ] **Step 9: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_behaviour_analyzer.py tests\test_backend_app.py -q
```

Expected: all focused tests pass.

- [ ] **Step 10: Commit task 1**

Run:

```powershell
git add requirements.txt src/backend/behaviour_analyzer.py src/backend/app.py tests/test_behaviour_analyzer.py tests/test_backend_app.py
git commit -m "feat: use yolov8 six-class behaviour rules"
```

## Task 2: Reusable Overlay, Counts, and Target-Level Color Tests

**Files:**
- Modify: `src/backend/app.py`
- Modify: `tests/test_backend_app.py`

- [ ] **Step 1: Add failing tests for Chinese counts and per-target colors**

Append to `tests/test_backend_app.py`:

```python
from src.backend.app import behaviour_counts, frame_status_text


def test_behaviour_counts_uses_yolov8_chinese_labels():
    assessments = [
        DetectionAssessment(
            detection=Detection(label="Sleeping", confidence=0.91, bbox=(10, 10, 50, 80)),
            status="abnormal",
            is_abnormal=True,
            is_alarm=True,
            reason="Sleeping",
            duration_seconds=3.1,
        ),
        DetectionAssessment(
            detection=Detection(label="Writing", confidence=0.88, bbox=(70, 10, 120, 100)),
            status="normal",
            is_abnormal=False,
            is_alarm=False,
            reason="Writing",
            duration_seconds=0.0,
        ),
        DetectionAssessment(
            detection=Detection(label="Writing", confidence=0.77, bbox=(20, 20, 70, 90)),
            status="normal",
            is_abnormal=False,
            is_alarm=False,
            reason="Writing",
            duration_seconds=0.0,
        ),
    ]

    counts = behaviour_counts(assessments)

    assert counts["睡觉"] == 1
    assert counts["写字"] == 2
    assert counts["举手"] == 0


def test_frame_status_text_keeps_alarm_frame_level_only():
    alarm = AlarmState(
        is_alarm=True,
        suspicious=True,
        duration_seconds=3.1,
        reason="multi_behaviour_abnormal",
        abnormal_count=1,
        abnormal_labels=("Sleeping",),
    )

    assert frame_status_text(alarm) == "ALARM: 1 abnormal - Sleeping"
```

Update `test_draw_overlay_colours_each_detection_by_its_own_status()` to use `Sleeping` and `Writing` instead of old `sleep` and `upright` labels.

- [ ] **Step 2: Run tests and verify they fail for missing helpers**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_app.py -q
```

Expected: import or name failure for `behaviour_counts` and `frame_status_text`.

- [ ] **Step 3: Add reusable helper functions**

In `src/backend/app.py`, import `display_label`:

```python
from src.backend.behaviour_analyzer import LABEL_DISPLAY_NAMES, BehaviourAnalyzer, display_label
```

Add helpers before `draw_overlay()`:

```python
def behaviour_counts(assessments: list[DetectionAssessment]) -> dict[str, int]:
    counts = {display: 0 for display in LABEL_DISPLAY_NAMES.values()}
    for assessment in assessments:
        if assessment.status == "ignored":
            continue
        label = display_label(assessment.detection.label)
        counts[label] = counts.get(label, 0) + 1
    return counts


def frame_status_text(alarm: AlarmState) -> str:
    labels = ", ".join(alarm.abnormal_labels)
    if alarm.is_alarm:
        status = f"ALARM: {alarm.abnormal_count} abnormal"
        return f"{status} - {labels}" if labels else status
    if alarm.suspicious:
        status = f"suspicious: {alarm.abnormal_count} abnormal"
        return f"{status} - {labels}" if labels else status
    return "normal"
```

Update `draw_overlay()` to use `frame_status_text(alarm)` instead of duplicating status-string construction. Keep color selection per assessment:

```python
if assessment.status == "ignored":
    color = (120, 120, 120)
else:
    color = (0, 0, 255) if assessment.is_abnormal else (0, 180, 0)
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_app.py -q
```

Expected: all backend app tests pass.

- [ ] **Step 5: Commit task 2**

Run:

```powershell
git add src/backend/app.py tests/test_backend_app.py
git commit -m "feat: add backend display helpers"
```

## Task 3: Frontend PyQt Camera Sender GUI

**Files:**
- Create: `src/frontend/gui_client.py`
- Create: `tests/test_frontend_gui.py`

- [ ] **Step 1: Write failing frontend GUI default-state test**

Create `tests/test_frontend_gui.py`:

```python
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
```

- [ ] **Step 2: Run test and verify it fails because GUI file does not exist**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frontend_gui.py -q
```

Expected: import failure for `src.frontend.gui_client`.

- [ ] **Step 3: Create frontend GUI implementation**

Create `src/frontend/gui_client.py` with these public classes and methods:

```python
from __future__ import annotations

import socket
import time

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from src.common.image_codec import encode_jpeg, resize_to_width
from src.common.protocol import send_packet


def cv_frame_to_qimage(frame) -> QtGui.QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()


class CameraSenderWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    metrics_ready = QtCore.pyqtSignal(float, str, int, int)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, host: str, port: int, camera: int, width: int, fps: float, quality: int) -> None:
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
        delay_seconds = 1.0 / max(0.1, self.fps)
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开摄像头 {self.camera}")
            self._running = False
            return

        frame_id = 0
        started_at = time.time()
        try:
            self.status_changed.emit("连接中")
            with socket.create_connection((self.host, self.port), timeout=10) as conn:
                self.status_changed.emit("已连接")
                while self._running:
                    loop_started = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        self.error_occurred.emit("读取摄像头画面失败")
                        break

                    frame = resize_to_width(frame, self.width)
                    image_bytes = encode_jpeg(frame, quality=self.quality)
                    timestamp_ms = int(time.time() * 1000)
                    send_packet(conn, frame_id=frame_id, timestamp_ms=timestamp_ms, image_bytes=image_bytes)
                    self.frame_ready.emit(cv_frame_to_qimage(frame))

                    elapsed = max(0.001, time.time() - started_at)
                    fps = (frame_id + 1) / elapsed
                    height, width = frame.shape[:2]
                    self.metrics_ready.emit(fps, f"{width}x{height}", frame_id + 1, len(image_bytes))
                    frame_id += 1

                    sleep_for = delay_seconds - (time.time() - loop_started)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        except OSError as exc:
            self.error_occurred.emit(f"连接或发送失败：{exc}")
        finally:
            cap.release()
            self._running = False
            self.status_changed.emit("已断开")


class CameraClientWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: CameraSenderWorker | None = None
        self.setWindowTitle("课堂行为监测 - 前端发送端")
        self.resize(900, 620)
        self._build_ui()
        self._set_running(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QGridLayout()
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
        root.addLayout(form)

        self.preview_label = QtWidgets.QLabel("摄像头预览")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background:#111;color:#ddd;")
        root.addWidget(self.preview_label, 1)

        metrics = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("状态：未连接")
        self.fps_label = QtWidgets.QLabel("发送 FPS：0.0")
        self.resolution_label = QtWidgets.QLabel("分辨率：-")
        self.frame_count_label = QtWidgets.QLabel("已发送：0")
        self.jpeg_size_label = QtWidgets.QLabel("JPEG：0 KB")
        for widget in [self.status_label, self.fps_label, self.resolution_label, self.frame_count_label, self.jpeg_size_label]:
            metrics.addWidget(widget)
        root.addLayout(metrics)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始发送")
        self.stop_button = QtWidgets.QPushButton("停止发送")
        self.start_button.clicked.connect(self.start_sender)
        self.stop_button.clicked.connect(self.stop_sender)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)

        self.setCentralWidget(central)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def start_sender(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = CameraSenderWorker(
            host=self.host_edit.text().strip(),
            port=self.port_spin.value(),
            camera=self.camera_spin.value(),
            width=self.width_spin.value(),
            fps=self.fps_spin.value(),
            quality=self.quality_spin.value(),
        )
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished.connect(lambda: self._set_running(False))
        self._set_running(True)
        self.worker.start()

    def stop_sender(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self._set_running(False)

    def update_preview(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def update_metrics(self, fps: float, resolution: str, frame_count: int, jpeg_size: int) -> None:
        self.fps_label.setText(f"发送 FPS：{fps:.1f}")
        self.resolution_label.setText(f"分辨率：{resolution}")
        self.frame_count_label.setText(f"已发送：{frame_count}")
        self.jpeg_size_label.setText(f"JPEG：{jpeg_size / 1024:.1f} KB")

    def update_status(self, status: str) -> None:
        self.status_label.setText(f"状态：{status}")

    def show_error(self, message: str) -> None:
        self.status_label.setText(f"状态：错误 - {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_sender()
        super().closeEvent(event)


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = CameraClientWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run frontend GUI test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frontend_gui.py -q
```

Expected: test passes.

- [ ] **Step 5: Commit task 3**

Run:

```powershell
git add src/frontend/gui_client.py tests/test_frontend_gui.py
git commit -m "feat: add frontend pyqt sender"
```

## Task 4: Backend PyQt Monitoring GUI

**Files:**
- Create: `src/backend/gui_app.py`
- Create: `tests/test_backend_gui.py`

- [ ] **Step 1: Write failing backend GUI default-state test**

Create `tests/test_backend_gui.py`:

```python
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

    assert window.host_edit.text() == "0.0.0.0"
    assert window.port_spin.value() == 5001
    assert window.model_edit.text() == DEFAULT_MODEL_PATH
    assert window.alarm_spin.value() == 3.0
    assert "未监听" in window.status_label.text()
    assert window.stop_button.isEnabled() is False
```

- [ ] **Step 2: Run test and verify it fails because GUI file does not exist**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_gui.py -q
```

Expected: import failure for `src.backend.gui_app`.

- [ ] **Step 3: Create backend GUI implementation**

Create `src/backend/gui_app.py` with these public classes and methods:

```python
from __future__ import annotations

import socket
import time
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from src.backend.app import DEFAULT_MODEL_PATH, append_alarm, behaviour_counts, draw_overlay, frame_status_text
from src.backend.behaviour_analyzer import BehaviourAnalyzer
from src.backend.detector import YoloDetector
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet


def cv_frame_to_qimage(frame) -> QtGui.QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()


class BackendReceiverWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    metrics_ready = QtCore.pyqtSignal(float, int, int, str)
    counts_ready = QtCore.pyqtSignal(dict)
    alarm_ready = QtCore.pyqtSignal(str)
    log_ready = QtCore.pyqtSignal(str)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, host: str, port: int, model: str, alarm_seconds: float, output_dir: Path) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.alarm_seconds = alarm_seconds
        self.output_dir = output_dir
        self._running = False
        self._server: socket.socket | None = None

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.close()

    def run(self) -> None:
        self._running = True
        detector = None
        try:
            detector = YoloDetector(model_path=self.model)
        except Exception as exc:
            self.error_occurred.emit(f"模型加载失败：{exc}")
            self._running = False
            return

        analyzer = BehaviourAnalyzer(threshold_seconds=self.alarm_seconds)
        csv_path = self.output_dir / "alarms.csv"
        last_alarm_frame = -1
        frame_count = 0
        fps_started = time.time()

        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(1)
            self.status_changed.emit(f"监听中 {self.host}:{self.port}")
            conn, addr = self._server.accept()
            self.status_changed.emit(f"已连接 {addr[0]}:{addr[1]}")
            with conn:
                while self._running:
                    packet = recv_packet(conn)
                    frame = decode_jpeg(packet.image_bytes)
                    detected_at = time.time()
                    latency_ms = int(detected_at * 1000) - packet.timestamp_ms
                    detect_started = time.time()
                    detections = detector.detect(frame)
                    detect_ms = int((time.time() - detect_started) * 1000)
                    assessments, alarm = analyzer.update(detections, now_seconds=detected_at)
                    frame_count += 1
                    fps = frame_count / max(0.001, detected_at - fps_started)
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
                        self.log_ready.emit(f"报警 {packet.frame_id}: {frame_status_text(alarm)}")
                        last_alarm_frame = packet.frame_id

                    self.frame_ready.emit(cv_frame_to_qimage(overlay))
                    self.metrics_ready.emit(fps, latency_ms, detect_ms, f"{frame.shape[1]}x{frame.shape[0]}")
                    self.counts_ready.emit(behaviour_counts(assessments))
                    self.alarm_ready.emit(frame_status_text(alarm))
        except OSError as exc:
            if self._running:
                self.error_occurred.emit(f"网络错误：{exc}")
        except Exception as exc:
            if self._running:
                self.error_occurred.emit(f"运行错误：{exc}")
        finally:
            self._running = False
            if self._server is not None:
                try:
                    self._server.close()
                except OSError:
                    pass
            self.status_changed.emit("未监听")


class BackendMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: BackendReceiverWorker | None = None
        self.setWindowTitle("课堂行为监测 - 后端分析端")
        self.resize(1120, 720)
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
        self.alarm_spin.setValue(3.0)
        self.output_edit = QtWidgets.QLineEdit("output/alarms")
        form.addWidget(QtWidgets.QLabel("监听地址"), 0, 0)
        form.addWidget(self.host_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("端口"), 0, 2)
        form.addWidget(self.port_spin, 0, 3)
        form.addWidget(QtWidgets.QLabel("模型"), 1, 0)
        form.addWidget(self.model_edit, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("报警秒数"), 2, 0)
        form.addWidget(self.alarm_spin, 2, 1)
        form.addWidget(QtWidgets.QLabel("输出目录"), 2, 2)
        form.addWidget(self.output_edit, 2, 3)
        root.addLayout(form)

        body = QtWidgets.QHBoxLayout()
        self.video_label = QtWidgets.QLabel("等待前端连接")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(720, 420)
        self.video_label.setStyleSheet("background:#111;color:#ddd;")
        body.addWidget(self.video_label, 2)

        side = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("状态：未监听")
        self.alarm_label = QtWidgets.QLabel("报警：normal")
        self.metrics_label = QtWidgets.QLabel("FPS：0.0  延迟：0ms  检测：0ms  分辨率：-")
        self.counts_text = QtWidgets.QTextBrowser()
        self.log_text = QtWidgets.QTextBrowser()
        side.addWidget(self.status_label)
        side.addWidget(self.alarm_label)
        side.addWidget(self.metrics_label)
        side.addWidget(QtWidgets.QLabel("行为统计"))
        side.addWidget(self.counts_text, 1)
        side.addWidget(QtWidgets.QLabel("报警日志"))
        side.addWidget(self.log_text, 1)
        body.addLayout(side, 1)
        root.addLayout(body, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动监听")
        self.stop_button = QtWidgets.QPushButton("停止监听")
        self.start_button.clicked.connect(self.start_backend)
        self.stop_button.clicked.connect(self.stop_backend)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)
        self.setCentralWidget(central)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def start_backend(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = BackendReceiverWorker(
            host=self.host_edit.text().strip(),
            port=self.port_spin.value(),
            model=self.model_edit.text().strip(),
            alarm_seconds=self.alarm_spin.value(),
            output_dir=Path(self.output_edit.text().strip()),
        )
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.counts_ready.connect(self.update_counts)
        self.worker.alarm_ready.connect(self.update_alarm)
        self.worker.log_ready.connect(self.append_log)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished.connect(lambda: self._set_running(False))
        self._set_running(True)
        self.worker.start()

    def stop_backend(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self._set_running(False)

    def update_frame(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def update_metrics(self, fps: float, latency_ms: int, detect_ms: int, resolution: str) -> None:
        self.metrics_label.setText(f"FPS：{fps:.1f}  延迟：{latency_ms}ms  检测：{detect_ms}ms  分辨率：{resolution}")

    def update_counts(self, counts: dict) -> None:
        lines = [f"{label}: {count}" for label, count in counts.items()]
        self.counts_text.setText("\n".join(lines))

    def update_alarm(self, status: str) -> None:
        self.alarm_label.setText(f"报警：{status}")

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

    def update_status(self, status: str) -> None:
        self.status_label.setText(f"状态：{status}")

    def show_error(self, message: str) -> None:
        self.status_label.setText(f"状态：错误 - {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_backend()
        super().closeEvent(event)


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = BackendMonitorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run backend GUI test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_gui.py -q
```

Expected: test passes.

- [ ] **Step 5: Commit task 4**

Run:

```powershell
git add src/backend/gui_app.py tests/test_backend_gui.py
git commit -m "feat: add backend pyqt monitor"
```

## Task 5: GUI Launch Scripts and Backend Package

**Files:**
- Create: `START_BACKEND_GUI.ps1`
- Create: `START_FRONTEND_GUI.ps1`
- Modify: `scripts/package_backend.ps1`
- Test: `tests/test_gui_defaults.py`

- [ ] **Step 1: Write failing launch-script and package-default tests**

Create `tests/test_gui_defaults.py`:

```python
from pathlib import Path


def test_gui_launch_scripts_exist_and_target_gui_modules():
    backend = Path("START_BACKEND_GUI.ps1").read_text(encoding="utf-8")
    frontend = Path("START_FRONTEND_GUI.ps1").read_text(encoding="utf-8")

    assert "-m src.backend.gui_app" in backend
    assert "-m src.frontend.gui_client" in frontend


def test_package_script_uses_six_class_model():
    script = Path("scripts/package_backend.ps1").read_text(encoding="utf-8")

    assert 'models\\classroom_behaviour_6cls.pt' in script
    assert "START_BACKEND_GUI.ps1" in script
```

- [ ] **Step 2: Run tests and verify missing files fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gui_defaults.py -q
```

Expected: failures for missing launch scripts and old package defaults.

- [ ] **Step 3: Add GUI launch scripts**

Create `START_BACKEND_GUI.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
.\.venv\Scripts\python.exe -m src.backend.gui_app
```

Create `START_FRONTEND_GUI.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
.\.venv\Scripts\python.exe -m src.frontend.gui_client
```

- [ ] **Step 4: Update backend package script defaults**

In `scripts/package_backend.ps1`, change the default model parameter to:

```powershell
[string]$ModelPath = "models\classroom_behaviour_6cls.pt"
```

Change copied model destination to:

```powershell
Copy-Item -LiteralPath $modelFullPath -Destination (Join-Path $packageDir "models\classroom_behaviour_6cls.pt")
```

Copy GUI launch script into the package:

```powershell
Copy-Item -LiteralPath "START_BACKEND_GUI.ps1" -Destination (Join-Path $packageDir "START_BACKEND_GUI.ps1")
```

Update packaged `README_BACKEND.md` text so default model is:

```text
models\classroom_behaviour_6cls.pt
```

Update generated `START_BACKEND.ps1` model argument:

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model models\classroom_behaviour_6cls.pt
```

- [ ] **Step 5: Run launch/package tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gui_defaults.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit task 5**

Run:

```powershell
git add START_BACKEND_GUI.ps1 START_FRONTEND_GUI.ps1 scripts/package_backend.ps1 tests/test_gui_defaults.py
git commit -m "feat: add gui launch and package defaults"
```

## Task 6: Offline Evidence, README, and Checklist

**Files:**
- Modify: `README.md`
- Modify: `docs/course-evidence/checklist.md`
- Create: `docs/course-evidence/yolov8-6cls-offline-test.md`

- [ ] **Step 1: Run six-class offline test against YOLOv8 sample images**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\classroom_behaviour_6cls.pt --output-dir output\offline_test\yolov8-6cls --limit 0 --conf 0.25
```

Expected:

```text
output\offline_test\yolov8-6cls\predictions.csv
output\offline_test\yolov8-6cls\*.jpg
```

- [ ] **Step 2: Create offline test evidence document**

Create `docs/course-evidence/yolov8-6cls-offline-test.md`:

```markdown
# YOLOv8 六类课堂行为模型离线测试记录

## 模型

- 来源：`D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt`
- 本项目路径：`models/classroom_behaviour_6cls.pt`
- 类别：`Hand-raise`、`Reading`、`Writing`、`Useing-Phone`、`Head-down`、`Sleeping`

## 测试命令

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\classroom_behaviour_6cls.pt --output-dir output\offline_test\yolov8-6cls --limit 0 --conf 0.25
```

## 输出

- 预测 CSV：`output/offline_test/yolov8-6cls/predictions.csv`
- 标注图片：`output/offline_test/yolov8-6cls/*.jpg`

## 课程说明

该测试用于证明六类模型能在本项目环境中加载并输出课堂行为类别。双机实时演示时，前端 GUI 负责摄像头采集与网络发送，后端 GUI 负责调用同一模型进行实时检测、目标级异常标红和报警记录。
```

- [ ] **Step 3: Update README GUI workflow**

Update `README.md` so the primary startup section contains:

```markdown
## 后端 GUI 启动

```powershell
.\START_BACKEND_GUI.ps1
```

后端默认监听 `0.0.0.0:5001`，模型路径为 `models/classroom_behaviour_6cls.pt`。

## 前端 GUI 启动

```powershell
.\START_FRONTEND_GUI.ps1
```

在前端窗口填写后端电脑的无线网卡 IPv4 地址，例如 `192.168.1.20`，点击“开始发送”。

## 六类课堂行为

正常状态：

- `Hand-raise`：举手
- `Reading`：看书
- `Writing`：写字

异常状态：

- `Useing-Phone`：使用手机
- `Head-down`：低头
- `Sleeping`：睡觉

报警触发时，系统只给异常目标标红，正常目标保持绿色。报警状态用于顶部提示、截图和 CSV 日志，不会把整帧所有目标统一变红。
```

Keep the existing command-line startup commands as a fallback section.

- [ ] **Step 4: Update evidence checklist**

Update `docs/course-evidence/checklist.md` so required screenshots include:

```markdown
- 前端 PyQt GUI 发送窗口截图
- 后端 PyQt GUI 监听窗口截图
- 后端六类 YOLO 检测框截图
- 同一帧中正常目标绿色、异常目标红色的目标级标红截图
- 异常持续超过 3 秒后的报警截图
- `output/alarms/alarms.csv` 报警记录截图
```

Update answer points to include:

```markdown
- 报警是帧级提示，但检测框颜色是目标级判断。
- 异常目标包括使用手机、低头和睡觉。
- 正常目标包括举手、看书和写字。
```

- [ ] **Step 5: Run documentation-adjacent tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gui_defaults.py tests\test_backend_app.py tests\test_behaviour_analyzer.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit task 6**

Run:

```powershell
git add README.md docs/course-evidence/checklist.md docs/course-evidence/yolov8-6cls-offline-test.md
git commit -m "docs: document yolov8 gui workflow"
```

## Task 7: End-to-End Verification and Packaging

**Files:**
- No source changes expected unless verification finds a defect.
- Runtime output: `output/offline_test/yolov8-6cls/`
- Runtime output: `dist/backend-student-sleep-server.zip`

- [ ] **Step 1: Run full automated test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Verify backend GUI can initialize headlessly**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
@'
import sys
from PyQt6 import QtWidgets
from src.backend.gui_app import BackendMonitorWindow
app = QtWidgets.QApplication(sys.argv)
window = BackendMonitorWindow()
print(window.windowTitle())
'@ | .\.venv\Scripts\python.exe -
Remove-Item Env:\QT_QPA_PLATFORM
```

Expected output:

```text
课堂行为监测 - 后端分析端
```

- [ ] **Step 3: Verify frontend GUI can initialize headlessly**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
@'
import sys
from PyQt6 import QtWidgets
from src.frontend.gui_client import CameraClientWindow
app = QtWidgets.QApplication(sys.argv)
window = CameraClientWindow()
print(window.windowTitle())
'@ | .\.venv\Scripts\python.exe -
Remove-Item Env:\QT_QPA_PLATFORM
```

Expected output:

```text
课堂行为监测 - 前端发送端
```

- [ ] **Step 4: Build backend package**

Run:

```powershell
.\scripts\package_backend.ps1
```

Expected:

```text
dist\backend-student-sleep-server\
dist\backend-student-sleep-server.zip
```

- [ ] **Step 5: Inspect package contents**

Run:

```powershell
Get-ChildItem -Recurse dist\backend-student-sleep-server | Select-Object FullName
```

Expected package includes:

```text
models\classroom_behaviour_6cls.pt
src\backend\gui_app.py
src\backend\app.py
src\common\protocol.py
START_BACKEND_GUI.ps1
START_BACKEND.ps1
requirements.txt
```

- [ ] **Step 6: Manual GUI smoke test on one machine**

Run backend GUI:

```powershell
.\START_BACKEND_GUI.ps1
```

Run frontend GUI in a second PowerShell window:

```powershell
.\START_FRONTEND_GUI.ps1
```

Use `127.0.0.1` as backend IP for same-machine smoke testing. Expected:

- Frontend preview shows local camera.
- Frontend status becomes connected.
- Backend shows received frames.
- Backend loads six-class model.
- Normal detections are green.
- `Useing-Phone`, `Head-down`, or `Sleeping` detections are red.
- Alarm status changes after the configured duration.

- [ ] **Step 7: Commit verification fixes if any**

If Step 1 through Step 6 required a code or doc fix, commit it:

```powershell
git add <changed-files>
git commit -m "fix: complete gui verification"
```

If no files changed, do not create an empty commit.

## Self-Review

- Spec coverage: Tasks cover model copy, six-class rules, target-level red boxes, frontend GUI, backend GUI, GUI scripts, offline evidence, README, checklist, packaging, and verification.
- Placeholder scan: No task uses unresolved marker words or unspecified implementation steps.
- Type consistency: GUI tests reference `CameraClientWindow`, `BackendMonitorWindow`, `DEFAULT_MODEL_PATH`, `behaviour_counts`, and `frame_status_text`, all defined in earlier tasks.
- Scope check: The plan keeps the existing TCP protocol and CLI entrypoints. It does not add identity tracking, database, web UI, or unrelated refactors.
