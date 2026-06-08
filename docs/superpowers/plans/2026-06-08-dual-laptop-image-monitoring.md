# Dual Laptop Image Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-laptop real-time classroom image monitoring system that streams camera frames over Wi-Fi, runs YOLO-based detection on the backend, and raises an alarm for suspected desk-sleeping behavior.

**Architecture:** The frontend laptop captures webcam frames, JPEG-encodes them, and sends length-prefixed packets over TCP. The backend laptop receives frames, decodes them, runs YOLO detection, applies a continuous-time sleep rule, displays annotated video, and saves alarm logs/screenshots for course acceptance.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, TCP sockets, pytest, ultralytics YOLO.

---

## File Structure

- Create: `requirements.txt` - Python dependencies for runtime and tests.
- Create: `pyproject.toml` - pytest import path and test configuration.
- Create: `README.md` - student-facing runbook for two laptops.
- Create: `src/__init__.py` - package marker.
- Create: `src/common/__init__.py` - shared package marker.
- Create: `src/common/protocol.py` - TCP frame packet encoding, parsing, sending, and receiving.
- Create: `src/common/image_codec.py` - JPEG encoding/decoding and frame resizing helpers.
- Create: `src/common/types.py` - shared detection and alarm dataclasses.
- Create: `src/frontend/__init__.py` - frontend package marker.
- Create: `src/frontend/camera_client.py` - camera capture and TCP sender CLI.
- Create: `src/backend/__init__.py` - backend package marker.
- Create: `src/backend/detector.py` - YOLO adapter and result conversion.
- Create: `src/backend/sleep_analyzer.py` - continuous multi-frame suspected-sleep rule.
- Create: `src/backend/app.py` - backend receiver, detection loop, overlay, alarm persistence, and CLI.
- Create: `tests/test_protocol.py` - protocol unit tests.
- Create: `tests/test_image_codec.py` - image codec unit tests.
- Create: `tests/test_sleep_analyzer.py` - sleep rule unit tests.
- Create: `tests/test_detector.py` - YOLO result conversion tests without loading a model.

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `src/common/__init__.py`
- Create: `src/frontend/__init__.py`
- Create: `src/backend/__init__.py`

- [ ] **Step 1: Initialize version control**

Run:

```powershell
git init
```

Expected: output contains `Initialized empty Git repository`.

- [ ] **Step 2: Create dependency file**

Create `requirements.txt`:

```txt
numpy>=1.26.0
opencv-python>=4.9.0
pytest>=8.0.0
ultralytics>=8.2.0
```

- [ ] **Step 3: Create pytest configuration**

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 4: Create package markers**

Create these empty files:

```txt
src/__init__.py
src/common/__init__.py
src/frontend/__init__.py
src/backend/__init__.py
```

- [ ] **Step 5: Create the initial runbook**

Create `README.md`:

````markdown
# 基于双机的实时图像远程监测与异常分析系统

## 运行环境

- Python 3.10+
- 两台笔记本位于同一无线局域网
- 前端笔记本连接摄像头
- 后端笔记本安装依赖并运行检测端

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 后端启动

在后端笔记本运行：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model yolov8n.pt
```

记录后端笔记本的无线网卡 IPv4 地址，例如 `192.168.1.20`。

## 前端启动

在前端笔记本运行：

```powershell
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 192.168.1.20 --port 5001 --camera 0
```

## 验收演示

1. 后端显示前端摄像头画面。
2. 后端画面显示检测框和检测标签。
3. 出现疑似趴桌睡觉动作并持续 3 秒后，后端显示报警信息。
4. `output/alarms/alarms.csv` 保存报警记录。
5. `output/alarms/*.jpg` 保存报警截图。
````

- [ ] **Step 6: Install dependencies**

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: dependency installation exits with code `0`.

- [ ] **Step 7: Run the empty test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: pytest reports that no tests were collected or exits successfully after later tasks add tests.

- [ ] **Step 8: Commit scaffold**

Run:

```powershell
git add requirements.txt pyproject.toml README.md src
git commit -m "chore: scaffold monitoring project"
```

Expected: commit succeeds.

## Task 2: TCP Frame Protocol

**Files:**
- Create: `src/common/protocol.py`
- Create: `tests/test_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Create `tests/test_protocol.py`:

```python
import socket
import threading

import pytest

from src.common.protocol import HEADER_SIZE, encode_packet, parse_header, recv_packet


def test_encode_parse_header_round_trip():
    image_bytes = b"jpeg-bytes"
    packet = encode_packet(frame_id=7, timestamp_ms=123456789, image_bytes=image_bytes)

    frame_id, timestamp_ms, image_len = parse_header(packet[:HEADER_SIZE])

    assert frame_id == 7
    assert timestamp_ms == 123456789
    assert image_len == len(image_bytes)
    assert packet[HEADER_SIZE:] == image_bytes


def test_parse_header_rejects_invalid_magic():
    packet = bytearray(encode_packet(frame_id=1, timestamp_ms=2, image_bytes=b"x"))
    packet[0:4] = b"BAD!"

    with pytest.raises(ValueError, match="Invalid frame magic"):
        parse_header(bytes(packet[:HEADER_SIZE]))


def test_recv_packet_handles_split_tcp_chunks():
    sender, receiver = socket.socketpair()
    packet = encode_packet(frame_id=3, timestamp_ms=99, image_bytes=b"abcdef")

    def write_chunks():
        try:
            sender.sendall(packet[:2])
            sender.sendall(packet[2:9])
            sender.sendall(packet[9:])
        finally:
            sender.close()

    writer = threading.Thread(target=write_chunks)
    writer.start()

    try:
        result = recv_packet(receiver)
    finally:
        receiver.close()
        writer.join(timeout=2)

    assert result.frame_id == 3
    assert result.timestamp_ms == 99
    assert result.image_bytes == b"abcdef"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_protocol.py
```

Expected: FAIL with import errors because `src.common.protocol` does not exist.

- [ ] **Step 3: Implement protocol module**

Create `src/common/protocol.py`:

```python
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass


MAGIC = b"NSGD"
HEADER_FORMAT = "!4sIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ms: int
    image_bytes: bytes


def encode_packet(frame_id: int, timestamp_ms: int, image_bytes: bytes) -> bytes:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"image_bytes exceeds {MAX_IMAGE_BYTES} bytes")

    header = struct.pack(HEADER_FORMAT, MAGIC, frame_id, timestamp_ms, len(image_bytes))
    return header + image_bytes


def parse_header(header_bytes: bytes) -> tuple[int, int, int]:
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes")

    magic, frame_id, timestamp_ms, image_len = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC:
        raise ValueError("Invalid frame magic")
    if image_len <= 0 or image_len > MAX_IMAGE_BYTES:
        raise ValueError(f"Invalid image length: {image_len}")
    return frame_id, timestamp_ms, image_len


def recvall(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if chunk == b"":
            raise ConnectionError("Socket closed before enough bytes were received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(conn: socket.socket, frame_id: int, timestamp_ms: int, image_bytes: bytes) -> None:
    conn.sendall(encode_packet(frame_id, timestamp_ms, image_bytes))


def recv_packet(conn: socket.socket) -> FramePacket:
    header = recvall(conn, HEADER_SIZE)
    frame_id, timestamp_ms, image_len = parse_header(header)
    image_bytes = recvall(conn, image_len)
    return FramePacket(frame_id=frame_id, timestamp_ms=timestamp_ms, image_bytes=image_bytes)
```

- [ ] **Step 4: Run protocol tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_protocol.py
```

Expected: all protocol tests pass.

- [ ] **Step 5: Commit protocol**

Run:

```powershell
git add src/common/protocol.py tests/test_protocol.py
git commit -m "feat: add tcp frame protocol"
```

Expected: commit succeeds.

## Task 3: Image Codec and Shared Types

**Files:**
- Create: `src/common/image_codec.py`
- Create: `src/common/types.py`
- Create: `tests/test_image_codec.py`

- [ ] **Step 1: Write failing image codec tests**

Create `tests/test_image_codec.py`:

```python
import numpy as np

from src.common.image_codec import decode_jpeg, encode_jpeg, resize_to_width


def test_encode_decode_jpeg_round_trip_shape():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :, 1] = 180

    data = encode_jpeg(frame, quality=80)
    decoded = decode_jpeg(data)

    assert isinstance(data, bytes)
    assert decoded.shape == frame.shape
    assert decoded.dtype == np.uint8


def test_resize_to_width_keeps_aspect_ratio():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_to_width(frame, target_width=50)

    assert resized.shape == (25, 50, 3)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_image_codec.py
```

Expected: FAIL with import errors because `src.common.image_codec` does not exist.

- [ ] **Step 3: Implement shared types**

Create `src/common/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class AlarmState:
    is_alarm: bool
    suspicious: bool
    duration_seconds: float
    reason: str
```

- [ ] **Step 4: Implement image codec**

Create `src/common/image_codec.py`:

```python
from __future__ import annotations

import cv2
import numpy as np


def resize_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    height, width = frame.shape[:2]
    if width == target_width:
        return frame
    scale = target_width / width
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    if frame is None or frame.size == 0:
        raise ValueError("frame must not be empty")
    if quality < 1 or quality > 100:
        raise ValueError("quality must be between 1 and 100")

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode frame as JPEG")
    return buffer.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("data must not be empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode JPEG frame")
    return frame
```

- [ ] **Step 5: Run image tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_image_codec.py
```

Expected: all image codec tests pass.

- [ ] **Step 6: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 7: Commit codec and types**

Run:

```powershell
git add src/common/image_codec.py src/common/types.py tests/test_image_codec.py
git commit -m "feat: add image codec helpers"
```

Expected: commit succeeds.

## Task 4: Sleep Analyzer

**Files:**
- Create: `src/backend/sleep_analyzer.py`
- Create: `tests/test_sleep_analyzer.py`

- [ ] **Step 1: Write failing sleep analyzer tests**

Create `tests/test_sleep_analyzer.py`:

```python
from src.backend.sleep_analyzer import SleepAnalyzer
from src.common.types import Detection


def test_explicit_sleep_label_alarms_after_threshold():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    detections = [Detection(label="sleep", confidence=0.91, bbox=(10, 10, 80, 50))]

    first = analyzer.update(detections, now_seconds=10.0)
    second = analyzer.update(detections, now_seconds=12.0)
    third = analyzer.update(detections, now_seconds=13.1)

    assert first.suspicious is True
    assert first.is_alarm is False
    assert second.is_alarm is False
    assert third.is_alarm is True
    assert third.reason == "sleep_label"


def test_normal_person_does_not_alarm():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    detections = [Detection(label="person", confidence=0.88, bbox=(10, 10, 60, 180))]

    result = analyzer.update(detections, now_seconds=20.0)

    assert result.suspicious is False
    assert result.is_alarm is False
    assert result.reason == "normal"


def test_wide_person_box_is_suspicious_for_desk_sleep():
    analyzer = SleepAnalyzer(threshold_seconds=1.0, wide_person_ratio=1.3)
    detections = [Detection(label="person", confidence=0.9, bbox=(10, 10, 160, 90))]

    analyzer.update(detections, now_seconds=1.0)
    result = analyzer.update(detections, now_seconds=2.2)

    assert result.is_alarm is True
    assert result.reason == "wide_person_box"


def test_normal_frame_resets_suspicious_timer():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    sleep = [Detection(label="sleep", confidence=0.91, bbox=(10, 10, 80, 50))]
    normal = [Detection(label="person", confidence=0.91, bbox=(10, 10, 60, 180))]

    analyzer.update(sleep, now_seconds=1.0)
    reset = analyzer.update(normal, now_seconds=2.0)
    later = analyzer.update(sleep, now_seconds=5.0)

    assert reset.suspicious is False
    assert later.is_alarm is False
    assert later.duration_seconds == 0.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sleep_analyzer.py
```

Expected: FAIL with import errors because `src.backend.sleep_analyzer` does not exist.

- [ ] **Step 3: Implement sleep analyzer**

Create `src/backend/sleep_analyzer.py`:

```python
from __future__ import annotations

from src.common.types import AlarmState, Detection


class SleepAnalyzer:
    def __init__(
        self,
        threshold_seconds: float = 3.0,
        min_confidence: float = 0.45,
        wide_person_ratio: float = 1.45,
    ) -> None:
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be positive")
        self.threshold_seconds = threshold_seconds
        self.min_confidence = min_confidence
        self.wide_person_ratio = wide_person_ratio
        self._suspicious_since: float | None = None

    def update(self, detections: list[Detection], now_seconds: float) -> AlarmState:
        suspicious, reason = self._is_suspicious(detections)
        if not suspicious:
            self._suspicious_since = None
            return AlarmState(False, False, 0.0, "normal")

        if self._suspicious_since is None:
            self._suspicious_since = now_seconds
            return AlarmState(False, True, 0.0, reason)

        duration = max(0.0, now_seconds - self._suspicious_since)
        return AlarmState(duration >= self.threshold_seconds, True, duration, reason)

    def _is_suspicious(self, detections: list[Detection]) -> tuple[bool, str]:
        for detection in detections:
            label = detection.label.lower()
            if detection.confidence < self.min_confidence:
                continue

            if label in {"sleep", "desk_sleep", "head_down", "lying"}:
                return True, "sleep_label"

            if label == "person" and detection.height > 0:
                width_height_ratio = detection.width / detection.height
                if width_height_ratio >= self.wide_person_ratio:
                    return True, "wide_person_box"

        return False, "normal"
```

- [ ] **Step 4: Run sleep analyzer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sleep_analyzer.py
```

Expected: all sleep analyzer tests pass.

- [ ] **Step 5: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 6: Commit sleep analyzer**

Run:

```powershell
git add src/backend/sleep_analyzer.py tests/test_sleep_analyzer.py
git commit -m "feat: add suspected sleep analyzer"
```

Expected: commit succeeds.

## Task 5: YOLO Detector Adapter

**Files:**
- Create: `src/backend/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing detector conversion tests**

Create `tests/test_detector.py`:

```python
from src.backend.detector import result_to_detections


class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls]


class FakeResult:
    names = {0: "person", 1: "sleep"}

    def __init__(self):
        self.boxes = [
            FakeBox([1.2, 2.8, 30.1, 50.9], 0.88, 0),
            FakeBox([5.0, 6.0, 40.0, 24.0], 0.92, 1),
        ]


def test_result_to_detections_converts_boxes():
    detections = result_to_detections(FakeResult())

    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].confidence == 0.88
    assert detections[0].bbox == (1, 3, 30, 51)
    assert detections[1].label == "sleep"
    assert detections[1].bbox == (5, 6, 40, 24)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py
```

Expected: FAIL with import errors because `src.backend.detector` does not exist.

- [ ] **Step 3: Implement detector adapter**

Create `src/backend/detector.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.types import Detection


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return _scalar(value[0])
    return float(value)


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    x1, y1, x2, y2 = value
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def result_to_detections(result: Any) -> list[Detection]:
    detections: list[Detection] = []
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(round(_scalar(box.cls)))
        label = str(names.get(cls_id, cls_id))
        confidence = round(_scalar(box.conf), 4)
        bbox = _xyxy(box.xyxy)
        detections.append(Detection(label=label, confidence=confidence, bbox=bbox))

    return detections


class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return []
        return result_to_detections(results[0])
```

- [ ] **Step 4: Run detector tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py
```

Expected: all detector tests pass without loading a YOLO model.

- [ ] **Step 5: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 6: Commit detector**

Run:

```powershell
git add src/backend/detector.py tests/test_detector.py
git commit -m "feat: add yolo detector adapter"
```

Expected: commit succeeds.

## Task 6: Backend Monitoring App

**Files:**
- Create: `src/backend/app.py`
- Modify: `README.md`

- [ ] **Step 1: Implement backend app**

Create `src/backend/app.py`:

```python
from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from src.backend.detector import YoloDetector
from src.backend.sleep_analyzer import SleepAnalyzer
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet
from src.common.types import AlarmState, Detection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend receiver and YOLO alarm display")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--alarm-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", default="output/alarms")
    return parser


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    alarm: AlarmState,
    fps: float,
    latency_ms: int,
) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 0, 255) if alarm.is_alarm else (0, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    status = "ALARM: suspected sleeping" if alarm.is_alarm else "normal"
    status_color = (0, 0, 255) if alarm.is_alarm else (0, 180, 0)
    cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(output, f"fps={fps:.1f} latency={latency_ms}ms reason={alarm.reason}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def append_alarm(csv_path: Path, frame_id: int, timestamp_ms: int, reason: str, duration: float, image_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["frame_id", "timestamp_ms", "reason", "duration_seconds", "image_path"])
        writer.writerow([frame_id, timestamp_ms, reason, f"{duration:.2f}", str(image_path)])


def run_backend(host: str, port: int, model: str, alarm_seconds: float, output_dir: Path) -> None:
    detector = YoloDetector(model_path=model)
    analyzer = SleepAnalyzer(threshold_seconds=alarm_seconds)
    csv_path = output_dir / "alarms.csv"
    last_alarm_frame = -1
    frame_count = 0
    fps_started = time.time()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Backend listening on {host}:{port}")

    try:
        conn, addr = server.accept()
        print(f"Frontend connected: {addr}")
        with conn:
            while True:
                packet = recv_packet(conn)
                frame = decode_jpeg(packet.image_bytes)
                now = time.time()
                latency_ms = int(now * 1000) - packet.timestamp_ms

                detections = detector.detect(frame)
                alarm = analyzer.update(detections, now_seconds=now)

                frame_count += 1
                elapsed = max(0.001, now - fps_started)
                fps = frame_count / elapsed
                overlay = draw_overlay(frame, detections, alarm, fps=fps, latency_ms=latency_ms)

                if alarm.is_alarm and packet.frame_id != last_alarm_frame:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = output_dir / f"alarm_{packet.frame_id}_{packet.timestamp_ms}.jpg"
                    cv2.imwrite(str(image_path), overlay)
                    append_alarm(csv_path, packet.frame_id, packet.timestamp_ms, alarm.reason, alarm.duration_seconds, image_path)
                    last_alarm_frame = packet.frame_id

                cv2.imshow("Backend Monitoring", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        server.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_backend(
        host=args.host,
        port=args.port,
        model=args.model,
        alarm_seconds=args.alarm_seconds,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src/backend/app.py
```

Expected: command exits with code `0`.

- [ ] **Step 3: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 4: Commit backend app**

Run:

```powershell
git add src/backend/app.py README.md
git commit -m "feat: add backend monitoring app"
```

Expected: commit succeeds.

## Task 7: Frontend Camera Client

**Files:**
- Create: `src/frontend/camera_client.py`
- Modify: `README.md`

- [ ] **Step 1: Implement frontend client**

Create `src/frontend/camera_client.py`:

```python
from __future__ import annotations

import argparse
import socket
import time

import cv2

from src.common.image_codec import encode_jpeg, resize_to_width
from src.common.protocol import send_packet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frontend camera sender")
    parser.add_argument("--host", required=True, help="Backend IPv4 address")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--quality", type=int, default=80)
    return parser


def run_client(host: str, port: int, camera: int, width: int, fps: float, quality: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")

    delay_seconds = 1.0 / fps
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")

    frame_id = 0
    print(f"Connecting to backend {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=10) as conn:
            print("Connected. Press q in the preview window to stop.")
            while True:
                started = time.time()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read frame from camera")

                frame = resize_to_width(frame, width)
                image_bytes = encode_jpeg(frame, quality=quality)
                timestamp_ms = int(time.time() * 1000)
                send_packet(conn, frame_id=frame_id, timestamp_ms=timestamp_ms, image_bytes=image_bytes)

                cv2.imshow("Frontend Camera", frame)
                frame_id += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                elapsed = time.time() - started
                sleep_for = delay_seconds - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_client(
        host=args.host,
        port=args.port,
        camera=args.camera,
        width=args.width,
        fps=args.fps,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src/frontend/camera_client.py
```

Expected: command exits with code `0`.

- [ ] **Step 3: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 4: Commit frontend client**

Run:

```powershell
git add src/frontend/camera_client.py README.md
git commit -m "feat: add frontend camera sender"
```

Expected: commit succeeds.

## Task 8: Single-Machine Smoke Test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Start backend on one laptop**

Run in terminal 1:

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 127.0.0.1 --port 5001 --model yolov8n.pt
```

Expected: backend prints `Backend listening on 127.0.0.1:5001`.

- [ ] **Step 2: Start frontend on the same laptop**

Run in terminal 2:

```powershell
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 127.0.0.1 --port 5001 --camera 0
```

Expected: frontend prints `Connected`; backend opens `Backend Monitoring`; frontend opens `Frontend Camera`.

- [ ] **Step 3: Verify baseline behavior**

Perform normal sitting posture for 10 seconds.

Expected:

```txt
Backend video is smooth enough for demonstration.
Status remains normal.
No new alarm image is saved under output/alarms.
```

- [ ] **Step 4: Verify suspected sleeping behavior**

Perform a desk-sleep posture for at least 4 seconds. If the default YOLO model only outputs `person`, lean forward so the visible person box becomes wider than tall. If a custom model is used, confirm its sleeping label is one of `sleep`, `desk_sleep`, `head_down`, or `lying`.

Expected:

```txt
Backend status changes to ALARM: suspected sleeping.
An alarm image is saved under output/alarms.
output/alarms/alarms.csv contains one or more alarm records.
```

- [ ] **Step 5: Add smoke test notes to README**

Append to `README.md`:

````markdown
## 单机烟测

在没有第二台笔记本时，可以先在同一台电脑上完成闭环测试：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 127.0.0.1 --port 5001 --model yolov8n.pt
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 127.0.0.1 --port 5001 --camera 0
```

正常坐姿应保持 normal 状态。趴桌姿态持续超过 3 秒后，应出现报警提示，并在 `output/alarms` 下生成报警记录。
````

- [ ] **Step 6: Commit smoke test documentation**

Run:

```powershell
git add README.md
git commit -m "docs: add smoke test runbook"
```

Expected: commit succeeds.

## Task 9: Two-Laptop Acceptance Test and Course Evidence

**Files:**
- Create: `docs/course-evidence/checklist.md`

- [ ] **Step 1: Create acceptance checklist**

Create `docs/course-evidence/checklist.md`:

```markdown
# 验收证据清单

## 双机环境

- 后端笔记本 IPv4 地址：
- 前端笔记本摄像头编号：0
- 后端端口：5001
- 图像分辨率：640 像素宽
- 发送帧率：8 FPS
- 报警阈值：连续 3 秒

## 必备截图

- 前端摄像头采集窗口截图
- 后端实时监控窗口正常状态截图
- 后端 YOLO 检测框截图
- 疑似睡觉报警截图
- `output/alarms/alarms.csv` 报警记录截图

## 必备视频

- 双机运行演示视频
- 正常坐姿无报警片段
- 趴桌睡觉触发报警片段

## 报告数据

- 平均接收帧率
- 平均检测延迟
- 正常坐姿测试次数
- 低头写字测试次数
- 趴桌睡觉测试次数
- 误报次数
- 漏报次数

## 答辩说明要点

- TCP 数据包格式包含 magic、frame_id、timestamp、image_len 和 image_data。
- 后端按 image_len 读取完整 JPEG，解决 TCP 粘包和拆包问题。
- YOLO 负责目标检测，睡觉判断使用连续多帧规则降低误报。
- 当前系统识别的是“疑似趴桌睡觉行为”，不是医学意义上的睡眠状态。
- 后续改进方向是采集课堂数据并训练 sleep/normal 自定义 YOLO 类别。
```

- [ ] **Step 2: Run two-laptop backend**

On backend laptop, find IPv4 address:

```powershell
ipconfig
```

Run:

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model yolov8n.pt
```

Expected: backend prints `Backend listening on 0.0.0.0:5001`.

- [ ] **Step 3: Run two-laptop frontend**

On frontend laptop, replace `192.168.1.20` with the backend IPv4 address:

```powershell
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 192.168.1.20 --port 5001 --camera 0
```

Expected: backend receives and displays live camera frames.

- [ ] **Step 4: Collect course evidence**

Capture the screenshots and videos listed in `docs/course-evidence/checklist.md`.

Expected:

```txt
The evidence folder contains screenshots of normal monitoring, YOLO detection, suspected sleep alarm, and alarm CSV records.
The demo video shows the complete frontend-to-backend flow.
```

- [ ] **Step 5: Commit evidence checklist**

Run:

```powershell
git add docs/course-evidence/checklist.md
git commit -m "docs: add acceptance evidence checklist"
```

Expected: commit succeeds.

## Self-Review

### Spec Coverage

- Dual-laptop image capture and transmission: covered by Tasks 2, 7, 8, and 9.
- Backend image receiving and display: covered by Task 6.
- YOLO detection: covered by Task 5 and integrated in Task 6.
- Classroom sleeping anomaly rule: covered by Task 4 and integrated in Task 6.
- Alarm display and saved records: covered by Task 6.
- Stage and acceptance evidence: covered by Task 9.
- Report-ready protocol and test explanation: covered by README, protocol tests, analyzer tests, and `docs/course-evidence/checklist.md`.

### Completion Scan

The plan contains complete code blocks and exact file paths for every implementation step.

### Type Consistency

- `Detection` and `AlarmState` are defined in `src/common/types.py`.
- `YoloDetector.detect()` returns `list[Detection]`.
- `SleepAnalyzer.update()` accepts `list[Detection]` and returns `AlarmState`.
- `draw_overlay()` consumes the same `Detection` and `AlarmState` types.
- TCP packet functions use `frame_id`, `timestamp_ms`, and `image_bytes` consistently across frontend and backend.
