import csv

import numpy as np

from src.backend.app import append_alarm, draw_overlay
from src.common.types import AlarmState, Detection


def test_append_alarm_creates_csv_with_header(tmp_path):
    csv_path = tmp_path / "alarms.csv"
    image_path = tmp_path / "alarm.jpg"

    append_alarm(
        csv_path=csv_path,
        frame_id=12,
        timestamp_ms=123456,
        reason="sleep_label",
        duration=3.25,
        image_path=image_path,
    )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["frame_id", "timestamp_ms", "reason", "duration_seconds", "image_path"]
    assert rows[1] == ["12", "123456", "sleep_label", "3.25", str(image_path)]


def test_draw_overlay_keeps_frame_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = [Detection(label="person", confidence=0.8, bbox=(10, 15, 80, 100))]
    alarm = AlarmState(is_alarm=True, suspicious=True, duration_seconds=3.1, reason="sleep_label")

    output = draw_overlay(frame, detections, alarm, fps=8.5, latency_ms=32)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.any(output != frame)
