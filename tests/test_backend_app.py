import csv

import numpy as np

from src.backend.app import append_alarm, behaviour_counts, build_arg_parser, draw_overlay, frame_status_text
from src.common.types import AlarmState, Detection, DetectionAssessment


def test_backend_parser_defaults_to_yolov8_six_class_model():
    args = build_arg_parser().parse_args([])

    assert args.model == "models/student_behaviour_v6_6cls_img960_e50_best.pt"


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


def test_append_alarm_creates_csv_with_header(tmp_path):
    csv_path = tmp_path / "alarms.csv"
    image_path = tmp_path / "alarm.jpg"

    append_alarm(
        csv_path=csv_path,
        frame_id=12,
        timestamp_ms=123456,
        reason="multi_behaviour_abnormal",
        duration=3.25,
        abnormal_count=2,
        abnormal_labels=("Sleeping", "Useing-Phone"),
        image_path=image_path,
    )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "frame_id",
        "timestamp_ms",
        "reason",
        "duration_seconds",
        "abnormal_count",
        "abnormal_labels",
        "image_path",
    ]
    assert rows[1] == [
        "12",
        "123456",
        "multi_behaviour_abnormal",
        "3.25",
        "2",
        "Sleeping|Useing-Phone",
        str(image_path),
    ]


def test_draw_overlay_keeps_frame_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assessments = [
        DetectionAssessment(
            detection=Detection(label="Sleeping", confidence=0.8, bbox=(10, 15, 80, 100)),
            status="abnormal",
            is_abnormal=True,
            is_alarm=True,
            reason="Sleeping",
            duration_seconds=3.1,
        )
    ]
    alarm = AlarmState(
        is_alarm=True,
        suspicious=True,
        duration_seconds=3.1,
        reason="multi_behaviour_abnormal",
        abnormal_count=1,
        abnormal_labels=("Sleeping",),
    )

    output = draw_overlay(frame, assessments, alarm, fps=8.5, latency_ms=32)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert np.any(output != frame)


def test_draw_overlay_colours_each_detection_by_its_own_status():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
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
            detection=Detection(label="Hand-raise", confidence=0.88, bbox=(70, 10, 120, 100)),
            status="normal",
            is_abnormal=False,
            is_alarm=False,
            reason="Hand-raise",
            duration_seconds=0.0,
        ),
    ]
    alarm = AlarmState(
        is_alarm=True,
        suspicious=True,
        duration_seconds=3.1,
        reason="multi_behaviour_abnormal",
        abnormal_count=1,
        abnormal_labels=("Sleeping",),
    )

    output = draw_overlay(frame, assessments, alarm, fps=8.5, latency_ms=32)

    assert tuple(output[10, 10]) == (0, 0, 255)
    assert tuple(output[10, 70]) == (0, 180, 0)
