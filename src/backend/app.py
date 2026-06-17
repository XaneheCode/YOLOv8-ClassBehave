from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from src.backend.behaviour_analyzer import LABEL_DISPLAY_NAMES, BehaviourAnalyzer, display_label
from src.backend.detector import YoloDetector
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet
from src.common.types import AlarmState, DetectionAssessment


DEFAULT_MODEL_PATH = "models/merged_classroom_6cls_v2_img960_e50_2026-06-13_best.pt"
DENSE_OVERLAY_TARGET_THRESHOLD = 12
LABEL_ABBREVIATIONS = {
    "Hand-raise": "Hand",
    "Reading": "Learn",
    "Writing": "Learn",
    "Useing-Phone": "Phone",
    "Head-down": "Head",
    "Sleeping": "Sleep",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend receiver and YOLO alarm display")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--alarm-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", default="output/alarms")
    return parser


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


def _assessment_color(assessment: DetectionAssessment) -> tuple[int, int, int]:
    if assessment.status == "ignored":
        return (120, 120, 120)
    return (0, 0, 255) if assessment.is_abnormal else (0, 180, 0)


def _short_detection_label(assessment: DetectionAssessment) -> str:
    label = LABEL_ABBREVIATIONS.get(assessment.detection.label, assessment.detection.label)
    return f"{label} {assessment.detection.confidence:.2f}"


def _compact_status_text(alarm: AlarmState) -> str:
    if alarm.is_alarm:
        status = "ALARM"
    elif alarm.suspicious:
        status = "SUSPICIOUS"
    else:
        status = "NORMAL"

    labels = list(alarm.abnormal_labels)
    if len(labels) > 2:
        label_text = f"{', '.join(labels[:2])} +{len(labels) - 2}"
    else:
        label_text = ", ".join(labels)

    if label_text:
        return f"{status} abnormal={alarm.abnormal_count} {label_text}"
    if alarm.suspicious or alarm.is_alarm:
        return f"{status} abnormal={alarm.abnormal_count}"
    return status


def _draw_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 1,
    background: tuple[int, int, int] = (20, 20, 20),
) -> None:
    x, y = origin
    height, width = image.shape[:2]
    padding = 4
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, min(x, max(0, width - text_width - padding * 2)))
    y = max(text_height + padding, min(y, height - padding - baseline))
    top_left = (x, y - text_height - padding)
    bottom_right = (min(width - 1, x + text_width + padding * 2), min(height - 1, y + baseline + padding))
    cv2.rectangle(image, top_left, bottom_right, background, -1)
    cv2.putText(image, text, (x + padding, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    assessments: list[DetectionAssessment],
    alarm: AlarmState,
    fps: float,
    latency_ms: int,
) -> np.ndarray:
    output = frame.copy()
    visible_assessments = [assessment for assessment in assessments if assessment.status != "ignored"]
    dense_mode = len(visible_assessments) > DENSE_OVERLAY_TARGET_THRESHOLD
    badge_number = 1

    for assessment in assessments:
        if assessment.status == "ignored":
            continue

        detection = assessment.detection
        x1, y1, x2, y2 = detection.bbox
        color = _assessment_color(assessment)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        if dense_mode:
            label = str(badge_number)
            badge_number += 1
            _draw_text_with_background(output, label, (x1, max(18, y1 - 4)), (255, 255, 255), 0.45, 1, color)
        else:
            label = _short_detection_label(assessment)
            _draw_text_with_background(output, label, (x1, max(18, y1 - 4)), color, 0.5, 1)

    status = _compact_status_text(alarm)
    status_color = (0, 0, 255) if alarm.is_alarm else (0, 180, 0)
    _draw_text_with_background(output, status, (20, 30), status_color, 0.75, 2)
    _draw_text_with_background(
        output,
        f"fps={fps:.1f} latency={latency_ms}ms reason={alarm.reason}",
        (20, 60),
        (255, 255, 255),
        0.5,
        1,
    )
    return output


def append_alarm(
    csv_path: Path,
    frame_id: int,
    timestamp_ms: int,
    reason: str,
    duration: float,
    abnormal_count: int,
    abnormal_labels: tuple[str, ...],
    image_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "frame_id",
                    "timestamp_ms",
                    "reason",
                    "duration_seconds",
                    "abnormal_count",
                    "abnormal_labels",
                    "image_path",
                ]
            )
        writer.writerow(
            [
                frame_id,
                timestamp_ms,
                reason,
                f"{duration:.2f}",
                abnormal_count,
                "|".join(abnormal_labels),
                str(image_path),
            ]
        )


def run_backend(host: str, port: int, model: str, alarm_seconds: float, output_dir: Path) -> None:
    detector = YoloDetector(model_path=model)
    analyzer = BehaviourAnalyzer(threshold_seconds=alarm_seconds)
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
                assessments, alarm = analyzer.update(detections, now_seconds=now)

                frame_count += 1
                elapsed = max(0.001, now - fps_started)
                fps = frame_count / elapsed
                overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=latency_ms)

                if alarm.is_alarm and packet.frame_id != last_alarm_frame:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = output_dir / f"alarm_{packet.frame_id}_{packet.timestamp_ms}.jpg"
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
