from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from src.backend.behaviour_analyzer import BehaviourAnalyzer
from src.backend.detector import YoloDetector
from src.common.image_codec import decode_jpeg
from src.common.protocol import recv_packet
from src.common.types import AlarmState, DetectionAssessment


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend receiver and YOLO alarm display")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default="output/training/student_behaviour_yolov8n_e20/weights/best.pt")
    parser.add_argument("--alarm-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", default="output/alarms")
    return parser


def draw_overlay(
    frame: np.ndarray,
    assessments: list[DetectionAssessment],
    alarm: AlarmState,
    fps: float,
    latency_ms: int,
) -> np.ndarray:
    output = frame.copy()
    for assessment in assessments:
        detection = assessment.detection
        x1, y1, x2, y2 = detection.bbox
        if assessment.status == "ignored":
            color = (120, 120, 120)
        else:
            color = (0, 0, 255) if assessment.is_abnormal else (0, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {assessment.status} {detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    labels = ", ".join(alarm.abnormal_labels)
    if alarm.is_alarm:
        status = f"ALARM: {alarm.abnormal_count} abnormal"
        if labels:
            status = f"{status} - {labels}"
    elif alarm.suspicious:
        status = f"suspicious: {alarm.abnormal_count} abnormal"
        if labels:
            status = f"{status} - {labels}"
    else:
        status = "normal"
    status_color = (0, 0, 255) if alarm.is_alarm else (0, 180, 0)
    cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(
        output,
        f"fps={fps:.1f} latency={latency_ms}ms reason={alarm.reason}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
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
