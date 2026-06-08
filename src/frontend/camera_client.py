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
