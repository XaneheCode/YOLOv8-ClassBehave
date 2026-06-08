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
