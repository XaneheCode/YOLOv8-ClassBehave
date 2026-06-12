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
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640) -> None:
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
