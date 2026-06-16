import sys
import types

import numpy as np

from src.backend.detector import DEFAULT_INFERENCE_SIZE, YoloDetector, result_to_detections


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


def test_result_to_detections_can_keep_only_person_labels():
    detections = result_to_detections(FakeResult(), allowed_labels={"person"})

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].bbox == (1, 3, 30, 51)


def test_yolo_detector_uses_configured_default_inference_size(monkeypatch):
    class FakeYOLO:
        instances = []

        def __init__(self, model_path):
            self.model_path = model_path
            self.calls = []
            FakeYOLO.instances.append(self)

        def __call__(self, frame, **kwargs):
            self.calls.append(kwargs)
            return [types.SimpleNamespace(names={}, boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    detector = YoloDetector(model_path="fake.pt")
    detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))

    assert detector.imgsz == DEFAULT_INFERENCE_SIZE
    assert FakeYOLO.instances[0].calls[0]["imgsz"] == DEFAULT_INFERENCE_SIZE
    assert FakeYOLO.instances[0].calls[0]["conf"] == 0.25


def test_yolo_detector_applies_person_label_filter(monkeypatch):
    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def __call__(self, frame, **kwargs):
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    detector = YoloDetector(model_path="yolov8s.pt", allowed_labels={"person"})
    detections = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))

    assert [detection.label for detection in detections] == ["person"]
